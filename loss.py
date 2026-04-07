from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from functools import partial
from typing import Any

import torch
import tqdm
from torch import Tensor, nn
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer, util
from sentence_transformers.losses.CachedMultipleNegativesRankingLoss import RandContext
from sentence_transformers.models import StaticEmbedding


class CLEARLoss(nn.Module):
    def __init__(
        self, 
        model: SentenceTransformer, 
        alpha: float = 0.4, 
        beta: float = 0.2,
        kl_div: bool = False,
        num_negative: int = 5, 
        scale: float = 20.0, 
        similarity_fct=util.cos_sim
    ) -> None:
        """
        Given a list of (anchor, positive) pairs, this loss sums the following two losses:

        1. Forward loss: Given an anchor, find the sample with the highest similarity out of all positives in the batch.
        2. Backward loss: Given a positive, find the sample with the highest similarity out of all anchors in the batch.
        3. KL Divergence: Aligns the probability distributions of the forward and backward tasks.

        Args:
            model: SentenceTransformer model
            alpha: Weight factor for the cross backward loss
            beta: Weight factor for the KL divergence loss
            kl_div: If True, calculates KL divergence between forward and backward scores
            num_negative: Number of negative examples for the cross query
            scale: Output of similarity function is multiplied by scale value
            similarity_fct: similarity function between sentence embeddings. By default, cos_sim.
        """
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.kl_div = kl_div
        self.n_neg = num_negative
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    # anchor, positive, negative: (q+, d+, d-) + (d+, q+, q-)
    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # Compute the embeddings and distribute them to anchor and candidates
        reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        
        anchor = reps[0] # query positive
        origin_candidates = torch.cat(reps[1:-(self.n_neg+1)]) # d+:d-

        cross_anchor = reps[1] # doc positive
        cross_candidates = torch.cat(reps[-(self.n_neg+1):]) # q+:q-

        # Compute origin scores
        origin_scores = self.similarity_fct(anchor, origin_candidates) * self.scale

        # anchor[i] should be most similar to candidates[i], as that is the paired positive
        range_labels = torch.arange(
            anchor.size(0), dtype=torch.long, device=origin_scores.device
        )

        # Compute cross scores
        cross_scores = self.similarity_fct(cross_anchor, cross_candidates) * self.scale

        origin_forward_loss = self.cross_entropy_loss(origin_scores, range_labels)
        cross_backward_loss = self.cross_entropy_loss(cross_scores, range_labels)

        if self.kl_div:
            # KL Divergence  (Origin_scores and cross_scores shape should be the same)
            origin_scores_log_softmax = F.log_softmax(origin_scores, dim=1)
            cross_scores_softmax = F.softmax(cross_scores, dim=1)
            kl_div_loss = F.kl_div(origin_scores_log_softmax, cross_scores_softmax, reduction='batchmean')

            # Final Loss
            loss = (1 - self.alpha - self.beta) * origin_forward_loss + self.alpha * cross_backward_loss + self.beta * kl_div_loss
        else:
            loss = (1 - self.alpha) * origin_forward_loss + self.alpha * cross_backward_loss

        return loss
    
    def get_config_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "kl_div": self.kl_div,
            "num_negative": self.n_neg,
            "scale": self.scale,
            "similarity_fct": self.similarity_fct.__name__
        }

###################################### Cached Version ######################################

def _backward_hook(
    grad_output: Tensor,
    sentence_features: Iterable[dict[str, Tensor]],
    loss_obj: CachedCLEARLoss,
) -> None:
    """A backward hook to backpropagate the cached gradients mini-batch by mini-batch."""
    assert loss_obj.cache is not None
    assert loss_obj.random_states is not None
    with torch.enable_grad():
        for sentence_feature, grad, random_states in zip(sentence_features, loss_obj.cache, loss_obj.random_states):
            for (reps_mb, _), grad_mb in zip(
                loss_obj.embed_minibatch_iter(
                    sentence_feature=sentence_feature,
                    with_grad=True,
                    copy_random_state=False,
                    random_states=random_states,
                ),
                grad,
            ):
                surrogate = torch.dot(reps_mb.flatten(), grad_mb.flatten()) * grad_output
                surrogate.backward()



class CachedCLEARLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        alpha: float = 0.4,
        beta: float = 0.2,
        kl_div: bool = False,
        num_negative: int = 5,
        scale: float = 20.0,
        similarity_fct: callable[[Tensor, Tensor], Tensor] = util.cos_sim,
        mini_batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> None:
        """
        Boosted version of :class:`MultipleNegativesCrossSymmetricRankingLoss` by GradCache (https://arxiv.org/pdf/2101.06983.pdf).

        Given a list of (anchor, positive) pairs, this loss sums cross and symmetric ranking losses between pairs of positive and negative sentences.
        This version allows for large batch sizes for better training signal with constant memory usage.

        The caching mechanism solves memory constraints while maintaining training signal accuracy.

        Args:
            model: SentenceTransformer model
            alpha: Weight factor between original and cross backward losses
            num_negative: Number of negative examples
            scale: Output of similarity function is multiplied by scale value
            similarity_fct: similarity function between sentence embeddings. By default, cos_sim.
            mini_batch_size: Mini-batch size for the forward pass, this denotes how much memory is actually used during
                training and evaluation. The larger the mini-batch size, the more memory efficient the training is, but
                the slower the training will be.
            show_progress_bar: If True, shows progress bar during processing

        Requirements:
            1. (anchor, positive, negative_passage, cross_anchor, negative_query) pairs.

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | (anchor, positive, negative_p1, ... ,negative_pn, cross_anchor, negative_q1, ... ,negative_qn) pairs   | none   |
            +---------------------------------------+--------+


        Recommendations:
            - Use ``BatchSamplers.NO_DUPLICATES`` (:class:`docs <sentence_transformers.training_args.BatchSamplers>`) to
              ensure that no in-batch negatives are duplicates of the anchor or positive samples.

        Relations:
            - Like :class:`MultipleNegativesRankingLoss`, but with an additional loss term.
            - :class:`CachedMultipleNegativesSymmetricRankingLoss` is equivalent to this loss, but it uses caching that
              allows for much higher batch sizes (and thus better performance) without extra memory usage. However, it
              is slightly slower.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."], # Query
                    "positive": ["It's so sunny.", "He took the car to the office."], # Doc
                    "negative_p1": ["Such a bad weather", "He didn't go to work."], # Doc
                    "cross_anchor": ["오늘 밖에 날씨가 좋네요.", "그는 자동차를 타고 일하러갔습니다."], # Cross query
                    "negative_q1" : ["오늘 안에 날씨가 덥네요", "그는 마차를 타고 집에 갔습니다"] # Cross query neg
                })
                loss = losses.CachedCLEARLoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        if isinstance(model[0], StaticEmbedding):
            raise ValueError(
                "CachedCLEARLoss is not compatible with a SentenceTransformer model based on a StaticEmbedding. "
                "Consider using CLEARLoss instead."
            )

        self.model = model
        self.alpha = alpha
        self.beta = beta
        self.kl_div = kl_div
        self.n_neg = num_negative
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mini_batch_size = mini_batch_size
        self.cache: list[list[Tensor]] | None = None
        self.random_states: list[list[RandContext]] | None = None
        self.show_progress_bar = show_progress_bar

    def embed_minibatch(
        self,
        sentence_feature: dict[str, Tensor],
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_state: RandContext | None = None,
    ) -> tuple[Tensor, RandContext | None]:
        """Embed a mini-batch of sentences."""
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        sentence_feature_minibatch = {k: v[begin:end] for k, v in sentence_feature.items()}
        with random_state_context:
            with grad_context():
                random_state = RandContext(*sentence_feature_minibatch.values()) if copy_random_state else None
                reps = self.model(sentence_feature_minibatch)["sentence_embedding"]
        return reps, random_state

    def embed_minibatch_iter(
        self,
        sentence_feature: dict[str, Tensor],
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext] | None = None,
    ) -> Iterator[tuple[Tensor, RandContext | None]]:
        """Iterate over mini-batches of sentences for embedding."""
        input_ids: Tensor = sentence_feature["input_ids"]
        bsz, _ = input_ids.shape
        for i, b in enumerate(
            tqdm.trange(
                0,
                bsz,
                self.mini_batch_size,
                desc="Embed mini-batches",
                disable=not self.show_progress_bar,
            )
        ):
            e = b + self.mini_batch_size
            reps, random_state = self.embed_minibatch(
                sentence_feature=sentence_feature,
                begin=b,
                end=e,
                with_grad=with_grad,
                copy_random_state=copy_random_state,
                random_state=None if random_states is None else random_states[i],
            )
            yield reps, random_state

    def calculate_loss_and_cache_gradients(self, reps: list[list[Tensor]]) -> Tensor:
        """Calculate the forward and cross backward losses and cache gradients."""
        loss = self.calculate_loss(reps, with_backward=True)
        loss = loss.detach().requires_grad_()

        self.cache = [[r.grad for r in rs] for rs in reps]  # e.g. 3 * bsz/mbsz * (mbsz, hdim)

        return loss

    def calculate_loss(self, reps: list[list[Tensor]], with_backward: bool = False) -> Tensor:
        """Calculate the forward and cross backward losses without caching gradients (for evaluation)."""
        anchor = torch.cat(reps[0])  # query positive
        origin_candidates = torch.cat([torch.cat(r) for r in reps[1:-(self.n_neg + 1)]])  # d+:d-

        cross_anchor = torch.cat(reps[1])  # doc positive
        cross_candidates = torch.cat([torch.cat(r) for r in reps[-(self.n_neg + 1):]])  # q+:q-

        batch_size = len(anchor)
        range_labels = torch.arange(batch_size, device=anchor.device)

        losses: list[torch.Tensor] = []
        for b in tqdm.trange(
            0,
            batch_size,
            self.mini_batch_size,
            desc="Calculating loss",
            disable=not self.show_progress_bar,
        ):
            e = min(b + self.mini_batch_size, batch_size)
            origin_scores = self.similarity_fct(anchor[b:e], origin_candidates) * self.scale
            origin_forward_loss = self.cross_entropy_loss(origin_scores, range_labels[b:e])

            cross_scores = self.similarity_fct(cross_anchor[b:e], cross_candidates) * self.scale
            cross_backward_loss = self.cross_entropy_loss(cross_scores, range_labels[b:e])

            if self.kl_div:
                # Calculate KL divergence between origin_scores and cross_scores
                origin_scores_log_softmax = F.log_softmax(origin_scores, dim=1)
                cross_scores_softmax = F.softmax(cross_scores, dim=1)
                kl_div_loss = F.kl_div(origin_scores_log_softmax, cross_scores_softmax, reduction='batchmean')

                # Combine all losses
                loss_mbatch = (1 - self.alpha - self.beta) * origin_forward_loss + self.alpha * cross_backward_loss + self.beta * kl_div_loss
            else:

                # Combine all losses
                loss_mbatch = (1 - self.alpha) * origin_forward_loss + self.alpha * cross_backward_loss

            if with_backward:
                loss_mbatch.backward()
                loss_mbatch = loss_mbatch.detach()
            losses.append(loss_mbatch)

        loss = sum(losses) / len(losses)
        return loss

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        """Forward pass of the loss function."""
        reps = []
        self.random_states = []
        for sentence_feature in sentence_features:
            reps_mbs = []
            random_state_mbs = []
            for reps_mb, random_state in self.embed_minibatch_iter(
                sentence_feature=sentence_feature,
                with_grad=False,
                copy_random_state=True,
            ):
                reps_mbs.append(reps_mb.detach().requires_grad_())
                random_state_mbs.append(random_state)
            reps.append(reps_mbs)
            self.random_states.append(random_state_mbs)

        if torch.is_grad_enabled():
            loss = self.calculate_loss_and_cache_gradients(reps)
            loss.register_hook(partial(_backward_hook, sentence_features=sentence_features, loss_obj=self))
        else:
            loss = self.calculate_loss(reps)

        return loss

    def get_config_dict(self) -> dict[str, Any]:
        """Get the configuration of the loss function."""
        return {
            "alpha": self.alpha,
            "num_negative": self.n_neg,
            "scale": self.scale,
            "similarity_fct": self.similarity_fct.__name__,
            "mini_batch_size": self.mini_batch_size,
        }
    
