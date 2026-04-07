import os, logging, sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset, load_from_disk

from collator import CrossDataCollator

import transformers
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments, BatchSamplers

from transformers import HfArgumentParser

from loss import CachedCLEARLoss

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="BAAI/bge-m3",
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    low_cpu_mem_usage: bool = field(
        default=True,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "For Jina"
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_path: Optional[str] = field(
        default="/mnt/raid6/dltmddbs100/data/cross/MLQA/train_hn/ar_en_paired_train_5HN_50-100_paired_train_formatted.jsonl", 
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    use_hf_dataset: bool = field(
        default=False,
        metadata={
            "help": (
                "If huggingface dataset or not"
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=32,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    eval_ratio: Optional[float] = field(
        default=0.1,
        metadata={"help": "eval_ratio"},
    )
    ds_config_path: Optional[str] = field(    
        default=None,
        metadata={"help": "Deepspeed config path"},
    )
    max_seq_length: Optional[int] = field(    
        default=512,
        metadata={"help": "max_seq_length"},
    )
    mini_batch_size: Optional[int] = field(    
        default=32,
        metadata={"help": "mini_batch_size"},
    )
    alpha: Optional[float] = field(    
        default=0.4,
        metadata={"help": "alpha"},
    )
    beta: Optional[float] = field(    
        default=0.2,
        metadata={"help": "beta"},
    )
    kl_div: bool = field(    
        default=False,
        metadata={"help": "Use KL or Not"},
    )
    num_negative: Optional[int] = field(    
        default=5,
        metadata={"help": "num_negative"},
    )
    loss_name: Optional[str] = field(    
        default='CachedMultipleNegativesCrossSymmetricRankingLoss',
        metadata={"help": "Loss name"},
    )

def _setup_logger():
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    return logger

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SentenceTransformerTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if not os.path.isdir(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)

    logger = _setup_logger()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


    # load dataset
    logger.info("Loading train_dataset...")
    if data_args.dataset_path is not None:
        if data_args.use_hf_dataset:
            train_dataset = load_from_disk(data_args.dataset_path)
        else:
            train_dataset = load_dataset('json', data_files=data_args.dataset_path)

    logger.info("Finished loading train dataset!")


    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    data_collator = CrossDataCollator(tokenizer=tokenizer)

    if "jina" in model_args.model_name_or_path:
        model_kwargs =  {"default_task": "retrieval.passage", "lora_main_params_trainable": True}
    else:
        model_kwargs = {"attn_implementation": "sdpa"}

    # set loss function
    if 'Cached' in data_args.loss_name:
        model = SentenceTransformer(model_args.model_name_or_path, model_kwargs=model_kwargs, trust_remote_code=model_args.trust_remote_code)
        model.max_seq_length = data_args.max_seq_length
        loss = CachedCLEARLoss(
            model=model, alpha=data_args.alpha, beta=data_args.beta, kl_div=data_args.kl_div, 
            num_negative=data_args.num_negative,
            mini_batch_size=data_args.mini_batch_size, 
        )
        logger.info(f"CachedCLEARLoss(alpha={data_args.alpha}, beta={data_args.beta}, kl_div={data_args.kl_div}) employed")
            
    else:
        model = SentenceTransformer(model_args.model_name_or_path, model_kwargs=model_kwargs, trust_remote_code=model_args.trust_remote_code)
        model.max_seq_length = data_args.max_seq_length
        loss = MultipleNegativesCrossSymmetricRankingLoss(
                model=model, alpha=data_args.alpha, num_negative=data_args.num_negative, 
            )
        logger.info("MultipleNegativesCrossSymmetricRankingLoss employed")


    if "jina" not in model_args.model_name_or_path:
        training_args.batch_sampler = BatchSamplers.NO_DUPLICATES
    
    # set trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset["train"].select(range(0,500)),
        eval_dataset=None,
        data_collator=data_collator,
        loss=loss,
    )
    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # # Evaluation
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")

    #     metrics = trainer.evaluate()

    #     max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    #     metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)



if __name__ == "__main__":
    main()