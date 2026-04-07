export OMP_NUM_THREADS=32

DEVICES=4,5,6,7

LR=5e-5
ALPHA=0.4
BETA=0.2


BATCH_SIZE=64
MINI_BATCH_SIZE=32
NUM_GPU=4
# LANGS=('ar' 'de' 'es' 'hi' 'vi' 'zh' 'ru' 'te' 'bn')

lang='de'
BASE_MASTER_PORT=25055


RUN_NAME=bge-m3-${lang}

CUDA_VISIBLE_DEVICES=$DEVICES torchrun --nproc_per_node $NUM_GPU --master_port $BASE_MASTER_PORT train.py \
    --run_name $RUN_NAME \
    --model_name_or_path BAAI/bge-m3 \
    --dataset_path dataset/train_example_${lang} \
    --output_dir ckpt/bge-m3-${lang}-CLEAR \
    --loss_name CachedCLEARLoss \
    --report_to wandb \
    --alpha $ALPHA \
    --beta $BETA \
    --kl_div \
    --dataloader_num_workers 16 \
    --use_hf_dataset \
    --trust_remote_code \
    --max_steps 50 \
    --learning_rate $LR \
    --do_train \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --mini_batch_size $MINI_BATCH_SIZE \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --max_seq_length 512 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 1 \
    --bf16 \
    --logging_steps 1 > logs/$RUN_NAME.out
