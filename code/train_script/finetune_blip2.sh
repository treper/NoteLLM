EXP_NAME=${1}
export NCCL_DEBUG=info
export NCCL_SOCKET_IFNAME=bond0
export WANDB_PROJECT=NoteLLM
PER_DEVICE_TRAIN_BATCH_SIZE=16
PER_DEVICE_EVAL_BATCH_SIZE=16



torchrun --nproc_per_node=8 --master_port=17005 run.py \
    --model_type BLIP2_INI \
    --model_name_or_path ../../model/blip2_opt7b  \
    --image_path ../data/images/ \
    --data_path ../data/notes_train.csv \
    --data_cache_path ../data/${EXP_NAME}_notes_cache \
    --noteid2id_path ../data/${EXP_NAME}_notes_id.npy \
    --train_pairdata_path ../data/note_pairs_train.csv \
    --val_pairdata_path ../data/note_pairs_val.csv \
    --fp16 True \
    --output_dir ../log/$EXP_NAME/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 4 \
    --learning_rate 3e-6 \
    --dataloader_num_workers 32 \
    --weight_decay 0.001 \
    --warmup_ratio 0.1 \
    --logging_steps 10 \
    --report_to none \
    --run_name ${EXP_NAME} \
    --remove_unused_columns False \
    --model_max_length 140 \
    --evaluation_strategy "steps" \
    --eval_steps 1750 \
    --save_strategy "steps" \
    --save_steps 1750 \
    --save_total_limit 1 \
    --load_best_model_at_end True \
    --greater_is_better False \
    --only_image False \
    --fix_text_encoder False \
    --fix_text_linear False \
    --fix_image_encoder True \
    --fix_connector_encoder False \
    --mm_hidden_size 1408 \
    --dim_reduction 64 \
    --deepspeed "ds_config/default_offload_opt_param.json" 