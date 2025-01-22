EXP_NAME=${1}
TYPE_MODEL=${2}
export NCCL_DEBUG=info
export NCCL_SOCKET_IFNAME=bond0
export WANDB_PROJECT=NoteLLM_infer
PER_DEVICE_EVAL_BATCH_SIZE=50
STEP='last_onlytext'
SUFFIX=''
SAVENAME=logits_${STEP}.pickle
NOTEID_CACHE=../log/${EXP_NAME}/test_note_id.npy
SEED=42

MODEL_PATH=../log/${EXP_NAME}${SUFFIX}
#python ../log/${EXP_NAME}${SUFFIX}/zero_to_fp32.py ../log/${EXP_NAME}${SUFFIX}/ ../log/${EXP_NAME}${SUFFIX}/pytorch_model.bin
DIM_REDUCTION=64

torchrun --nproc_per_node=8 --master_port=17005 inference.py \
    --model_type ${TYPE_MODEL} \
    --image_path ../data/images_test \
    --model_name_or_path ${MODEL_PATH} \
    --save_path ../log/${EXP_NAME}/${SAVENAME} \
    --output_dir ../log/$EXP_NAME/ \
    --data_path ../data/notes_test.csv \
    --data_cache_path ../log/${EXP_NAME}/notes_test_onlytext_cache \
    --noteid2id_path ${NOTEID_CACHE} \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --remove_unused_columns False \
    --model_max_length 140 \
    --dataloader_num_workers 32 \
    --bf16 True \
    --eval_accumulation_steps 300 \
    --mm_vision_select_layer -1 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_feature 'cls_patch' \
    --late_fusion False \
    --mICL False \
    --dim_reduction ${DIM_REDUCTION} \
    --clip_path "../../model/ViT-bigG" \
    --mm_hidden_size 1664 \
    --query_hidden_size 1536 \
    --only_image False \
    --only_text True \
    --seed ${SEED}



python eval_recall.py \
 ../log/${EXP_NAME}/${SAVENAME} \
 ${NOTEID_CACHE} \
 ../data/note_pairs_test.csv \
 logits \
 ${SEED} 2>&1 | tee ../log/${EXP_NAME}/results_${STEP}_allpair.txt

 python eval_recall.py \
 ../log/${EXP_NAME}/${SAVENAME} \
 ${NOTEID_CACHE} \
 ../data/note_pairs_test_shortquery.csv \
 logits \
 ${SEED} 2>&1 | tee ../log/${EXP_NAME}/results_${STEP}_shortquery.txt

 python eval_recall.py \
 ../log/${EXP_NAME}/${SAVENAME} \
 ${NOTEID_CACHE} \
 ../data/note_pairs_test_shortgtdoc.csv \
 logits \
 ${SEED} 2>&1 | tee ../log/${EXP_NAME}/results_${STEP}_shortgtdoc.txt