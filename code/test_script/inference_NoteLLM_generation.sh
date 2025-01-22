EXP_NAME=${1}
MODEL_TYPE=${2}
MODEL_PATH=${3}

if [ -z "${MODEL_PATH}" ]; then
    MODEL_PATH=../log/${EXP_NAME}${SUFFIX}
fi

deepspeed --num_gpus 1 generation.py \
    --base_model_path ${MODEL_PATH} \
    --data_path ../data/notes_test.csv \
    --cache_path ../data/note_pairs_test_generation_category_cache \
    --save_path ../log/${EXP_NAME}/ \
    --type category \
    --max_length 240 \

deepspeed --num_gpus 1 generation.py \
    --base_model_path ${MODEL_PATH} \
    --data_path ../data/notes_test.csv \
    --cache_path ../data/note_pairs_test_generation_topic_cache \
    --save_path ../log/${EXP_NAME}/ \
    --type topic \
    --max_length 240 \

python eval_generation.py \
    --save_path ../log/${EXP_NAME}/ \
    --answer_path ../log/${EXP_NAME}/generation_category.csv \
    --type category 

python eval_generation.py \
    --save_path ../log/${EXP_NAME}/ \
    --answer_path ../log/${EXP_NAME}/generation_topic.csv \
    --type topic 