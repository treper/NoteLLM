

##########zero shot experiments##########

EXP_NAME=Llama_zero_shot
./test_script/inference_LLM.sh ${EXP_NAME} Llama ../../model/Llama-2-7B-fp16

EXP_NAME=blip2_zeroshot
./test_script/inference_blip2.sh ${EXP_NAME} ../../model/blip2_opt7b
./test_script/inference_blip2_onlyimage.sh ${EXP_NAME} ../../model/blip2_opt7b
./test_script/inference_blip2_onlytext.sh ${EXP_NAME} ../../model/blip2_opt7b

EXP_NAME=llava1p5_zeroshot
./test_script/inference_llava.sh ${EXP_NAME} ../../model/llava-1.5-7b-hf
./test_script/inference_llava_onlyimage.sh ${EXP_NAME} ../../model/llava-1.5-7b-hf
./test_script/inference_llava_onlytext.sh ${EXP_NAME} ../../model/llava-1.5-7b-hf

EXP_NAME=qwvl_zeroshot
./test_script/inference_qwvl.sh ${EXP_NAME} ../../model/qw_vl/
./test_script/inference_qwvl_onlyimage.sh ${EXP_NAME} ../../model/qw_vl/
./test_script/inference_qwvl_onlytext.sh ${EXP_NAME} ../../model/qw_vl/

EXP_NAME=qwvl_chat_zeroshot
./test_script/inference_qwvl.sh ${EXP_NAME} ../../model/qw_vl_chat/
./test_script/inference_qwvl_onlyimage.sh ${EXP_NAME} ../../model/qw_vl_chat/
./test_script/inference_qwvl_onlytext.sh ${EXP_NAME} ../../model/qw_vl_chat/

EXP_NAME=llama2_zero_shot
./test_script/inference_LLM.sh ${EXP_NAME} Llama ../../model/Llama-2-7B-fp16

EXP_NAME=qwen_chat_zero_shot
./test_script/inference_LLM.sh ${EXP_NAME} Qwen ../../model/Qwen-7B-Chat


##########Text LLM experiments##########

EXP_NAME=finetune_qwen
./train_script/finetune_LLM.sh ${EXP_NAME} Qwen ../../model/Qwen-7B-Chat
./test_script/inference_LLM.sh ${EXP_NAME} Qwen

EXP_NAME=finetune_Llama
./train_script/finetune_LLM.sh ${EXP_NAME} Llama ../../model/Llama-2-7B-fp16
./test_script/inference_LLM.sh ${EXP_NAME} Llama

EXP_NAME=finetune_Llama_Notellm
./train_script/finetune_NoteLLM.sh ${EXP_NAME} LlamaNoteLLM ../../model/Llama-2-7B-fp16
./test_script/inference_NoteLLM.sh ${EXP_NAME} LlamaNoteLLM
./test_script/inference_NoteLLM_generation.sh ${EXP_NAME} LlamaNoteLLM

##########finetuned MLLM experiments##########

EXP_NAME=finetune_multimodal_blip2
./train_script/finetune_blip2.sh ${EXP_NAME}
./test_script/inference_blip2.sh ${EXP_NAME}
./test_script/inference_blip2_onlyimage.sh ${EXP_NAME}
./test_script/inference_blip2_onlytext.sh ${EXP_NAME}

EXP_NAME=finetune_multimodal_qwenvl
./train_script/finetune_qwen_vl.sh ${EXP_NAME}
./test_script/inference_qwvl.sh ${EXP_NAME}
./test_script/inference_qwvl_onlyimage.sh ${EXP_NAME}
./test_script/inference_qwvl_onlytext.sh ${EXP_NAME}

##########finetuned MLlama-Base experiments##########

EXP_NAME=finetune_multimodal_Llama_onlylatefusion
./train_script/finetune_latefusion.sh ${EXP_NAME} LateFusion ../../model/Llama-2-7B-fp16 ../../pretrain/clip-vit-base-patch16
./test_script/inference_latefusion.sh ${EXP_NAME} LateFusion

EXP_NAME=finetune_multimodal_Llama_lateearlyfusion
./train_script/finetune_lateearlyfusion.sh ${EXP_NAME} MLlama ../../model/Llama-2-7B-fp16 ../../pretrain/clip-vit-base-patch16
./test_script/inference_lateearlyfusion.sh ${EXP_NAME} MLlama

EXP_NAME=finetune_multimodal_Llama
./train_script/finetune_basic.sh ${EXP_NAME} MLlama ../../model/Llama-2-7B-fp16 ../../pretrain/clip-vit-base-patch16
./test_script/inference_basic.sh ${EXP_NAME} MLlama
./test_script/inference_basic_onlyimage.sh ${EXP_NAME} MLlama
./test_script/inference_basic_onlytext.sh ${EXP_NAME} MLlama

EXP_NAME=finetune_multimodal_Llama_mICL_tv9_1
./train_script/finetune_mICL.sh ${EXP_NAME} MLlama ../../model/Llama-2-7B-fp16 ../../pretrain/clip-vit-base-patch16 9 1
./test_script/inference_mICL.sh ${EXP_NAME} MLlama

EXP_NAME=finetune_multimodal_Llama_mICL_latefusion_tv9_1_vtoken16 #NoteLLM-2
./train_script/finetune_mICL_latefusion.sh ${EXP_NAME} MLlama ../../model/Llama-2-7B-fp16 ../../pretrain/clip-vit-base-patch16 9 1 16
./test_script/inference_mICL_latefusion.sh ${EXP_NAME} MLlama 16

EXP_NAME=finetune_multimodal_Llama_mICL_latefusion_tv1_1_vtoken16
./train_script/finetune_mICL_latefusion.sh ${EXP_NAME} MLlama ../../model/Llama-2-7B-fp16 ../../pretrain/clip-vit-base-patch16 1 1 16
./test_script/inference_mICL_latefusion.sh ${EXP_NAME} MLlama 16

EXP_NAME=finetune_multimodal_Llama_mICL_latefusion_tv3_1_vtoken16
./train_script/finetune_mICL_latefusion.sh ${EXP_NAME} MLlama ../../model/Llama-2-7B-fp16 ../../pretrain/clip-vit-base-patch16 3 1 16
./test_script/inference_mICL_latefusion.sh ${EXP_NAME} MLlama 16

EXP_NAME=finetune_multimodal_Llama_mICL_latefusion_tv19_1_vtoken16
./train_script/finetune_mICL_latefusion.sh ${EXP_NAME} MLlama ../../model/Llama-2-7B-fp16 ../../pretrain/clip-vit-base-patch16 19 1 16
./test_script/inference_mICL_latefusion.sh ${EXP_NAME} MLlama 16

EXP_NAME=finetune_multimodal_Llama_mICL_latefusion_tv9_1_vtoken8
./train_script/finetune_mICL_latefusion.sh ${EXP_NAME} MLlama ../../model/Llama-2-7B-fp16 ../../pretrain/clip-vit-base-patch16 9 1 8
./test_script/inference_mICL_latefusion.sh ${EXP_NAME} MLlama 8

EXP_NAME=finetune_multimodal_Llama_mICL_latefusion_tv9_1_vtoken32
./train_script/finetune_mICL_latefusion.sh ${EXP_NAME} MLlama ../../model/Llama-2-7B-fp16 ../../pretrain/clip-vit-base-patch16 9 1 32
./test_script/inference_mICL_latefusion.sh ${EXP_NAME} MLlama 32

EXP_NAME=finetune_multimodal_Llama_mICL_latefusion_tv9_1_vtoken48
./train_script/finetune_mICL_latefusion.sh ${EXP_NAME} MLlama ../../model/Llama-2-7B-fp16 ../../pretrain/clip-vit-base-patch16 9 1 48
./test_script/inference_mICL_latefusion.sh ${EXP_NAME} MLlama 48

##########finetuned MQwen-Base experiments##########

EXP_NAME=finetune_multimodal_qwen_selfvl_onlylatefusion
./train_script/finetune_latefusion.sh ${EXP_NAME} Qwen_LateFusion ../../model/Qwen-7B-Chat ../../pretrain/clip-vit-base-patch16
./test_script/inference_latefusion.sh ${EXP_NAME} Qwen_LateFusion

EXP_NAME=finetune_multimodal_qwen_selfvl_lateearlyfusion
./train_script/finetune_lateearlyfusion.sh ${EXP_NAME} Qwen_selfvl ../../model/Qwen-7B-Chat ../../pretrain/clip-vit-base-patch16
./test_script/inference_lateearlyfusion.sh ${EXP_NAME} Qwen_selfvl

EXP_NAME=finetune_multimodal_qwen_selfvl
./train_script/finetune_basic.sh ${EXP_NAME} Qwen_selfvl ../../model/Qwen-7B-Chat ../../pretrain/clip-vit-base-patch16
./test_script/inference_basic.sh ${EXP_NAME} Qwen_selfvl
./test_script/inference_basic_onlyimage.sh ${EXP_NAME} Qwen_selfvl
./test_script/inference_basic_onlytext.sh ${EXP_NAME} Qwen_selfvl

EXP_NAME=finetune_multimodal_qwen_selfvl_mICL_tv9_1
./train_script/finetune_mICL.sh ${EXP_NAME} Qwen_selfvl ../../model/Qwen-7B-Chat ../../pretrain/clip-vit-base-patch16 9 1
./test_script/inference_mICL.sh ${EXP_NAME} Qwen_selfvl

EXP_NAME=finetune_multimodal_qwen_selfvl_mICL_latefusion_tv9_1_vtoken16 #NoteLLM-2
./train_script/finetune_mICL_latefusion.sh ${EXP_NAME} Qwen_selfvl ../../model/Qwen-7B-Chat ../../pretrain/clip-vit-base-patch16 9 1 16
./test_script/inference_mICL_latefusion.sh ${EXP_NAME} Qwen_selfvl 16

##########finetuned MQwen-bigG experiments##########

EXP_NAME=finetune_multimodal_qwen_selfvl_largevision_onlylatefusion
./train_script/finetune_largevision_latefusion.sh ${EXP_NAME} Qwen_LateFusion ../../model/Qwen-7B-Chat ../../model/ViT-bigG
./test_script/inference_largevision_latefusion.sh ${EXP_NAME} Qwen_LateFusion

EXP_NAME=finetune_multimodal_qwen_selfvl_largevision_lateearlyfusion
./train_script/finetune_largevision_lateearlyfusion.sh ${EXP_NAME} Qwen_selfvl ../../model/Qwen-7B-Chat ../../model/ViT-bigG
./test_script/inference_largevision_lateearlyfusion.sh ${EXP_NAME} Qwen_selfvl

EXP_NAME=finetune_multimodal_qwen_selfvl_largevision
./train_script/finetune_largevision_basic.sh ${EXP_NAME} Qwen_selfvl ../../model/Qwen-7B-Chat ../../model/ViT-bigG
./test_script/inference_largevision_basic.sh ${EXP_NAME} Qwen_selfvl
./test_script/inference_largevision_basic_onlyimage.sh ${EXP_NAME} Qwen_selfvl
./test_script/inference_largevision_basic_onlytext.sh ${EXP_NAME} Qwen_selfvl

EXP_NAME=finetune_multimodal_qwen_selfvl_largevision_mICL_tv9_1
./train_script/finetune_largevision_mICL.sh ${EXP_NAME} Qwen_selfvl ../../model/Qwen-7B-Chat ../../model/ViT-bigG 9 1
./test_script/inference_largevision_mICL.sh ${EXP_NAME} Qwen_selfvl

EXP_NAME=finetune_multimodal_qwen_selfvl_largevision_mICL_latefusion_tv9_1_vtoken16 #NoteLLM-2
./train_script/finetune_largevision_mICL_latefusion.sh ${EXP_NAME} Qwen_selfvl ../../model/Qwen-7B-Chat ../../model/ViT-bigG 9 1 16
./test_script/inference_largevision_mICL_latefusion.sh ${EXP_NAME} Qwen_selfvl 16


