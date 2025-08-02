# -*- coding: utf-8 -*-
import sys
import os
import transformers
import pickle
import logging
import torch
from trainer import Inference
from transformers.integrations import WandbCallback
from data import smart_tokenizer_and_embedding_resize
from transformers import AutoModelForCausalLM, AutoTokenizer, Blip2Processor
from model.qwen_vl.tokenization_qwen import QWenTokenizer
from parameter import ModelArguments, DataArguments, TrainingArguments
import torch.distributed as dist
from data import make_inference_data_module

from model.Llama import Llama
from model.LlamaNoteLLM import LlamaNoteLLM
from model.Blip2 import BLIP2_INI
from model.llava.llava import Llava1p5
from model.llava.processing_llava import LlavaProcessor
from model.qwen_vl.modeling_qwen import QWenVL
from model.qwen.modeling_qwen import Qwen

from model.MLlama import LateFusion, MLlama
from model.MQwen import Qwen_LateFusion, Qwen_selfvl

os.environ["TOKENIZERS_PARALLELISM"] = "true"
if os.environ.get('WANDB_PROJECT') is not None:
    logging.warning(f"Logging wandb to project: '{os.environ['WANDB_PROJECT']}'.")
    os.environ["WANDB_DISABLED"]="false"
    os.environ["WANDB_WATCH"]="false"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_IMAGE_TOKEN = "<IMG>"
DEFAULT_IMAGE_EMB = "<IMG_EMB>"
DEFAULT_TEXT_EMB = "<TEXT_EMB>"

def main(
    base_model_path: str,
    save_path: str,
    model_args=None,
    data_args=None,
    training_args=None
):  

    if model_args.model_type == 'QWenVL' or 'qwen' in base_model_path.lower():
        print(f'qwen model max length:{training_args.model_max_length}')
        tokenizer = QWenTokenizer.from_pretrained(
            base_model_path,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
            add_eos_token=False
        )
    else:
        print(f'model max length:{training_args.model_max_length}')
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
            add_eos_token=False
        )

    if device == "cuda":
        selectLlama = eval(model_args.model_type)
        model = selectLlama.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            kwargs = model_args,
            ignore_mismatched_sizes = True
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path, device_map={"": device}, low_cpu_mem_usage=True
        )

    tokenizer.mICL = model_args.mICL
    training_args.mICL = model_args.mICL

    if tokenizer.pad_token is None and model_args.model_type != 'QWenVL' and model_args.model_type != 'Qwen':
        tokenizer.add_special_tokens(dict(eos_token=DEFAULT_EOS_TOKEN))
        print('eos token added', tokenizer.eos_token, tokenizer.eos_token_id)
        logging.warning("Adding pad token...")
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    if model_args.model_type == 'BLIP2_INI':
        smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(),
                tokenizer=tokenizer,
                model=model,
                normal_tokens=[DEFAULT_IMAGE_TOKEN]
            )
        model.image_token_id = tokenizer.encode(DEFAULT_IMAGE_TOKEN)[-1]
        model.image_emb = tokenizer.encode(DEFAULT_IMAGE_EMB)[-1]
        model.text_emb = tokenizer.encode(DEFAULT_TEXT_EMB)[-1]
        print(f'{DEFAULT_IMAGE_TOKEN} is added as {model.image_token_id}')
        processor = Blip2Processor.from_pretrained('../../model/blip2_opt7b')
    elif model_args.model_type == 'Llava1p5':
        processor = LlavaProcessor.from_pretrained('../../model/llava-1.5-7b-hf')
    elif model_args.model_type == 'QWenVL' or model_args.model_type == 'Qwen' or 'qw' in tokenizer.name_or_path.lower():
        tokenizer.pad_token_id = tokenizer.eod_id
        if not hasattr(model.transformer.config,'visual'):
            model.transformer.config.visual = {'image_start_id':tokenizer.encode('<img>')[-1]}
        model.image_emb = tokenizer.encode('<|extra_0|>')[-1]
        processor = None
    else:
        model.model.image_token_id = tokenizer.encode(DEFAULT_IMAGE_TOKEN)[-1]
        model.image_emb = tokenizer.encode(DEFAULT_IMAGE_EMB)[-1]
        model.text_emb = tokenizer.encode(DEFAULT_TEXT_EMB)[-1]
        processor = None
        if model_args.model_type=='LlamaNoteLLM':
            model.generation = False
            model.causal_weight = model_args.causal_weight
            model.contrastive = model_args.contrastive

    model.infer = True
    model.config.pad_token_id = tokenizer.pad_token_id

    model.half()
    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    

    data_module = make_inference_data_module(tokenizer=tokenizer, data_args=data_args,image_processor=processor,model_type=model_args.model_type)
    dist.barrier()
    
    trainer = Inference(model=model, 
                      tokenizer=tokenizer, 
                      args=training_args, 
                      callbacks= [WandbCallback()],
                      **data_module)
    
    logits = trainer.predict(data_module['train_dataset'])
    
    if int(os.environ["LOCAL_RANK"])==0:
        with open(save_path, 'wb') as file:
            pickle.dump(logits.predictions, file)
            print('save logits')

if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(
        base_model_path=model_args.model_name_or_path,
        save_path=data_args.save_path,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
        )
