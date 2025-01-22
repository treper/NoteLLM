import transformers
from transformers import Blip2Processor
from transformers.integrations import WandbCallback
from transformers import LlamaTokenizer, AutoTokenizer
from model.qwen_vl.tokenization_qwen import QWenTokenizer
import torch
import os
from parameter import ModelArguments, DataArguments, TrainingArguments
from data import make_training_val_data_module, smart_tokenizer_and_embedding_resize
from trainer import TrainingTrainer
import logging

from model.Llama import Llama
from model.LlamaNoteLLM import LlamaNoteLLM
from model.Blip2 import BLIP2_INI
from model.qwen_vl.modeling_qwen import QWenVL
from model.qwen.modeling_qwen import Qwen

from model.MLlama import LateFusion, MLlama
from model.MQwen import Qwen_LateFusion, Qwen_selfvl

if os.environ.get('WANDB_PROJECT') is not None:
    logging.warning(f"Logging wandb to project: '{os.environ['WANDB_PROJECT']}'.")
    os.environ["WANDB_DISABLED"]="false"
    os.environ["WANDB_WATCH"]="false"

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_IMAGE_TOKEN = "<IMG>"
DEFAULT_IMAGE_EMB = "<IMG_EMB>"
DEFAULT_TEXT_EMB = "<TEXT_EMB>"

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    selectLlama = eval(model_args.model_type)
    model = selectLlama.from_pretrained(
            model_args.model_name_or_path,
            kwargs = model_args,
        )
    model.config.use_cache = False

    if model_args.model_type == 'BLIP2_INI':
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
            add_eos_token=False
        )
    elif model_args.model_type == 'QWenVL' or 'Qwen' in model_args.model_name_or_path:
        print(f'model max length:{training_args.model_max_length}')
        tokenizer = QWenTokenizer.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
            add_eos_token=False,
            pad_token='<|endoftext|>'
        )
    else:
        tokenizer = LlamaTokenizer.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
            add_eos_token=False
        )
        tokenizer.add_special_tokens(dict(eos_token=DEFAULT_EOS_TOKEN))
        print('eos token added', tokenizer.eos_token, tokenizer.eos_token_id)

    if tokenizer.pad_token is None:
        logging.warning("Adding pad token...")
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
            normal_tokens=[DEFAULT_IMAGE_TOKEN]
        )
    elif 'qw' not in model_args.model_name_or_path.lower():
        smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(),
                tokenizer=tokenizer,
                model=model,
                normal_tokens=[DEFAULT_IMAGE_TOKEN]
            )
        
    if (model_args.mICL and 'qw' not in model_args.model_name_or_path.lower()) or model_args.model_type=='LlamaNoteLLM':
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(),
            tokenizer=tokenizer,
            model=model,
            normal_tokens=[DEFAULT_IMAGE_EMB, DEFAULT_TEXT_EMB]
        )
    tokenizer.mICL = model_args.mICL
    model.config.pad_token_id = tokenizer.pad_token_id
    if model_args.model_type == 'BLIP2_INI':
        model.image_token_id = tokenizer.encode(DEFAULT_IMAGE_TOKEN)[-1]
        model.image_emb = tokenizer.encode(DEFAULT_IMAGE_EMB)[-1]
        model.text_emb = tokenizer.encode(DEFAULT_TEXT_EMB)[-1]
        processor = Blip2Processor.from_pretrained(model_args.model_name_or_path)
    elif 'qw' not in model_args.model_name_or_path.lower():
        model.model.image_token_id = tokenizer.encode(DEFAULT_IMAGE_TOKEN)[-1]
        model.image_emb = tokenizer.encode(DEFAULT_IMAGE_EMB)[-1]
        model.text_emb = tokenizer.encode(DEFAULT_TEXT_EMB)[-1]
        processor = None
        if model_args.model_type=='LlamaNoteLLM':
            model.generation = False
            model.causal_weight = model_args.causal_weight
            model.contrastive = model_args.contrastive
    else:
        if not hasattr(model.transformer.config,'visual'):
            model.transformer.config.visual = {'image_start_id':tokenizer.encode('<img>')[-1]}
        model.image_emb = tokenizer.encode('<|extra_0|>')[-1]
        processor = None
    model.infer = False
    data_module = make_training_val_data_module(tokenizer=tokenizer, data_args=data_args, image_processor=processor, model_type=model_args.model_type)
    trainer = TrainingTrainer(model=model, 
                      tokenizer=tokenizer, 
                      args=training_args, 
                      callbacks=[WandbCallback()],
                      **data_module)

    if training_args.resume_from_checkpoint == 'True':
        training_args.resume_from_checkpoint = True
    if training_args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()
    trainer.save_state()
    trainer.save_model()


if __name__ == "__main__":
    train()
