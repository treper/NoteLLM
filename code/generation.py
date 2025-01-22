import sys
import os
import io
import csv
from torch.utils.data import Dataset
import tqdm
import datasets
import torch
from transformers import GenerationConfig, AutoModelForCausalLM, LlamaTokenizer
from transformers.generation.utils import StoppingCriteriaList, StoppingCriteria
from model.LlamaNoteLLM import LlamaNoteLLM
import pandas as pd
import argparse
import deepspeed
import torch.distributed as dist
import re
from data import smart_tokenizer_and_embedding_resize, _tokenize_fn_wotemplate

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_PAD_TOKEN = "[PAD]"
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

deepspeed.init_distributed("nccl")
rank = dist.get_rank()

class NoteLengthDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        super(NoteLengthDataset, self).__init__()
        self.answers = []
        self.inputs = []
        out_template_topic = '笔记：{{"标题":"{}","内容":"{}"}}，压缩为一个词：“<TEXT_EMB>”。生成笔记5个话题：'
        out_template_category = '笔记：{{"标题":"{}","话题":"{}","内容":"{}"}}，压缩为一个词：“<TEXT_EMB>”。生成笔记类目：'
        with open(data_path,'r') as fp:
            reader = csv.reader( (line.replace('\0','') for line in fp) )
            for ind, data in enumerate(tqdm.tqdm(reader)):
                noteid, title, content, imgid, category = data[0], data[1], data[2], data[3], data[4]
                category = category.split('@')
                category = '，'.join(category)
                category = category[:-1] if category.endswith('，') else category
                category = category + '。'
                content, tags = split_content_hashtag(content)
                content = content.replace('\n', '。')
                content = content[:args.input_max_content_length].strip()
                title = title[:20].strip()
                tags = tags.strip() + '。'
                if args.type == 'category':
                    output_note = out_template_category.format(title,tags,content)
                    self.answers.append(category)
                else:
                    output_note = out_template_topic.format(title,content)
                    self.answers.append(tags)
                self.inputs.append(output_note)
        self.inputs = _tokenize_fn_wotemplate(self.inputs,tokenizer,tokenizer.model_max_length)['input_ids']
        self.answers = _tokenize_fn_wotemplate(self.answers,tokenizer,tokenizer.model_max_length)['input_ids']

    def __len__(self):
        return len(self.answers)
    
    def __getitem__(self, i):
        return dict(inputs=self.inputs[i],answer=self.answers[i],inputs_len=self.inputs[i].shape[0])

def split_content_hashtag(content):
        offset = -1
        all_tags = []

        pattern = r"#(\w+)#"
        content = content.replace('[话题]', '')
        tags = re.finditer(pattern, content)

        for ent in tags:
            this_offset = ent.start(0)
            this_tag = ent.group()
            if len(this_tag) <= 16:
                all_tags.append(this_tag.replace('#', ''))
                content = content.replace(this_tag, '')

        all_tags = '，'.join(all_tags[:10])
        return content, all_tags

def print_rank0(*msg):
    if rank != 0:
        return
    print(*msg)

def left_pad(instruction,tokenizer):
    reversed_instruction = [i.flip(dims=(0,)) for i in instruction]
    instruction = torch.nn.utils.rnn.pad_sequence(
            reversed_instruction, batch_first=True, padding_value=tokenizer.pad_token_id,)
    return instruction.flip(dims=(1,))

def evaluate(
        notes,
        generation_config,
        max_length=30,
        tokenizer=None,
        model = None,
        **kwargs,
    ):
    notes = left_pad(notes,tokenizer)
    notes = notes.to(torch.cuda.current_device())
    mask = notes.ne(tokenizer.pad_token_id).to(torch.cuda.current_device())
    input_len = mask.shape[1]
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=notes,
            attention_mask=mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_length,
            output_attentions=True
        )
    s = generation_output.sequences
    new_tokens = generation_output.sequences[:,input_len:]
    outputs = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return [output.strip() for output in outputs]

def main(
    args: str = None
):
    base_model_path = args.base_model_path
    save_path = args.save_path
    data_path = args.data_path
    cache_path = args.cache_path
    max_length = args.max_length
    kernel_inject = args.kernel

    if kernel_inject:
        # for current ds-inference only works with fp16
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    tokenizer = LlamaTokenizer.from_pretrained(
        base_model_path,
    )
    tokenizer.model_max_length = max_length

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(dict(eos_token=DEFAULT_EOS_TOKEN))
        print('eos token added', tokenizer.eos_token, tokenizer.eos_token_id)
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    model = LlamaNoteLLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype
    )
    
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.infer = True
    model.generation = True
    model.contrastive = False
    model.half()
    model.eval()

    model = deepspeed.init_inference(
        model,
        mp_size=world_size,
        dtype=dtype,
        replace_with_kernel_inject=True
    )
    model = model.module
    
    if os.path.exists(os.path.join(cache_path, "dataset_info.json")):
        dataset = datasets.Dataset.load_from_disk(cache_path)
    else:
        notedatas = NoteLengthDataset(data_path,tokenizer)
        dataset = datasets.Dataset.from_list(notedatas)
        dataset.set_format(type="torch", columns=['inputs','answer','inputs_len'])
        dataset = dataset.sort('inputs_len')
        if rank==0:
            dataset.save_to_disk(cache_path)
        
    os.makedirs(save_path, exist_ok=True)

    generation_config = GenerationConfig(
        temperature=0.2,
        top_p=0.1,
        top_k=10,
        num_beams=1,
        do_sample=False,
        repetition_penalty=1.2,
    )
    outputs = []
    batch_size = 3
    for i in range(0, len(dataset), batch_size ):
        batch = dataset[i:i + batch_size]
        note = batch['inputs']
        answer = tokenizer.batch_decode(batch['answer'], skip_special_tokens=True)
        responses = evaluate(
                note,
                generation_config = generation_config,
                max_length=max_length,
                tokenizer = tokenizer,
                model=model
            )
        for i in range(len(responses)):
            outputs.append([responses[i],answer[i]])
        

    csvfile = open(save_path+f'generation_{args.type}.csv', 'w', newline='')
    writer = csv.writer(csvfile, delimiter=',')
    cnt=0
    for data in outputs:
        try:
            writer.writerow(data)
            cnt+=1
        except:
            print('wrong!')
            continue
    print(f'sample num: {cnt}')
    csvfile.close()
                    

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--base_model_path", type=str)
    arg_parser.add_argument("--save_path", type=str, default="category.txt")
    arg_parser.add_argument("--data_path", type=str)
    arg_parser.add_argument("--cache_path", type=str)
    arg_parser.add_argument("--input_max_content_length", type=int, default=80)
    arg_parser.add_argument("--max_length", type=int, default=2048)
    arg_parser.add_argument("--type", type=str, default="category")
    arg_parser.add_argument("--kernel", type=bool, default=True)
    arg_parser.add_argument("--local_rank", type=int, default=0)

    args = arg_parser.parse_args()
    print(args)
    main(args=args)
