from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
from typing import Optional, Dict, Sequence
import transformers
import tqdm
from torch.utils.data import Dataset
import os
import csv
import logging
import datasets
import math
import copy
import torch
import numpy as np
from PIL import Image
import re
import cv2
import random
from cv2 import (IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_IGNORE_ORIENTATION,
                 IMREAD_UNCHANGED)
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip, ToPILImage
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
IGNORE_INDEX = -100
    
def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    normal_tokens: List = []
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict) + tokenizer.add_tokens(normal_tokens)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg


def _tokenize_fn(
        strings: Sequence[str], 
        tokenizer: transformers.PreTrainedTokenizer, 
        has_image: bool, 
        model_type: str = None
        ):
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length if has_image else tokenizer.model_max_length,
            truncation=True,
        )
        for text in tqdm.tqdm(strings)
    ]
    prompt='，压缩为一个词：“'
    prompt_pt = tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
    start_idx = 0 if model_type == 'QWenVL' or 'qw' in tokenizer.name_or_path.lower() else 1
    input_ids = [torch.cat([tokenized.input_ids[0], prompt_pt.input_ids[0][start_idx:]]) for tokenized in tokenized_list]

    return dict(
        input_ids=input_ids,
    )

def _tokenize_fn_wotemplate(
        strings: Sequence[str], 
        tokenizer: transformers.PreTrainedTokenizer, 
        max_length: int = None
        ):
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        )
        for text in tqdm.tqdm(strings)
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    model_type: str = None
):
    """Preprocess the data by tokenizing."""
    sources_tokenized = _tokenize_fn(sources, tokenizer, has_image, model_type=model_type)
    input_ids = sources_tokenized["input_ids"]
    print(tokenizer.decode(input_ids[0]))
    return dict(input_ids=input_ids)

def preprocess_input_output(
    sources: Sequence[str],
    output: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    model_type: str = None,
    num_output: Sequence[int] = None,
):
    
    sources_tokenized = _tokenize_fn_wotemplate(sources, tokenizer, tokenizer.model_max_length)
    output_tokenized = _tokenize_fn_wotemplate(output, tokenizer, tokenizer.model_max_length)

    start_idx = 2

    if num_output is None:
        templates = '，压缩为一个词：“<TEXT_EMB>”。生成笔记类目：'
        templates_tokenized = _tokenize_fn_wotemplate([templates], tokenizer, 2048)
        input_ids = [torch.cat([sources_tokenized['input_ids'][i],templates_tokenized['input_ids'][0][start_idx:],output_tokenized['input_ids'][i][start_idx:]]) for i in range(len(sources_tokenized['input_ids']))]
    else:
        template = '，压缩为一个词：“<TEXT_EMB>”。生成笔记{}个话题：'
        templates = [template.format(i) for i in num_output]
        templates_tokenized = _tokenize_fn_wotemplate(templates, tokenizer, 2048)
        input_ids  = [torch.cat([sources_tokenized['input_ids'][i],templates_tokenized['input_ids'][i][start_idx:],output_tokenized['input_ids'][i][start_idx:]]) for i in range(len(sources_tokenized['input_ids']))]

    labels = copy.deepcopy(input_ids)
    for label, target_len in zip(labels, output_tokenized["input_ids_lens"]):
        label[:-target_len+2] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)



class NoteDataset(Dataset):

    def __init__(
            self, 
            data_path: str, 
            tokenizer: transformers.PreTrainedTokenizer, 
            max_text_length: int = 80, 
            has_image:bool = False, 
            only_image:bool = False, 
            model_type:str = None
            ):
        
        super(NoteDataset, self).__init__()
        logging.warning("Loading data...")
        self.noteid2id = {}
        self.id2noteid = {}
        self.input_ids = []
        contents = []
        self.max_text_length = max_text_length
        self.has_image = has_image
        self.only_image = only_image
        
        # prompt template
        if self.has_image and 'LateFusion' not in model_type:
            if self.only_image:
                self.out_template = '笔记：{{"图片":"<IMG>","标题":"","话题":"","内容":""}}'
            else:
                if tokenizer.mICL:
                    if 'qw' in tokenizer.name_or_path.lower():
                        self.out_template = '笔记：{{"图片":"<img></img>"}}，压缩为一个词：“<|extra_0|>”。笔记：{{"标题":"{}","话题":"{}","内容":"{}"}}'
                    else:
                        self.out_template = '笔记：{{"图片":"<IMG>"}}，压缩为一个词：“<IMG_EMB>”。笔记：{{"标题":"{}","话题":"{}","内容":"{}"}}'
                else:
                    if 'qw' in tokenizer.name_or_path.lower():
                        self.out_template = '笔记：{{"图片":"<img></img>","标题":"{}","话题":"{}","内容":"{}"}}'
                    else:
                        self.out_template = '笔记：{{"图片":"<IMG>","标题":"{}","话题":"{}","内容":"{}"}}'
        else:
            self.out_template = '笔记：{{"标题":"{}","话题":"{}","内容":"{}"}}'

        # read data
        with open(data_path,'r') as fp:
            reader = csv.reader( (line.replace('\0','') for line in fp) )
            for ind, data in enumerate(tqdm.tqdm(reader)):
                noteid, title, content, imgid, category = data[0], data[1], data[2], data[3], data[4]
                if self.only_image:
                    output_note = self.out_template
                else:
                    content, tags = self.split_content_hashtag(content)
                    content = content.replace('\n', '。')
                    content = content[:self.max_text_length].strip()
                    tags = tags.strip()
                    title = title[:20].strip()
                    output_note = self.out_template.format(title,tags,content)
                self.noteid2id[noteid] = ind
                self.id2noteid[ind] = noteid
                contents.append(output_note)
                if ind==0:
                    print(output_note)

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(contents, tokenizer, self.has_image, model_type=model_type)

        for ind in range(len(self.noteid2id)):
            self.input_ids.append(data_dict['input_ids'][ind])


    def split_content_hashtag(self, content):
        all_tags = []

        pattern = r"#(\w+)#"
        content = content.replace('[话题]', '')
        tags = re.finditer(pattern, content)

        for ent in tags:
            this_tag = ent.group()
            if len(this_tag) <= 16:
                all_tags.append(this_tag.replace('#', ''))
                content = content.replace(this_tag, '')

        all_tags = '，'.join(all_tags[:10])
        return content, all_tags

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i])

class MMNoteDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self, 
            image_path, 
            data_path, 
            note_data, 
            img_size=224, 
            image_processor=None, 
            only_text=False
            ):
        
        super(MMNoteDataset, self).__init__()
        logging.warning("Loading mm data...")

        self.note_data = note_data
        self.image_path = image_path
        self.id2iid = {}
        self.img_size = img_size
        self.toimage = ToPILImage()
        self.only_text = only_text
        if image_processor is None:
            self.transform = _transform(self.img_size)
        else:
            self.transform = image_processor

        with open(data_path,'r') as fp:
            reader = csv.reader( (line.replace('\0','') for line in fp) )
            for ind, data in enumerate(tqdm.tqdm(reader)):
                noteid, title, content, imgid, category = data[0], data[1], data[2], data[3], data[4]
                self.id2iid[ind] = imgid
        
    def __len__(self):
        return len(self.note_data)

    def read_img(self,ipath):
        try:
            with open(ipath, 'rb') as f:
                content = f.read()
            img_np = np.frombuffer(content, np.uint8)
            img = cv2.imdecode(img_np, IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.toimage(img)
            if type(self.transform)==Compose:
                img = self.transform(img)
            else:
                img = self.transform(images=img,return_tensors="pt")
                img = img['pixel_values'][0]
        except:
            print('wrong image. pad zero.')
            img = torch.zeros((3,self.img_size,self.img_size))
        return img

    def __getitem__(self, i):
        text_data = self.note_data[i]
        input_ids=text_data['input_ids']
        if 'id' in text_data:
            # for inference
            i = text_data['id']
        imgid = self.id2iid[i]
        if self.only_text:
            image = torch.zeros((3,self.img_size,self.img_size))
        else:
            image = self.read_img(os.path.join(self.image_path,imgid))
        if 'id' in text_data:
            return dict(input_ids=input_ids,image=image,id=i)
        else:
            return dict(input_ids=input_ids,image=image)

class NoteLLMNoteDataset(NoteDataset):
    def __init__(
            self, 
            data_path: str, 
            tokenizer: transformers.PreTrainedTokenizer, 
            max_text_length: int = 80, 
            has_image:bool = False, 
            only_image:bool = False, 
            model_type:str = None
            ):
        logging.warning("Loading data...")
        self.noteid2id = {}
        self.id2noteid = {}
        category_contents = []
        categorys = []
        topic_contents = []
        topics = []
        num_topics = []
        self.max_text_length = max_text_length
        self.has_image = has_image
        self.only_image = only_image
        
        # prompt template
        self.out_template = '笔记：{{"标题":"{}","话题":"{}","内容":"{}"}}'

        # read data
        with open(data_path,'r') as fp:
            reader = csv.reader( (line.replace('\0','') for line in fp) )
            for ind, data in enumerate(tqdm.tqdm(reader)):
                noteid, title, content, imgid, category = data[0], data[1], data[2], data[3], data[4]
                category = category.split('@')[:2]
                category = '，'.join(category)
                category = category[:-1] if category.endswith('，') else category
                category = category + f'。{tokenizer.eos_token}'
                content, tags = self.split_content_hashtag(content)
                content_topic, predict_topic, topic_size = self.select_topic(tags)
                content = content.replace('\n', '。')
                content = content[:self.max_text_length].strip()
                content_topic = content_topic.strip()
                predict_topic = predict_topic.strip() + f'。{tokenizer.eos_token}'
                title = title[:20].strip()
                tags = tags.strip()
                output_note_topic = self.out_template.format(title,content_topic,content)
                output_note_category = self.out_template.format(title,tags,content)
                self.noteid2id[noteid] = ind
                self.id2noteid[ind] = noteid
                topic_contents.append(output_note_topic)
                topics.append(predict_topic)
                num_topics.append(topic_size)
                category_contents.append(output_note_category)
                categorys.append(category)
                if ind==0:
                    print(output_note_topic, predict_topic)
                    print(output_note_category, category)

        logging.warning("Tokenizing inputs... This may take some time...")
        self.topic_data_dict = preprocess_input_output(topic_contents, topics, tokenizer, self.has_image, model_type=model_type, num_output=num_topics)
        self.category_data_dict = preprocess_input_output(category_contents, categorys, tokenizer, self.has_image, model_type=model_type)

    def select_topic(self, tags):
        topic = tags.split('，')
        if topic[0]=='' and len(topic)==1:
            topic = []
        topic_size = min(1,len(topic))
        # topic_size = random.randint(min(1,len(topic)),min(5,len(topic))) # OOM
        selected_topic = random.sample(topic, topic_size)
        content_topic = [t for t in topic if t not in selected_topic]
        content_topic = '，'.join(content_topic)
        predict_topic = '，'.join(selected_topic)
        return content_topic, predict_topic, topic_size

    def __len__(self):
        return len(self.topic_data_dict['input_ids'])

    def __getitem__(self, i):
        return dict(
            topic_input_ids=self.topic_data_dict['input_ids'][i],
            topic_labels=self.topic_data_dict['labels'][i],
            category_input_ids=self.category_data_dict['input_ids'][i],
            category_labels=self.category_data_dict['labels'][i],
            )

class PairDataset(Dataset):
    def __init__(
            self, 
            pairdata_path: str, 
            noteid2id: dict, 
            note_info: Dataset, 
            has_image:bool = False
            ):
        
        super(PairDataset, self).__init__()
        logging.warning("Loading pair data...")

        self.note_info = note_info
        self.noteid2id = noteid2id
        self.pair_info = []
        self.has_image = has_image

        with open(pairdata_path,'r') as fp:
            reader = csv.reader( (line.replace('\0','') for line in fp) )
            for ind, data in enumerate(tqdm.tqdm(reader)):
                if data[0] not in self.noteid2id or data[1] not in self.noteid2id:
                    continue
                id1 = self.noteid2id[data[0]]
                id2 = self.noteid2id[data[1]]
                self.pair_info.append((id1,id2))
        if 'val' in pairdata_path:
            random.shuffle(self.pair_info)
        print(f'pair num: {len(self.pair_info)}')

    def __len__(self):
        return len(self.pair_info)

    def __getitem__(self, i):
        id1=self.pair_info[i][0]
        id2=self.pair_info[i][1]
        if self.has_image:
            note1 = self.note_info[id1]
            note2 = self.note_info[id2]
            input_ids1 = torch.tensor(note1['input_ids'])
            input_ids2 = torch.tensor(note2['input_ids'])
            return dict(input_ids1=input_ids1,input_ids2=input_ids2,image1=note1['image'],image2=note2['image'])
        else:
            return dict(input_ids1=self.note_info[id1]['input_ids'],input_ids2=self.note_info[id2]['input_ids'])

class NoteLLMPairDataset(PairDataset):
    def __init__(
            self, 
            pairdata_path: str, 
            noteid2id: dict, 
            note_info: Dataset, 
            has_image:bool = False
            ):
        super().__init__(
            pairdata_path, 
            noteid2id, 
            note_info, 
            has_image
            )
    
    def __getitem__(self, i):
        id1=self.pair_info[i][0]
        id2=self.pair_info[i][1]
        return dict(
            input_ids1=self.note_info[id1]['category_input_ids'],
            input_ids2=self.note_info[id2]['category_input_ids'],
            input_ids3=self.note_info[id1]['topic_input_ids'],
            input_ids4=self.note_info[id2]['topic_input_ids'],
            labels1=self.note_info[id1]['category_labels'],
            labels2=self.note_info[id2]['category_labels'],
            labels3=self.note_info[id1]['topic_labels'],
            labels4=self.note_info[id2]['topic_labels'],
            )

@dataclass
class DataCollatorForPairDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    has_image: bool = False
    def __call__(self, instances: Sequence[Dict]):
        input_ids1, input_ids2 = tuple([instance[key] for instance in instances] for key in ("input_ids1", "input_ids2"))
        input_ids1.extend(input_ids2)
        input_ids1 = torch.nn.utils.rnn.pad_sequence(
            input_ids1, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        
        if self.has_image:
            image1, image2 = tuple([instance[key] for instance in instances] for key in ("image1", "image2"))
            image1.extend(image2)
            image1 = torch.stack(image1)

            return dict(
                input_ids=input_ids1,
                attention_mask=input_ids1.ne(self.tokenizer.pad_token_id),
                images=image1
            )
        
        else:
            return dict(
                input_ids=input_ids1,
                attention_mask=input_ids1.ne(self.tokenizer.pad_token_id),
            )
        
@dataclass
class NoteLLMDataCollatorForPairDataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    topic_rate: float
    def __call__(self, instances: Sequence[Dict]):
        input_ids = tuple([instance[key] for instance in instances] for key in [f'input_ids{i}' for i in range(1,5)])
        train_input_ids1 = input_ids[0]
        train_input_ids2 = input_ids[1]
        bs=len(train_input_ids1)
        topic_size = int(self.topic_rate*bs)
        random_1 = random.sample([i for i in range(bs)],topic_size)
        random_2 = random.sample([i for i in range(bs)],topic_size)
        for i in random_1:
            train_input_ids1[i] = input_ids[2][i]
        for i in random_2:
            train_input_ids2[i] = input_ids[3][i]
        train_input_ids1.extend(train_input_ids2)
        input_ids1 = torch.nn.utils.rnn.pad_sequence(
            train_input_ids1, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = tuple([instance[key] for instance in instances] for key in [f'labels{i}' for i in range(1,5)])
        train_labels1 = labels[0]
        train_labels2 = labels[1]
        for i in random_1:
            train_labels1[i] = labels[2][i]
        for i in random_2:
            train_labels2[i] = labels[3][i]
        train_labels1.extend(train_labels2)
        labels1 = torch.nn.utils.rnn.pad_sequence(train_labels1, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids1,
            attention_mask=input_ids1.ne(self.tokenizer.pad_token_id),
            labels=labels1
        )


def make_training_val_data_module(
        tokenizer: transformers.PreTrainedTokenizer, 
        data_args, 
        image_processor = None, 
        model_type=None):
    
    print(f'{os.environ["LOCAL_RANK"]} starts processing data')

    if model_type != 'LlamaNoteLLM':
        notedataset = NoteDataset
        pairdataset = PairDataset
    else:
        notedataset = NoteLLMNoteDataset
        pairdataset = NoteLLMPairDataset

    if os.path.exists(os.path.join(data_args.data_cache_path, "dataset_info.json")):
        logging.warning("Loading cached dataset...")
        dataset = datasets.Dataset.load_from_disk(data_args.data_cache_path)
        noteid2id = np.load(data_args.noteid2id_path,allow_pickle=True)
        noteid2id = noteid2id.item()
    else:
        notedata = notedataset(
            tokenizer=tokenizer, 
            data_path=data_args.data_path, 
            has_image=data_args.image_path is not None, 
            only_image=data_args.only_image,
            model_type=model_type
            )
        dataset = datasets.Dataset.from_list(notedata)
        noteid2id = notedata.noteid2id
        if int(os.environ["LOCAL_RANK"])==0:
            if not os.path.exists(data_args.data_cache_path):
                os.makedirs(data_args.data_cache_path, exist_ok=True)
            dataset.save_to_disk(data_args.data_cache_path)
            np.save(data_args.noteid2id_path,notedata.noteid2id)

    if data_args.image_path is not None:
        mm_notedata = MMNoteDataset(
            image_path=data_args.image_path, 
            data_path=data_args.data_path, 
            note_data=dataset, 
            img_size=data_args.image_size, 
            image_processor=image_processor
            )
        train_dataset = PairDataset(
            pairdata_path=data_args.train_pairdata_path, 
            noteid2id=noteid2id, 
            note_info=mm_notedata,
            has_image=True
            )
        eval_dataset = PairDataset(
            pairdata_path=data_args.val_pairdata_path, 
            noteid2id=noteid2id, 
            note_info=mm_notedata,
            has_image=True
            )
    else:
        train_dataset = pairdataset(
            pairdata_path=data_args.train_pairdata_path, 
            noteid2id=noteid2id, 
            note_info=dataset,
            )
        train_dataset = datasets.Dataset.from_list(train_dataset)
        eval_dataset = pairdataset(
            pairdata_path=data_args.val_pairdata_path, 
            noteid2id=noteid2id, 
            note_info=dataset,
            )
        eval_dataset = datasets.Dataset.from_list(eval_dataset)

    if data_args.image_path is None:
        train_dataset, eval_dataset = train_dataset.shuffle(), eval_dataset.shuffle()
        if model_type != 'LlamaNoteLLM':
            train_dataset.set_format(type="torch", columns=["input_ids1","input_ids2"])
            eval_dataset.set_format(type="torch", columns=["input_ids1","input_ids2"])
            data_collator = DataCollatorForPairDataset(tokenizer=tokenizer)
        else:
            train_dataset.set_format(type="torch", columns=[f'input_ids{i}' for i in range(1,5)]+[f'labels{i}' for i in range(1,5)])
            eval_dataset.set_format(type="torch", columns=[f'input_ids{i}' for i in range(1,5)]+[f'labels{i}' for i in range(1,5)])
            data_collator = NoteLLMDataCollatorForPairDataset(tokenizer=tokenizer,topic_rate=data_args.topic_rate)
    else:
        data_collator = DataCollatorForPairDataset(tokenizer=tokenizer,has_image=True)

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

class NoteLengthDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self, 
            data_path: str, 
            tokenizer: transformers.PreTrainedTokenizer, 
            max_text_length:int = 80, 
            has_image:bool = False, 
            only_image:bool = False, 
            model_type:str = None
            ):
        
        super(NoteLengthDataset, self).__init__()
        logging.warning("Loading data...")

        self.noteid2id = {}
        contents = []
        self.max_text_length = max_text_length
        self.has_image = has_image

        if model_type == 'Llava1p5':
            self.image_token = '<image>'
        elif model_type == 'QWenVL' or 'qw' in tokenizer.name_or_path.lower():
            self.image_token = '<img></img>'
        else:
            self.image_token = '<IMG>'

        if self.has_image and 'LateFusion' not in model_type:
            if tokenizer.mICL:
                if model_type == 'QWenVL' or 'qw' in tokenizer.name_or_path.lower():
                    self.out_template = '笔记：{{"图片":"'+self.image_token+'"}}，压缩为一个词：“<|extra_0|>”。笔记：{{"标题":"{}","话题":"{}","内容":"{}"}}'
                else:
                    self.out_template = '笔记：{{"图片":"'+self.image_token+'"}}，压缩为一个词：“<IMG_EMB>”。笔记：{{"标题":"{}","话题":"{}","内容":"{}"}}'
            else:
                self.out_template = '笔记：{{"图片":"'+self.image_token+'","标题":"{}","话题":"{}","内容":"{}"}}'
        else:
            self.out_template = '笔记：{{"标题":"{}","话题":"{}","内容":"{}"}}'

        self.only_image = only_image
        print(self.out_template)

        with open(data_path,'r') as fp:
            reader = csv.reader( (line.replace('\0','') for line in fp) )
            for ind, data in enumerate(tqdm.tqdm(reader)):
                noteid, title, content, imgid, category = data[0], data[1], data[2], data[3], data[4]

                if self.only_image:
                    title = ''
                    tags = ''
                    content = ''
                else:
                    content, tags = self.split_content_hashtag(content)
                    content = content.replace('\n', '。')
                    content = content[:self.max_text_length].strip()
                    tags = tags.strip()
                    title = title[:20].strip()

                output_note = self.out_template.format(title,tags,content)
                self.noteid2id[noteid] = ind
                contents.append(output_note)

                if ind==0:
                    print(output_note)

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(contents, tokenizer, self.has_image, model_type)

        self.id2noteid = {v:k for k,v in self.noteid2id.items()}
        self.input_ids = data_dict["input_ids"]
        self.note_len = {i:data_dict["input_ids"][i].shape[0] for i in range(len(contents))}

    def split_content_hashtag(self, content):
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

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i],id=i,length=self.note_len[i])

@dataclass
class DataCollatorDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    has_image: bool = False

    def __call__(self, instances: Sequence[Dict]):
        input_ids1 = [torch.tensor(instance["input_ids"]) for instance in instances]
        note_id = torch.stack([torch.tensor(instance["id"]) for instance in instances])
        input_ids1 = torch.nn.utils.rnn.pad_sequence(
            input_ids1, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        if self.has_image:
            image1 = [instance["image"] for instance in instances]
            image1 = torch.stack(image1)
            return dict(
                input_ids=input_ids1,
                attention_mask=input_ids1.ne(self.tokenizer.pad_token_id),
                id=note_id,
                images=image1
            )
        return dict(
                input_ids=input_ids1,
                attention_mask=input_ids1.ne(self.tokenizer.pad_token_id),
                id=note_id
            )


def make_inference_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, image_processor = None, model_type = None):
    """Make dataset and collator for supervised fine-tuning."""

    print(f'{os.environ["LOCAL_RANK"]} starts processing data')
    if os.path.exists(os.path.join(data_args.data_cache_path, "dataset_info.json")):
        logging.warning("Loading cached dataset...")
        dataset = datasets.Dataset.load_from_disk(data_args.data_cache_path)
        noteid2id = np.load(data_args.noteid2id_path,allow_pickle=True)
        noteid2id = noteid2id.item()
    else:
        notedata = NoteLengthDataset(tokenizer=tokenizer, data_path=data_args.data_path, has_image = data_args.image_path is not None, only_image=data_args.only_image, model_type=model_type)
        dataset = datasets.Dataset.from_list(notedata)
        noteid2id = notedata.noteid2id
        if int(os.environ["LOCAL_RANK"])==0:
            if not os.path.exists(data_args.data_cache_path):
                os.makedirs(data_args.data_cache_path, exist_ok=True)
            dataset.save_to_disk(data_args.data_cache_path)
            np.save(data_args.noteid2id_path,notedata.noteid2id)

    dataset = dataset.sort('length', reverse=True)
    dataset = dataset.remove_columns('length')

    if data_args.image_path is not None:
        dataset = MMNoteDataset(image_path=data_args.image_path, data_path=data_args.data_path, note_data=dataset, img_size=data_args.image_size, image_processor=image_processor, only_text=data_args.only_text)
    
    data_collator = DataCollatorDataset(tokenizer=tokenizer,has_image=data_args.image_path is not None)

    return dict(train_dataset=dataset, data_collator=data_collator)
