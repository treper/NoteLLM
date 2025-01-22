from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
from typing import Optional, Dict, Sequence
import transformers
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    model_type: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    clip_path: Optional[str] = field(default=None)
    fix_text_encoder: bool = field(default=False, metadata={"help": "Whether to fix LLMs."})
    fix_text_linear: bool = field(default=False, metadata={"help": "Whether to fix the text linear layer after LLMs."})
    fix_image_encoder: bool = field(default=True, metadata={"help": "Whether to fix image encoders."})
    fix_connector_encoder: bool = field(default=False, metadata={"help": "Whether to fix connectors."})
    fix_late_fusion: bool = field(default=False, metadata={"help": "Whether to fix late fusion."})
    mm_projector_type: Optional[str] = field(default='linear')
    mm_vision_select_layer: int = field(
        default=-1, metadata={"help": "Choose which feature layer to use from the visual encoder in the input LLM."}
    )
    mm_vision_select_feature: Optional[str] = field(default='cls_patch', \
                                metadata={"help": "Choose which features from the visual encoder in the input LLM."})
    mm_hidden_size: int = field(
        default=768, metadata={"help": "The hidden state size of the connector."}
    )
    hidden_size: int = field(
        default=4096, metadata={"help": "The hidden state size of the LLM."}
    )
    num_query_tokens: int = field(
        default=16, metadata={"help": "The number of image tokens input into the LLM."}
    )
    query_hidden_size: int = field(
        default=768, metadata={"help": "The dimension size of query tokens."}
    )
    blip_vocab_size: int = field(
        default=0,
    )
    blip_num_hidden_layers: int = field(
        default=6, metadata={"help": "The number of layers of the connector."}
    )
    mICL: bool = field(default=False, metadata={"help": "Whether to use mICL."})
    late_fusion: bool = field(default=False, metadata={"help": "Whether to use late fusion."})
    vision_weight: float = field(default=1)
    text_weight: float = field(default=1)
    dim_reduction: int = field(
        default=64, metadata={"help": "The dimension of reduction."}
    )
    temperature: float = field(
        default=3.0,
        metadata={"help":"Temperature of contrastive learning."}
    )
    causal_weight: float = field(
        default=0.01,
        metadata={"help":"Control the size of the generated loss"}
    )
    contrastive: bool = field(default=False, metadata={"help": "Whether to use contrastive learning."})

@dataclass
class DataArguments:
    image_size: int = field(
        default=224,
        metadata={"help": "Image size."}
    )
    image_path: str = field(default=None, metadata={"help": "Path to note images."})
    data_path: str = field(default=None, metadata={"help": "Path to note information."})
    data_cache_path: str = field(default="", metadata={"help": "Cache path to note information."})
    noteid2id_path: str = field(default="", metadata={"help": "Cache path to note id information"})
    train_pairdata_cache_path: str = field(default=None)
    train_pairdata_path: str = field(default=None, metadata={"help": "Path to training set of note pairs."})
    val_pairdata_cache_path: str = field(default=None)
    val_pairdata_path: str = field(default=None, metadata={"help": "Path to validation set of note pairs."})
    save_path: str = field(default=None, metadata={"help": "Path of inferenced notes logits."})
    only_image: bool = field(default=False, metadata={"help": "Load only image data."})
    only_text: bool = field(default=False, metadata={"help": "Load only text data."})
    topic_rate: float = field(
        default=0.4,
        metadata={"help":"The proportion of performing topic prediction tasks."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
