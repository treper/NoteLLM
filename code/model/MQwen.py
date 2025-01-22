import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Blip2QFormerModel, Blip2QFormerConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from .qwen.modeling_qwen import Qwen
from .Llama import concat_all_gather

from .image_encoder.builder import build_vision_tower
from .image_projector.builder import build_vision_projector

from .Gate import Gate

@dataclass
class mICL_CausalLMOutputWithPast(CausalLMOutputWithPast):
    image_logits: torch.FloatTensor = None

class MQwen(Qwen):

    def encode_images(self):
        pass

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        images = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if images is not None:
            image_features, clip_feature = self.encode_images(images)
        else:
            image_features = None

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            image_features=image_features
        )
        logits = transformer_outputs[0]
        image_length = logits.shape[1] - input_ids.shape[1]

        loss = None
        batch_size = logits.shape[0]

        if input_ids is not None:
            sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            sequence_lengths += image_length
            if self.kwargs.mICL:
                _, image_emb_loc = torch.where(input_ids == self.image_emb) 
                image_emb_loc = image_emb_loc - 1 + image_length

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        if self.kwargs.mICL:
            image_logits = logits[torch.arange(batch_size, device=logits.device), image_emb_loc]

        if self.late_fusion:
            pooled_logits = self.gate(pooled_logits, clip_feature)
            if self.kwargs.mICL:
                image_logits = self.gate(image_logits, clip_feature)

        if self.dim_reduction!=-1:
            pooled_logits = self.linear(pooled_logits)
            if self.kwargs.mICL:
                image_logits = self.linear(image_logits)

        if not self.infer:
            if not self.kwargs.mICL:
                loss = self.concat_compute_oneloss(pooled_logits,batch_size//2)
            else:
                v_loss = self.concat_compute_oneloss(image_logits,batch_size//2)
                m_loss = self.concat_compute_oneloss(pooled_logits,batch_size//2)
                loss = (m_loss*self.kwargs.text_weight + v_loss*self.kwargs.vision_weight)/(self.kwargs.text_weight+self.kwargs.vision_weight)
        else:
            loss = None

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        if self.kwargs.mICL:
            return mICL_CausalLMOutputWithPast(
                loss=loss,
                logits=pooled_logits,
                image_logits=image_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )
        else:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=pooled_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )
    
    def concat_compute_oneloss(self,pooled_logits,pair_num):
        bs = pooled_logits.shape[0]
        pooled_logits = pooled_logits/torch.norm(pooled_logits,dim=1,keepdim=True)
        all_pooled_logits, rank = concat_all_gather(pooled_logits)
        all_bs = all_pooled_logits.shape[0]
        all_rank = all_bs//bs
        pooled_logits = (all_pooled_logits @ all_pooled_logits.T)
        shape = pooled_logits.shape
        pooled_logits = (pooled_logits - torch.eye(shape[0], shape[1],device=pooled_logits.device) * 1e12) * torch.exp(self.temperature)
        labels = []
        for i in range(all_rank):
            for j in range(pair_num):
                labels.append(bs*i+j+pair_num)
            for j in range(pair_num):
                labels.append(bs*i+j)
        labels = torch.tensor(labels).to(pooled_logits.device)
        loss = F.cross_entropy(pooled_logits, labels)
        return loss

    def concat_compute_dualloss(self,pooled_logits,pair_num):
        bs = pooled_logits.shape[0]
        pooled_logits = pooled_logits/torch.norm(pooled_logits,dim=1,keepdim=True)
        query = pooled_logits[:pair_num]
        doc = pooled_logits[pair_num:]

        all_query, rank = concat_all_gather(query)
        all_doc, rank = concat_all_gather(doc)
        
        logits = (all_query @ all_doc.T) * torch.exp(self.temperature)
        n_samples = all_query.shape[0]
        labels = torch.arange(n_samples, device=logits.device, dtype=torch.long)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))/2
        return loss
    

class Qwen_selfvl(MQwen):
    def __init__(self, config, kwargs=None):
        super().__init__(config, kwargs)
        self.kwargs = kwargs
        self.vision_tower = build_vision_tower(kwargs, delay_load=False)
        self.query_tokens = nn.Parameter(torch.zeros(1, kwargs.num_query_tokens, kwargs.query_hidden_size))
        print(f'query token size: {self.query_tokens.shape}')
        qformer_config = Blip2QFormerConfig(vocab_size=kwargs.blip_vocab_size,num_hidden_layers=kwargs.blip_num_hidden_layers,encoder_hidden_size=kwargs.mm_hidden_size,hidden_size=kwargs.query_hidden_size)
        self.qformer = Blip2QFormerModel(qformer_config)
        self.mm_projector = build_vision_projector(kwargs)
        self.late_fusion = kwargs.late_fusion
        if kwargs.fix_text_encoder:
            self.model.requires_grad_(False)
        if kwargs.fix_text_linear:
            self.linear.requires_grad_(False)
        if kwargs.fix_connector_encoder:
            self.qformer.requires_grad_(False)
            self.mm_projector.requires_grad_(False)
            self.query_tokens.requires_grad_(False)
        if kwargs.late_fusion:
            self.mm_projector_latefusion = nn.Linear(kwargs.mm_hidden_size, kwargs.hidden_size, bias=False)
            self.gate = Gate(kwargs.hidden_size)
            if kwargs.fix_late_fusion:
                self.mm_projector_latefusion.requires_grad_(False)
                self.gate.requires_grad_(False)
        print('learnable paramaters:')
        print([name for name, param in self.named_parameters() if param.requires_grad])
        self.post_init()

    def encode_images(self,images):
        query_tokens = self.query_tokens.expand(images.shape[0], -1, -1)
        image_features = self.vision_tower(images)
        clip_feature = image_features[:, 0]
        if self.late_fusion:
            clip_feature = self.mm_projector_latefusion(clip_feature)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_features,
        )
        query_outputs = query_outputs['last_hidden_state']
        image_features = self.mm_projector(query_outputs)

        return image_features, clip_feature


class Qwen_LateFusion(MQwen):
    def __init__(self, config, kwargs=None):
        super().__init__(config, kwargs)
        self.kwargs = kwargs
        self.vision_tower = build_vision_tower(kwargs, delay_load=False)
        self.late_fusion = kwargs.late_fusion
        if kwargs.fix_text_encoder:
            self.model.requires_grad_(False)
        if kwargs.fix_text_linear:
            self.linear.requires_grad_(False)
        if kwargs.late_fusion:
            self.mm_projector_latefusion = nn.Linear(kwargs.mm_hidden_size, kwargs.hidden_size, bias=False)
            self.gate = Gate(kwargs.hidden_size)
            if kwargs.fix_late_fusion:
                self.mm_projector_latefusion.requires_grad_(False)
                self.gate.requires_grad_(False)

        print('learnable paramaters:')
        print([name for name, param in self.named_parameters() if param.requires_grad])
        self.post_init()
    
    def encode_images(self,images):
        image_features = self.vision_tower(images)
        clip_feature = image_features[:, 0]
        if self.late_fusion:
            clip_feature = self.mm_projector_latefusion(clip_feature)
        return None, clip_feature