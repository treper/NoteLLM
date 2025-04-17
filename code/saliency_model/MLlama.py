import torch
import torch.nn as nn
from transformers import Blip2QFormerModel, Blip2QFormerConfig
from .Llama import Llama
from dataclasses import dataclass
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from typing import Optional, Tuple, Union, List

from .image_encoder.builder import build_vision_tower
from .image_projector.builder import build_vision_projector

from .Gate import Gate

import numpy as np
import os

@dataclass
class mICL_SequenceClassifierOutputWithPast(SequenceClassifierOutputWithPast):
    image_logits: torch.FloatTensor = None
    text_logits: torch.FloatTensor = None


class CustomMLLama(Llama):

    def image_fuse(self,input_ids,image):
        return input_ids

    def encode_images(self):
        pass

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        self.attentionermanger.zero_grad()
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if images is not None:
            image_features, clip_feature = self.encode_images(images)
        else:
            image_features = None
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            image_feature=image_features
        )
        hidden_states = transformer_outputs[0]
        
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(hidden_states.device)
                if image_features is not None:
                    sequence_lengths += image_features.shape[1] - 1 
                if self.kwargs.mICL:
                    _, image_emb_loc = torch.where(input_ids == self.image_emb) 
                    image_emb_loc = image_emb_loc - 1 + image_features.shape[1] - 1 
            else:
                sequence_lengths = -1

        pooled_logits = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        if self.kwargs.mICL:
            image_logits = hidden_states[torch.arange(batch_size, device=hidden_states.device), image_emb_loc]
        
        if self.late_fusion:
            pooled_logits = self.gate(pooled_logits, clip_feature)
            if self.kwargs.mICL:
                image_logits = self.gate(image_logits, clip_feature)
        
        if self.dim_reduction!=-1:
            pooled_logits = self.linear(pooled_logits)
            if self.kwargs.mICL:
                image_logits = self.linear(image_logits)

        def get_image_pos(input_ids):
            image_token_indices = torch.where(input_ids == self.model.image_token_id)[1]
            image_start = image_token_indices
            image_end = image_token_indices + image_features.shape[1] - 1
            if self.kwargs.mICL:
                image_end = image_emb_loc + 3 # Find the position of the compression symbol and extend the length of the image-related tokens.
            return (image_start, image_end)
        
        def get_proportion(saliency, image_poss, final_poss):
            saliency = saliency.detach().clone().cpu()
            (image_starts, image_ends) = image_poss
            image_starts = image_starts.detach().clone().cpu()
            image_ends = image_ends.detach().clone().cpu()
            final_poss = final_poss.detach().clone().cpu()
            saliency = saliency.to(torch.float32).numpy()
            for i in range(saliency.shape[0]):
                np.fill_diagonal(saliency[i], 0)
            saliency[np.isnan(saliency)] = 0
            proportions = []

            for sample, (image_start, image_end, final_pos) in enumerate(zip(image_starts, image_ends, final_poss)):
                image_len = image_end-image_start+1
                remain_len = final_pos-image_len
                proportion1 = saliency[sample, final_pos, image_start:image_end+1].sum()
                all_tokens = np.arange(saliency.shape[2])
                non_image_tokens = np.delete(all_tokens, np.arange(image_start, image_end+1))
                proportion2 = saliency[sample, final_pos, non_image_tokens].sum()
                proportion3 = saliency[sample].sum() - proportion1 - proportion2

                N = int(final_pos)
                sum3 = (N + 1) * N / 2 - image_len - remain_len
                proportion1 = proportion1 / image_len
                proportion2 = proportion2 / remain_len
                proportion3 = proportion3 / sum3
                proportion = np.array([proportion1, proportion2, proportion3])
                proportions.append(proportion)
            
            return proportions

        loss = self.concat_compute_oneloss(pooled_logits,batch_size//2)
        print(loss)
        loss.backward()
        image_poss = get_image_pos(input_ids)
        final_poss = sequence_lengths
        pros = []
        for i in range(len(self.attentionermanger.attention_adapters)):
            saliency = self.attentionermanger.grad(use_abs=True)[i] # notice self.attentionermanger.zero_grad()
            pro = get_proportion(saliency, image_poss, final_poss)
            pro = np.stack(pro,axis=0)
            # pro: [batch,3]
            pros.append(pro)
        # pros: [batch, layers, 3]
        pros = np.stack(pros,axis=1)
        self.pros_list.append(pros)
        if len(self.pros_list)==1000:
            self.pros_list = np.concatenate(self.pros_list,axis=0)
            exp_name = self.kwargs.model_type
            root_file = '/root/path'
            save_name = 'attn_score.pkl'
            file_path = os.path.join(root_file , exp_name , save_name)
            import pickle
            with open(file_path, 'wb') as f:
                pickle.dump(self.pros_list, f)
            raise
        
        loss = torch.zeros(1,device=loss.device,requires_grad=True).squeeze()

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        if self.kwargs.mICL:
            return mICL_SequenceClassifierOutputWithPast(
                loss=loss,
                logits=pooled_logits,
                text_logits=pooled_logits,
                image_logits=image_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )
        else:
            return SequenceClassifierOutputWithPast(
                loss=loss,
                logits=pooled_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )
        


class MLlama(CustomMLLama):

    def __init__(self, config, kwargs=None):
        super().__init__(config, kwargs)
        self.kwargs = kwargs
        self.vision_tower = build_vision_tower(kwargs, delay_load=False)
        self.query_tokens = nn.Parameter(torch.zeros(1, kwargs.num_query_tokens, kwargs.mm_hidden_size))
        print(f'query token size: {self.query_tokens.shape}')
        qformer_config = Blip2QFormerConfig(vocab_size=kwargs.blip_vocab_size,num_hidden_layers=kwargs.blip_num_hidden_layers,encoder_hidden_size=kwargs.mm_hidden_size)
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
        
        from .AttentionAdapter import LLAMAAttentionerManager
        self.attentionermanger = LLAMAAttentionerManager(self.model)
        self.pros_list = []

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


class LateFusion(CustomMLLama):

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