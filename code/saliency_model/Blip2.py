from transformers import Blip2Model
import torch
import torch.nn as nn
from typing import Any, Optional, Tuple, Union
import torch.nn.functional as F
from .Llama import concat_all_gather
from dataclasses import dataclass
from transformers.utils import ModelOutput

from .Gate import Gate

import numpy as np
import os


@dataclass
class Blip2ForConditionalGenerationModelOutput(ModelOutput):
    """
    Class defining the outputs of [`Blip2ForConditionalGeneration`].

    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Language modeling loss from the language model.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head of the language model.
        vision_outputs (`BaseModelOutputWithPooling`):
            Outputs of the vision encoder.
        qformer_outputs (`BaseModelOutputWithPoolingAndCrossAttentions`):
            Outputs of the Q-Former (Querying Transformer).
        language_model_outputs (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`):
            Outputs of the language model.
    """

    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    vision_outputs: Optional[torch.FloatTensor] = None
    qformer_outputs: Optional[Tuple[torch.FloatTensor]] = None
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["vision_outputs", "qformer_outputs", "language_model_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )

@dataclass
class ICL3_Blip2ForConditionalGenerationModelOutput(Blip2ForConditionalGenerationModelOutput):
    image_logits: Optional[Tuple[torch.FloatTensor]] = None

class BLIP2_INI(Blip2Model):

    def __init__(self, config, kwargs=None):
        super().__init__(config)
        self.kwargs = kwargs
        self.resize_token_embeddings(getattr(config,'vocab_size',None))
        self.config.hidden_size = config.text_config.hidden_size
        self.dim_reduction = kwargs.dim_reduction
        if self.dim_reduction!=-1:
            self.linear = nn.Linear(config.text_config.hidden_size, self.dim_reduction, bias=False)
        self.temperature = nn.Parameter(torch.tensor([kwargs.temperature]))
        self.image_token_id = -1
        self.late_fusion = kwargs.late_fusion

        if kwargs.fix_image_encoder:
            self.vision_model.requires_grad_(False)
        if kwargs.late_fusion:
            self.mm_projector_latefusion = nn.Linear(kwargs.mm_hidden_size, kwargs.hidden_size, bias=False)
            self.gate = Gate(kwargs.hidden_size)
            if kwargs.fix_late_fusion:
                self.mm_projector_latefusion.requires_grad_(False)
                self.gate.requires_grad_(False)

        from .AttentionAdapter import OPTAttentionerManager
        self.attentionermanger = OPTAttentionerManager(self.language_model)
        self.pros_list = []

        self.post_init()
    
    def prepare_inputs_embeds_for_multimodal(
        self, input_ids, images
    ):  
        image_token_indices = torch.where(input_ids == self.image_token_id)[1]
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        multimodal_embeds = []
        for inputs_embed, image, image_token_indice in zip(inputs_embeds, images, image_token_indices):
            multimodal_embed = [inputs_embed[:image_token_indice],image,inputs_embed[image_token_indice+1:]]
            multimodal_embed = torch.cat(multimodal_embed,dim=0)
            multimodal_embeds.append(multimodal_embed)
        multimodal_embeds = torch.stack(multimodal_embeds,dim=0)
        return multimodal_embeds

    def forward(
        self,
        input_ids: torch.FloatTensor,
        images: torch.FloatTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        self.attentionermanger.zero_grad()
        pixel_values = images
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if images is not None:
            # step 1: forward the images through the vision encoder,
            # to get image embeddings of shape (batch_size, seq_len, hidden_size)
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            image_embeds = vision_outputs[0]
            if self.late_fusion:
                clip_feature = vision_outputs[1]
                clip_feature = self.mm_projector_latefusion(clip_feature)

            # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
            image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs = self.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            query_output = query_outputs[0]

            # step 3: use the language model, conditioned on the query outputs and the prompt
            language_model_inputs = self.language_projection(query_output)
            language_model_attention_mask = torch.ones(
                language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
            )[:,:-1]
            inputs_embeds = self.prepare_inputs_embeds_for_multimodal(input_ids,language_model_inputs)

            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            expected_device = language_model_attention_mask.device
            attention_mask = torch.cat([language_model_attention_mask, attention_mask.to(expected_device)], dim=1)
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        if self.config.use_decoder_only_language_model:
            outputs = self.language_model.model.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            logits = outputs.last_hidden_state if return_dict else outputs[0]

            # we compute the loss here since we need to take into account the sequence length of the query embeds
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
                if images is not None:
                    sequence_lengths += language_model_inputs.shape[1] - 1 
                if self.kwargs.mICL:
                    _, image_emb_loc = torch.where(input_ids == self.image_emb) 
                    image_emb_loc = image_emb_loc - 1 + language_model_inputs.shape[1] - 1 
            else:
                sequence_lengths = -1
            
            batch_size = logits.shape[0]
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

            def get_image_pos(input_ids):
                image_token_indices = torch.where(input_ids == self.image_token_id)[1]
                image_start = image_token_indices
                image_end = image_token_indices + query_output.shape[1] - 1
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
                    # print(image_start, image_end, final_pos)
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

        else:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        if self.kwargs.mICL:
            return ICL3_Blip2ForConditionalGenerationModelOutput(
                loss=loss,
                logits=pooled_logits,
                image_logits=image_logits,
                vision_outputs=vision_outputs if images is not None else None,
                qformer_outputs=query_outputs if images is not None else None,
                language_model_outputs=outputs,
            )
        else:
            return Blip2ForConditionalGenerationModelOutput(
                loss=loss,
                logits=pooled_logits,
                vision_outputs=vision_outputs if images is not None else None,
                qformer_outputs=query_outputs if images is not None else None,
                language_model_outputs=outputs,
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