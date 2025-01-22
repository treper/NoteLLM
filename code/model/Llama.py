from transformers import LlamaModel, LlamaPreTrainedModel
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPast, SequenceClassifierOutputWithPast
import torch.nn.functional as F

def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    #print(f'all rank num : {torch.distributed.get_world_size()}')
    with torch.no_grad():
        tensors_gather = [
            torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    rank = torch.distributed.get_rank()
    tensors_gather[rank] = tensor
    output = torch.cat(tensors_gather, dim=0)
    return output, rank

class CustomLlamaModel(LlamaModel):

    def __init__(self, config):
        super().__init__(config)
        self.image_token_id = -1

    def prepare_inputs_embeds_for_multimodal(
        self, input_ids, images
    ):  
        image_token_indices = torch.where(input_ids == self.image_token_id)[1]
        inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds_before = [inputs_embed[:image_token_indice] for inputs_embed, image_token_indice in zip(inputs_embeds, image_token_indices)]
        inputs_embeds_after = [inputs_embed[image_token_indice+1:] for inputs_embed, image_token_indice in zip(inputs_embeds, image_token_indices)]
        
        inputs_embeds_before = torch.stack(inputs_embeds_before, dim=0)
        inputs_embeds_after = torch.stack(inputs_embeds_after, dim=0)
        
        multimodal_embeds = torch.cat([inputs_embeds_before, images, inputs_embeds_after], dim=1)
        return multimodal_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        image_feature: Optional[torch.FloatTensor] = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        
        if image_feature is not None:
            if past_key_values is None:
                seq_length_image = image_feature.shape[1]
                seq_length = seq_length - 1 + seq_length_image

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            if image_feature is not None:
                inputs_embeds = self.prepare_inputs_embeds_for_multimodal(input_ids,image_feature)
            else:
                inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None or image_feature is not None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length#, valid_lengths
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
class Llama(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, kwargs=None):
        super().__init__(config)
        self.dim_reduction = kwargs.dim_reduction
        self.model = CustomLlamaModel(config)
        if self.dim_reduction!=-1:
            self.linear = nn.Linear(config.hidden_size, self.dim_reduction, bias=False)
        self.temperature = nn.Parameter(torch.tensor([kwargs.temperature]))
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
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
            else:
                sequence_lengths = -1

        pooled_logits = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        
        if self.dim_reduction!=-1:
            pooled_logits = self.linear(pooled_logits)

        if not self.infer:
            loss = self.concat_compute_oneloss(pooled_logits,batch_size//2)
        else:
            loss = None

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
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

    def collect_logits(self,pooled_logits,input_ids,pair_num):
        bs = pooled_logits.shape[0]
        pooled_logits = pooled_logits/torch.norm(pooled_logits,dim=1,keepdim=True)
        query = pooled_logits[:pair_num]
        doc = pooled_logits[pair_num:]

        all_query, rank = concat_all_gather(query)
        all_doc, rank = concat_all_gather(doc)

        return all_query, all_doc
    
    def compute_dualloss(self,all_query,all_doc, return_logits=False):
        logits = (all_query @ all_doc.T) * torch.exp(self.temperature)
        n_samples = all_query.shape[0]
        labels = torch.arange(n_samples, device=logits.device, dtype=torch.long)
        loss = (F.cross_entropy(logits, labels, reduction='mean') + F.cross_entropy(logits.T, labels, reduction='mean'))/2
        if return_logits:
            return loss, logits
        return loss
