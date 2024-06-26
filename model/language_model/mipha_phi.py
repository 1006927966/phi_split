import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
    PhiModel, PhiPreTrainedModel

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..mipha_arch  import MiphaMetaModel, MiphaMetaForCausalLM

from transformers.utils import logging
from .configuration_mipha import MiphaPhiConfig

logger = logging.get_logger(__name__)


class MiphaPhiModel(MiphaMetaModel, PhiModel):
    config_class = MiphaPhiConfig

    def __init__(self, config):
        super(MiphaPhiModel, self).__init__(config)

# 基于图像进行生成，然后计算loss 交叉熵
class MiphaPhiForCausalLM(PhiPreTrainedModel, MiphaMetaForCausalLM): #
    config_class = MiphaPhiConfig #只有视觉和对齐模块的config
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super(PhiPreTrainedModel, self).__init__(config)
        self.model = MiphaPhiModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation( # 模型的输入以字典形式展现
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # 尺度input_ids = [batch, length]
        # past_key_values [batch, T, length]*2
        # attention mask = [TXT]
        # inputs_embeds [batch, T, length]

        if past_key_values:
            input_ids = input_ids[:, -1:] # 最后一位的indexes

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        # 如果有embeds 输入embeds 如果没有那么则输入inputids
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs


AutoConfig.register("mipha_phi", MiphaPhiConfig)
AutoModelForCausalLM.register(MiphaPhiConfig, MiphaPhiForCausalLM)
