import math
import os

import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaModel,
    LlamaPreTrainedModel,
    Qwen2PreTrainedModel,
    Qwen2Model,
    PreTrainedModel
)


def build_rm(model_name, **kwargs):
    if "01-AI/Yi-34B" in model_name or "01-AI/Yi-132B" in model_name or "01-AI/Yi-9B" in model_name:
        assert 'model_path' in kwargs
        model_path = kwargs['model_path']
        del kwargs['model_path']
        kwargs['with_bias'] = False

        reward_model = Llama4SequenceClassification.from_pretrained(model_path, **kwargs)
    elif "01-AI/Yi-34B-LF" in model_name or "01-AI/Yi-132B-LF" in model_name or "01-AI/Yi-9B-LF" in model_name:
        assert 'model_path' in kwargs
        model_path = kwargs['model_path']
        del kwargs['model_path']
        kwargs['with_bias'] = True
        
        reward_model = Llama4SequenceClassification.from_pretrained(model_path, **kwargs)
    elif "Qwen2-72B-LF" in model_name:
        assert 'model_path' in kwargs
        model_path = kwargs['model_path']
        del kwargs['model_path']

        kwargs['with_bias'] = True
        reward_model = Qwen2RewardModel.from_pretrained(model_path, **kwargs)
        
    else:
        raise ValueError(
            f"Model {model_name} not found in Starling reward models. Supported are {SUPPORTED_STARLING_MODELS}"
        )

    reward_model.eval().requires_grad_(False)
    return reward_model

class Llama4SequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config, with_bias=False):
        super().__init__(config)
        self.transformer = LlamaModel(config)
        self.v_head = nn.Linear(config.hidden_size, 1, bias=with_bias)
        self.PAD_ID = 0
        # Initialize weights and apply final processing
        self.post_init()

    def get_device(self):
        return self.transformer.device

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        
        hidden_states = transformer_outputs.hidden_states[-1]
        scores = []
        rewards = self.v_head(hidden_states).squeeze(-1)
        bs = int(input_ids.shape[0])
        for i in range(bs):
            c_inds = (input_ids[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else input_ids.shape[1]
            scores.append(rewards[i, c_ind - 1])
        scores = torch.stack(scores)
        return scores

class Qwen2RewardModel(Qwen2PreTrainedModel):
    def __init__(self, config, with_bias=False):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.v_head = nn.Linear(config.hidden_size, 1, bias=with_bias)
        self.PAD_ID = 151643
        # Initialize weights and apply final processing
        self.post_init()

    def get_device(self):
        return self.model.device

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
    ):
        """
        input_ids, attention_mask: torch.Size([bs, seq_len])
        return: scores: List[bs]
        """
        bs = input_ids.shape[0]

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        hidden_states = transformer_outputs[0]
        scores = []
        rewards = self.v_head(hidden_states).squeeze(-1)
        for i in range(bs):
            c_inds = (input_ids[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else input_ids.shape[1]
            scores.append(rewards[i, c_ind - 1])

        scores = torch.stack(scores)
        return scores

