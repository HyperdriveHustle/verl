from typing import List, Dict, Any

import vllm
from transformers import AutoTokenizer

from harpy.tools.base import Tool, SortTool
from harpy.tools.rm.model_builder import build_rm
import torch

class RMTool(SortTool):
    def __init__(self, name: str, model_path: str, tokenizer_path: str = None, **kwargs) -> None:
        self.name = name
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path if tokenizer_path else model_path

        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16
            # "attn_implementation": "flash_attention_2"
        }

        self.model = build_rm(name, model_path = model_path, **kwargs, **model_kwargs)
        if 'Qwen2' in name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=kwargs.get('trust_remote_code', None))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=False, trust_remote_code=kwargs.get('trust_remote_code', None))

    def sort(self, responses: List[str], messages: List[Dict[str, str]], **kwargs) -> List[Dict[str, Any]]:
        ## nulti trun not supported
        assert len(messages) == 1
        # scores = []
        res = []
        ## 每一个样本单独推理，避免pad影响
        for resp in responses:
            actual_messages = messages + [{"role":"assistant","content":resp}]
            inputs = self.tokenizer.apply_chat_template(actual_messages, tokenize=False)
            inp = self.tokenizer(inputs, return_tensors="pt").to("cuda")
            rewards = self.model(**inp)

            if isinstance(rewards, dict):
                score = rewards[0]['score']
            # for classes that directly output scores (custom code)
            else:
                score = rewards.to(torch.float).cpu().numpy().tolist()[0]

            res.append({'response':resp,'score':score})

        res = sorted(res, key=lambda item:item['score'], reverse=True)
        return res
