from typing import List, Dict

import vllm
from transformers import AutoTokenizer, GenerationConfig

from harpy.tools.base import Tool, GenerateTool


class VllmTool(GenerateTool):
    def __init__(self, name: str, model_path: str, tokenizer_path: str = None, generation_config_path: str = None, **kwargs) -> None:
        self.name = name
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path if tokenizer_path else model_path
        self.generation_config_path = generation_config_path if generation_config_path else model_path

        self.model = vllm.LLM(
            model=self.model_path,
            tokenizer=self.tokenizer_path,
            enable_chunked_prefill = True, 
            enable_prefix_caching = False,
            **kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=False, trust_remote_code=kwargs.get('trust_remote_code', None))
        self.generation_config = GenerationConfig.from_pretrained(self.generation_config_path)

        if self.generation_config and self.generation_config.eos_token_id:
            if isinstance(self.generation_config.eos_token_id, int):
                self.stop_token_ids = [self.generation_config.eos_token_id]
            elif isinstance(self.generation_config.eos_token_id, list):
                self.stop_token_ids = self.generation_config.eos_token_id
        
        if self.generation_config and hasattr(self.generation_config, "eos_token") and self.generation_config.eos_token:
            if isinstance(self.generation_config.eos_token, str):
                self.stop = [self.generation_config.eos_token]
            elif isinstance(self.generation_config.eos_token, list):
                self.stop = self.generation_config.eos_token

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> List[str]:
        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        sampling_params = self.extract_sampling_params(**kwargs)
        request_outputs = self.model.generate(
            prompt_token_ids=[input_ids],
            sampling_params=sampling_params
        )
        return [output.text for output in request_outputs[0].outputs]

    def batch_generate(self, batch_messages: List[List[Dict[str, str]]], **kwargs) -> List[List[str]]:
        input_ids = [self.tokenizer.apply_chat_template(messages, add_generation_prompt=True) for messages in batch_messages]
        sampling_params = self.extract_sampling_params(**kwargs)
        request_outputs = self.model.generate(
            prompt_token_ids=input_ids,
            sampling_params=sampling_params
        )
        batch_responses = []
        for request_output in request_outputs:
            responses = []
            for output in request_output.outputs:
                responses.append(output.text)
            batch_responses.append(responses)
        return batch_responses

    def extract_sampling_params(self, **kwargs):
        if 'stop_token_ids' not in kwargs and self.stop_token_ids:
            kwargs['stop_token_ids'] = self.stop_token_ids
        if 'stop' not in kwargs and hasattr(self, "stop") and self.stop:
            kwargs['stop'] = self.stop
        sampling_params = vllm.SamplingParams(**kwargs)
        print(f"> sampling_params = {sampling_params}")
        return sampling_params