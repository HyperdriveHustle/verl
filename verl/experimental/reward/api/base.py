# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import os
import time

from tqdm import tqdm
import openai
from openai import AzureOpenAI


class OpenAIClientTool:
    def __init__(self, name: str, api_key: str = None, api_type: str = None, api_version: str = None, base_url: str = None, timeout: float = None, max_retries: int = None) -> None:
        self.name = name
        self.api_key = api_key
        self.api_type = api_type
        self.base_url = base_url
        self.api_version = api_version
        
        # 从环境变量或参数获取超时和重试配置
        self.timeout = timeout if timeout is not None else float(os.getenv("OPENAI_TIMEOUT", "30.0"))  # 默认600秒
        self.max_retries = max_retries if max_retries is not None else int(os.getenv("OPENAI_MAX_RETRIES", "3"))  # 默认3次重试

        assert self.api_key, 'api_key is required'
        assert self.base_url, 'base_url is required'

        if self.api_type is not None and self.api_type == "azure":
            assert self.api_version, 'api_version is required'
            self.client = AzureOpenAI(
                azure_endpoint=self.base_url,
                api_key=self.api_key, 
                api_version=self.api_version,
                timeout=self.timeout,
                max_retries=self.max_retries
            )
        else:
            self.client = openai.OpenAI(
                api_key=self.api_key, 
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries
            )

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> List[str]:
        assert 'model_name' in kwargs, 'model_name is required'
        model_name = kwargs['model_name']
        temperature = kwargs.get('temperature', 0.0)
        top_p = kwargs.get('top_p', 1.0)
        max_tokens = kwargs.get('max_tokens', None)
        n = kwargs.get('n', 1)
        presence_penalty = kwargs.get('presence_penalty', 0.0)
        frequency_penalty = kwargs.get('frequency_penalty', 0.0)
        stop = kwargs.get('stop', None)
        
        # 从kwargs获取重试配置，如果没有则使用实例默认值
        max_retries = kwargs.get('max_retries', self.max_retries)
        retry_delay = kwargs.get('retry_delay', 2.0)  # 重试延迟（秒）

        call_kwargs = dict(
            model=model_name,
            messages=messages,
            n=n,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            stop=stop,
        )
        # 仅在 Qwen3 系列模型时添加 extra_body
        if model_name.lower().startswith("qwen3"):
            call_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }

        completion = None
        last_exception = None
        
        # 实现重试机制
        for attempt in range(max_retries + 1):  # 总共尝试 max_retries + 1 次
            try:
                completion = self.client.chat.completions.create(**call_kwargs)
                return [choice.message.content for choice in completion.choices]
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    # 如果不是最后一次尝试，等待后重试
                    wait_time = retry_delay * (2 ** attempt)  # 指数退避
                    print(f">>> Exception while process {model_name} api (attempt {attempt + 1}/{max_retries + 1}), error: {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    # 最后一次尝试失败，记录错误并返回None
                    print(f">>> Exception while process {model_name} api, completion: {completion}, error: {e} (after {max_retries + 1} attempts)")
        
        return None

    def get_completion(self, input_item, model_name, temperature, top_p, n, frequency_penalty, presence_penalty, max_tokens, stop):
        messages = input_item["messages"]
        max_retry = self.max_retries + 1  # 使用实例的max_retries配置
        retry_delay = 2.0  # 重试延迟（秒）
        last_exception = None
        
        for attempt in range(max_retry):
            try:
                completions = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}}, # nothinking
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    max_tokens=max_tokens,
                    stop=stop
                )
                return {"id": input_item["id"], "completions": completions}
            except Exception as e:
                last_exception = e
                if attempt < max_retry - 1:
                    # 如果不是最后一次尝试，等待后重试
                    wait_time = retry_delay * (2 ** attempt)  # 指数退避
                    print(f">>> Exception while get_completion (attempt {attempt + 1}/{max_retry}), error: {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    print(f">>> Exception while get_completion, error: {e} (after {max_retry} attempts)")
        return {"id": input_item["id"], "completions": None}
    

    def batch_generate(self, batch_messages, **kwargs) -> Dict:
        assert 'model_name' in kwargs, 'model_name is required'
        workers = kwargs.get('workers', 5)
        
        model_name = kwargs['model_name']
        temperature = kwargs.get('temperature', 0.0)
        top_p = kwargs.get('top_p', 1.0)
        max_tokens = kwargs.get('max_tokens', None)
        n = kwargs.get('n', 1)
        presence_penalty = kwargs.get('presence_penalty', 0.0)
        frequency_penalty = kwargs.get('frequency_penalty', 0.0)
        stop = kwargs.get('stop', None)
        
        # Initialize a dict to hold the responses
        responses = {}
        for index, x in enumerate(batch_messages):
            if "id" not in x:
                x = {"id": index, "messages": x}
                batch_messages[index] = x
            responses[index] = {"id": x["id"], "response_index": x.get("response_index", 0), "responses": [None]}

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(
                    self.get_completion,
                    input_item=msg,
                    model_name=model_name,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    max_tokens=max_tokens,
                    stop=stop

            ): index for index, msg in enumerate(batch_messages)
            }

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                completion = None
                try:
                    result = future.result()
                    xid, completion = result["id"], result["completions"]
                    response_index = batch_messages[index].get("response_index", None)

                    responses[index] = {"id": xid,
                                        "response_index": response_index,
                                        "responses": [choice.message.content for choice in completion.choices]}  # Store the response at the correct index
                except Exception as e:
                    print(f">>> Exception while processing {model_name} API, completion: {completion}, error: {e}")
                    response_index = batch_messages[index].get("response_index", None)
                    responses[index] = {"id": index,
                                        "response_index": response_index,
                                        "responses": [None]}  # In case of an exception, store None at the correct index
        return responses

