from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

from tqdm import tqdm
import openai
from openai import AzureOpenAI


from verl.utils.reward_score.remote_reward.tools.base import Tool, GenerateTool


class OpenAIClientTool(GenerateTool):
    def __init__(self, name: str, api_key: str = None, api_type: str = None, api_version: str = None, base_url: str = None) -> None:
        self.name = name
        self.api_key = api_key
        self.api_type = api_type
        self.base_url = base_url
        self.api_version = api_version

        assert self.api_key, 'api_key is required'
        assert self.base_url, 'base_url is required'

        if self.api_type is not None and self.api_type == "azure":
            assert self.api_version, 'api_version is required'
            self.client = AzureOpenAI(
                azure_endpoint=self.base_url,
                api_key=self.api_key, 
                api_version=self.api_version
            )
        else:
            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

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
        # from pprint import pprint
        # pprint(call_kwargs)

        completion = None


        try:
            completion = self.client.chat.completions.create(**call_kwargs)
            return [choice.message.content for choice in completion.choices]
        except Exception as e:
            print(f">>> Exception while process {model_name} api, completion: {completion}, error: {e}")
            return None

    def get_completion(self, input_item, model_name, temperature, top_p, n, frequency_penalty, presence_penalty, max_tokens, stop):
        messages = input_item["messages"]
        # messages = input_item
        # print(f">>>> get_completion, input_item = {input_item}")
        max_retry = 10
        while max_retry > 0:
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
                print(f">>> Exception while get_completion, error: {e}")
                max_retry -= 1
        return {"id": input_item["id"], "completions": None}
    

    def batch_generate(self, batch_messages, **kwargs) -> List[List[str]]:
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
        
        # Initialize a list with None to hold the responses
        responses = {}
        for index, x in enumerate(batch_messages):
            if "id" not in x:
                x = {"id": index, "messages": x}
                batch_messages[index] = x
            responses[index] = {"id": x["id"], "response_index": x["response_index"], "responses": [None]}

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

            # for future in tqdm(as_completed(future_to_index), desc=f"{model_name} api processing", total=len(batch_messages)):
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
                    # print(f"> result, responses[{index}] = ", responses[index])
                except Exception as e:
                    print(f">>> Exception while processing {model_name} API, completion: {completion}, error: {e}")
                    response_index = batch_messages[index].get("response_index", None)
                    responses[index] = {"id": index,
                                        "response_index": response_index,
                                        "responses": [None]}  # In case of an exception, store None at the correct index
        # print(f"> api return responses = {responses}")
        return responses