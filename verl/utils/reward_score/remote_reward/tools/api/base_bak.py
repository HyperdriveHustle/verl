from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

from tqdm import tqdm
import openai

from verl.utils.reward_score.remote_reward.tools.base import Tool, GenerateTool


class OpenAIClientTool(GenerateTool):
    def __init__(self, name: str, api_key: str = None, base_url: str = None) -> None:
        self.name = name
        self.api_key = api_key
        self.base_url = base_url

        assert self.api_key, 'api_key is required'
        assert self.base_url, 'base_url is required'

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

        completion = None

        try:
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                n=n,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                stop=stop
            )
            return [choice.message.content for choice in completion.choices]
        except Exception as e:
            print(f">>> Exception while process {model_name} api, completion: {completion}, error: {e}")
            return None

    def batch_generate(self, batch_messages: List[List[Dict[str, str]]], **kwargs) -> List[List[str]]:
        assert 'model_name' in kwargs, 'model_name is required'
        workers = kwargs.get('workers', 10)

        model_name = kwargs['model_name']
        temperature = kwargs.get('temperature', 0.0)
        top_p = kwargs.get('top_p', 1.0)
        max_tokens = kwargs.get('max_tokens', None)
        n = kwargs.get('n', 1)
        presence_penalty = kwargs.get('presence_penalty', 0.0)
        frequency_penalty = kwargs.get('frequency_penalty', 0.0)
        stop = kwargs.get('stop', None)

        responses = [None] * len(batch_messages)  # Initialize a list with None to hold the responses

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(
                    self.client.chat.completions.create,
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    max_tokens=max_tokens,
                    stop=stop
                ): index for index, messages in enumerate(batch_messages)
            }

            for future in tqdm(as_completed(future_to_index), desc=f"{model_name} api processing", total=len(batch_messages)):
                index = future_to_index[future]
                completion = None
                try:
                    completion = future.result()
                    responses[index] = [choice.message.content for choice in completion.choices]  # Store the response at the correct index
                except Exception as e:
                    print(f">>> Exception while processing {model_name} API, completion: {completion}, error: {e}")
                    responses[index] = None  # In case of an exception, store None at the correct index

        return responses