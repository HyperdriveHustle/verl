# Copyright 2025 Individual Contributor: Mert Unsal
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

from collections import defaultdict

import torch

from verl import DataProto
from verl.workers.reward_manager import register

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
import os
import json

save_num_examine_path_remote_batch = os.environ.get("SAVE_NUM_EXAMINE_PATH_REMOTE_BATCH", "/afs/chatrl/users/hwq/log/verl/logs_sensecore/save_num_examine_remote_batch.jsonl")  # 允许默认路径







@register("remote_batch")
class REMOTEBatchRewardManager:
    """
    A batch reward manager that computes rewards for a batch of data.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for decoding the responses.
        num_examine (int): The number of responses to examine.
        compute_score (callable): The function to compute the rewards.
        reward_fn_key (str): The key to use for the reward function.
        reward_kwargs (dict): The keyword arguments to pass to the reward function.
    """

    def __init__(self,
                 tokenizer,
                 num_examine,
                 compute_score=None,
                 reward_fn_key="data_source",
                 remote_reward_cfg=None,
                 max_resp_len=None,
                 overlong_buffer_cfg=None,
                 **reward_kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.remote_reward_cfg = remote_reward_cfg
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.reward_kwargs = reward_kwargs

        from verl.utils.reward_score.remote_reward_batch import init_remote_reward
        init_remote_reward(remote_reward_cfg or {})

    def verify(self, data):
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)

        ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extras = data.non_tensor_batch.get("extra_info", [None] * len(data))
        prompts_into_extras = data.non_tensor_batch.get("prompt", [None] * len(data))

        scores = self.compute_score(
            data_source=data_sources,
            solution_str=responses_str,
            ground_truth=ground_truths,
            extra_info=prompts_into_extras,
            **self.reward_kwargs,
        )

        return scores

    def __call__(self, data: DataProto, return_dict=False):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        scores = self.verify(data)
        rewards = []
        already_printed = {}

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            score = scores[i]

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            rewards.append(reward)
            reward_tensor[i, length - 1] = reward

            data_source = data_sources[i]
            response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
            prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
            ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
            # 保存为 JSONL
            output_item = {
                "data_source": data_source,
                "prompt": prompt_str,
                "response": response_str,
                "ground_truth": ground_truth,
                "score": scores[i]
            }
            with open(save_num_examine_path_remote_batch, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(output_item, ensure_ascii=False) + "\n")

            if already_printed.get(data_source, 0) < self.num_examine:
            # if already_printed.get(data_source, 0) < 500:
                # response_str = self.tokenizer.decode(data.batch["responses"][i][:length], skip_special_tokens=True)
                # prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                # ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", scores[i])

                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
