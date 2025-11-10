# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import asyncio
import logging
import os

import ray
from omegaconf import DictConfig

from verl.experimental.reward.reward_model import (
    compute_reward_result_tag,
    IMPROVED_PROMPT_TEMPLATE,
    BASE_URL,
    API_KEY,
    MODEL_NAME,
    MAX_WORKERS,
    OPENAI_TIMEOUT,
    OPENAI_MAX_RETRIES,
)
from verl.protocol import DataProto
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.experimental.reward.api.base import OpenAIClientTool

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@ray.remote
class RewardManagerWorker:
    def __init__(self, config: DictConfig, reward_router_address: str = None):
        self.config = config
        self.reward_router_address = reward_router_address
        self._init_reward_fn()

    def _init_reward_fn(self):
        input_tokenizer_local_path = copy_to_local(self.config.actor_rollout_ref.model.path)
        self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path, trust_remote_code=True)
        
        # 初始化 LLM-as-a-Judge 客户端（使用远程API）
        self.judge_client = OpenAIClientTool(
            name=MODEL_NAME,
            api_key=API_KEY,
            base_url=BASE_URL,
            timeout=OPENAI_TIMEOUT,
            max_retries=OPENAI_MAX_RETRIES
        )
        logger.info(f"RewardManagerWorker initialized with remote LLM-judge: {BASE_URL}, model: {MODEL_NAME}, timeout: {OPENAI_TIMEOUT}s, max_retries: {OPENAI_MAX_RETRIES}")

    async def compute_score(self, data: DataProto) -> dict:
        """计算单个样本的奖励分数
        
        Args:
            data: DataProto containing single data item with fields:
                - batch["responses"]: response token ids
                - batch["attention_mask"]: attention mask
                - non_tensor_batch["reward_model"]: dict with "ground_truth"
                - non_tensor_batch["extra_info"]: dict with "raw_problem" and other fields
                - non_tensor_batch["data_source"]: data source identifier
                - non_tensor_batch["raw_problem"]: optional raw problem text
        
        Returns:
            dict with keys:
                - "reward_score": float, computed reward score
                - "reward_extra_info": dict with "acc", "pred"
        """
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        # 将张量标量转换为 Python 整数用于切片
        valid_response_length_int = int(valid_response_length.item() if hasattr(valid_response_length, "item") else int(valid_response_length))
        valid_response_ids = response_ids[:valid_response_length_int]

        # 从data_item中获取完整的数据字段
        reward_model_data = data_item.non_tensor_batch.get("reward_model", {})
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        data_source = data_item.non_tensor_batch.get("data_source", None)
        
        # 获取 ground_truth（优先从 reward_model 中获取）
        ground_truth = None
        if isinstance(reward_model_data, dict) and "ground_truth" in reward_model_data:
            ground_truth = reward_model_data["ground_truth"]
        
        # 解码响应文本（从张量解码）
        response_str = await asyncio.get_running_loop().run_in_executor(
            None, lambda: self.input_tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )
        
        # 截断并提取答案（取最后三行，与参考代码保持一致）
        response_str_truncated = response_str[-300:]
        lines = response_str_truncated.strip().split('\n')
        last_three_lines = lines[-3:] if len(lines) >= 3 else lines
        pred = '\n'.join(last_three_lines)
        
        # 获取问题文本（raw_problem）
        # 优先级：1. extra_info["raw_problem"] > 2. non_tensor_batch["raw_problem"] > 3. other fields
        problem_text = None
        
        # 优先级1：从 extra_info 中获取 raw_problem
        if isinstance(extra_info, dict) and "raw_problem" in extra_info:
            problem_text = extra_info["raw_problem"]
        
        # 优先级2：直接从 non_tensor_batch 获取 raw_problem
        if problem_text is None and "raw_problem" in data_item.non_tensor_batch:
            problem_text = data_item.non_tensor_batch["raw_problem"]
        
        # 优先级3：如果以上都没有，尝试从其他字段获取
        if problem_text is None:
            # 尝试获取 problem 字段
            candidate = data_item.non_tensor_batch.get('problem', None)
            if candidate and candidate != 'None':
                problem_text = str(candidate)
            else:
                # 最后的备选：从 prompt 中提取
                prompt = data_item.non_tensor_batch.get('prompt', None)
                if prompt:
                    if isinstance(prompt, list) and len(prompt) > 0:
                        v = prompt[0]
                        if isinstance(v, dict) and v.get('content'):
                            problem_text = v['content']
                        else:
                            problem_text = str(v)
                    else:
                        problem_text = str(prompt)
                else:
                    problem_text = str(extra_info) if extra_info else ""
        
        # 构建judge prompt
        prompt = IMPROVED_PROMPT_TEMPLATE.format(
            problem=problem_text if problem_text else "",
            pred=pred if pred else "",
            ground_truth=ground_truth if ground_truth else ""
        )
        messages = [{"role": "user", "content": prompt}]
        
        gen_args = {
            "model_name": MODEL_NAME,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 500,
            "workers": MAX_WORKERS
        }
        
        judge_response = None
        try:
            # 使用线程池执行同步的generate调用，避免阻塞事件循环
            judge_output = await asyncio.get_running_loop().run_in_executor(
                None, lambda: self.judge_client.generate(messages, **gen_args)
            )
            judge_response = judge_output[0] if judge_output else None
        except Exception as e:
            logger.error(f"Judge API call failed: {e}")
            judge_response = None
        
        # 计算奖励（与规则计算保持一致：正确=1.0，错误=-1.0，格式错误=-1.0）
        # 检查格式：如果pred为空或只包含空白，认为格式错误
        if pred is not None and pred.strip() != "":
            format_correct = 1.0
        else:
            format_correct = -1.0

        # 使用LLM-judge判断答案正确性
        if judge_response is not None:
            answer_correct = compute_reward_result_tag(judge_response)
        else:
            # Judge失败时，返回-1.0（错误），与规则计算保持一致
            logger.warning("Judge API call failed or returned None, treating as incorrect answer")
            answer_correct = -1.0

        # 奖励计算逻辑：格式错误时返回-1.0，否则返回答案正确性（1.0或-1.0）
        if format_correct < 0:
            reward = format_correct  # -1.0（格式错误）
        else:
            reward = answer_correct  # 1.0（正确）或-1.0（错误）

        # 构建奖励额外信息
        reward_extra_info = {
            "acc": 1.0 if answer_correct == 1.0 else 0.0,  # acc为1.0表示正确，0.0表示错误
            "pred": "" if pred is None else pred,
        }
        
        # 返回上游预期的结构（dict）
        result = {
            "reward_score": float(reward),
            "reward_extra_info": reward_extra_info,
        }
        
        logger.debug(
            f"[RewardManagerWorker.compute_score] "
            f"reward_score={result['reward_score']}, "
            f"acc={reward_extra_info.get('acc', 'N/A')}, "
            f"pred_len={len(reward_extra_info.get('pred', ''))}, "
            f"judge_response_exists={judge_response is not None}"
        )
        
        return result
