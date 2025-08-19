
import logging
import os
import re
import numpy as np
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

ANSWER_REWARD = 1.0
FORMAT_REARD = 0.1

PY_IMPORTS = """import heapq
import itertools
import random
import functools
import collections
import string
import math
import datetime

from typing import *
from functools import *
from collections import *
from itertools import *
from heapq import *
from bisect import *
from string import *
from operator import *
from math import *

inf = float('inf')

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def list_node(values: list):
    if not values:
        return None
    head = ListNode(values[0])
    p = head
    for val in values[1:]:
        node = ListNode(val)
        p.next = node
        p = node
    return head

def is_same_list(p1, p2):
    if p1 is None and p2 is None:
        return True
    if not p1 or not p2:
        return False
    return p1.val == p2.val and is_same_list(p1.next, p2.next)

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def tree_node(values: list):
    if not values:
        return None
    root = TreeNode(values[0])
    i = 1
    queue = deque()
    queue.append(root)
    while queue:
        node = queue.popleft()
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
            i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
            i += 1
    return root

def is_same_tree(p, q):
    if not p and not q:
        return True
    elif not p or not q:
        return False
    elif p.val != q.val:
        return False
    else:
        return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)

"""
@register("code_execution_agent_multi_turn")
class CodeExecutionAgentLoop_Multi_turn(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level CodeExecutionAgentLoop initialization")

        cls.tokenizer = tokenizer
        
        # 初始化用于奖励计算的工具
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        cls.max_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        if not cls.tools:
            logger.warning("CodeExecutionAgentLoop initialized, but no reward tool was found in the config.")
        else:
            cls.code_tool = next(iter(cls.tools.values()))
            print(f"Initialized reward tool: {cls.code_tool.name}")

        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True)
        
    def validate_response_structure(self, processed_str: str) -> bool:
        pattern = re.compile(r'<think>.*</think>.*<answer>.*</answer>$', re.DOTALL)
        return bool(pattern.match(processed_str.strip()))

    @rollout_trace_op
    async def run(self, messages: list[dict[str, Any]], sampling_params_w_test_code: dict[str, Any]) -> AgentLoopOutput:
        metrics = {}
        request_id = uuid4().hex
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            ),
        )

        sampling_params = sampling_params_w_test_code["sampling_params"]
        test_code = sampling_params_w_test_code["test_code"]
        response_mask = []
        turns = 0
        assistant_turns = 0
        
        answer_reward = 0.0
        timeout_reward = 0.0
        format_reward = 0.0
        is_validate = sampling_params_w_test_code["validate"]
        cur_max_turns = 1 if is_validate else self.max_turns
        while turns < cur_max_turns:
            turns += 1
            with simple_timer("generate_sequence", metrics):
                response_ids = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
                )
            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)
            assistant_turns += 1

            solution_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            format_check = self.validate_response_structure(solution_text)
            code_pattern = re.compile(
                r"<answer>.*?```(?:python\n)?(.*?)```.*?</answer>", 
                re.DOTALL
            )
            code_match = code_pattern.search(solution_text)
            if not code_match:
                code_match = re.search(r"```(?:python\n)?(.*?)```", solution_text, re.DOTALL)
            extracted_code = code_match.group(1).strip() if code_match else None
    
            format_reward += FORMAT_REARD if format_check else -FORMAT_REARD

            if extracted_code and hasattr(self, 'code_tool'):
                extracted_code_w_test = PY_IMPORTS + extracted_code + "\n" + test_code
                with simple_timer(f"tool_calls_turn_{turns}", metrics):
                    instance_id = None
                    try:
                        instance_id = await self.code_tool.create()
                        response, score, meta_data = await self.code_tool.execute(instance_id, {"code": extracted_code_w_test})
                        #breakpoint()
                        if meta_data["status"] == "timeout":
                            metrics["timeout"] = 1
                            error_message = "Code execution timeout, please reflect your answer and asnwer again to slove the problem based on the error message and your previous responses."
                        else:
                            if score.lower() == "success":
                                answer_reward = ANSWER_REWARD
                                metrics["success_at_turn"] = turns
                                break
                            else:
                                stdout = meta_data.get("stdout", "")
                                stderr = meta_data.get("stderr", "")
                                error_message = f"Code test failed.\n\nError message:\n{stdout}\n{stderr}\n\nPlease reflect your answer and asnwer again to slove the problem based on the error message and your previous responses."
                    except Exception as e:
                        breakpoint()
                        logger.error(f"Error during reward calculation: {e}")
                        answer_reward = 0.0
                        break
                    finally:
                        if instance_id:
                            await self.code_tool.release(instance_id)
                if turns >= cur_max_turns:
                    break
                tool_response_ids = await self.loop.run_in_executor(
                    None,
                    lambda messages=error_message: self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True
                    ),
                )
                tool_response_ids = tool_response_ids[len(self.system_prompt) :]
                if len(response_mask) + len(tool_response_ids) >= self.response_length:
                    break
                prompt_ids += tool_response_ids
                response_mask += [0] * len(tool_response_ids)
            else:
                break
            

        breakpoint()
        response_ids = prompt_ids[-len(response_mask) :]
        prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]
        format_reward = np.clip(format_reward, -FORMAT_REARD, FORMAT_REARD)

        metrics["answer_reward"] = answer_reward
        metrics["format_reward"] = format_reward
        metrics["timeout_reward"] = timeout_reward
        reward = answer_reward + format_reward + timeout_reward

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            num_turns=turns, 
            metrics=metrics,
            reward=reward
        )
        return output