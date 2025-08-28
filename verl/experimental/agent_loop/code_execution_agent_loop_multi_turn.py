
import logging
import os
import re
import numpy as np
from typing import Any
from uuid import uuid4
from time import perf_counter
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

ANSWER_REWARD = 1.0
FORMAT_REARD = 0.1
REWARD_DECAY_FACTOR = 0.5
TIMEOUTDECAY = -0.2
MAX_TURN_LEN = 4096


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
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        cls.system_prompt = tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True)
        
    def validate_response_structure(self, processed_str: str) -> bool:
        pattern = re.compile(r'<think>.*</think>.*<answer>.*</answer>$', re.DOTALL)
        return bool(pattern.match(processed_str.strip()))

    @rollout_trace_op
    async def run(self, messages: list[dict[str, Any]], sampling_params_w_test_code: dict[str, Any]) -> AgentLoopOutput:
        #logger.warning("**************agent run start**************")
        metrics = {}
        request_id = uuid4().hex
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, enable_thinking=True
            ),
        )

        sampling_params = sampling_params_w_test_code["sampling_params"]
        test_code = sampling_params_w_test_code["test_code"]
        response_mask = []
        turns = 0
        assistant_turns = 0

        answer_reward = 0.0
        format_reward = 0.0
        timeout_reward = 0.0 
        reward_decay = 0.0
        format_ok_turns = 0
        is_validate = sampling_params_w_test_code["validate"]
        cur_max_turns = 1 if is_validate else self.max_turns
        while turns < cur_max_turns:
            turns += 1
            with simple_timer("generate_sequence", metrics):
                response_ids = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
                )
            # if len(response_ids) > MAX_TURN_LEN:
            #     breakpoint()
            #     response_ids = response_ids[:MAX_TURN_LEN]
            #     reward_decay -= -0.1
            #     print("max_turn_len_exceed")
            
            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)
            assistant_turns += 1
            solution_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            format_check = self.validate_response_structure(solution_text)
            format_ok_turns += 1 if format_check else 0

            code_pattern =  re.compile(r"```(?:python\n)?(.*?)```", re.DOTALL)
            first_code_match = code_pattern.search(solution_text)
            extracted_code = None
            if first_code_match:
                extracted_code = first_code_match.group(1).strip()
            #     match_start_index, match_end_index = first_code_match.span()
            #     answer_start_pos = solution_text.rfind('<answer>', 0, match_start_index)
            #     if answer_start_pos != -1:
            #         answer_end_pos = solution_text.find('</answer>', match_end_index)
            #         if answer_end_pos != -1:
            #             is_in_answer_tag = True
            error_message = ""
            if extracted_code and hasattr(self, 'code_tool'):
                extracted_code_w_test = extracted_code + "\n" + test_code
                with simple_timer(f"tool_calls", metrics):
                    instance_id = None
                    try:
                        instance_id = await self.code_tool.create()

                        response, score, meta_data = await self.code_tool.execute(instance_id, {"code": extracted_code_w_test})
                        #breakpoint()
                        if meta_data["status"] == "timeout":
                            metrics["timeout"] = 1
                            error_message = f"""
### Instruction
Your previously generated code resulted in a 'timeout'. This usually means the code is too slow or has an infinite loop. Analyze your code for potential performance issues and provide a more efficient, corrected version.

### Previous Code
```python
{extracted_code}
```

### Your Task
1. Potential Root Cause: What is the likely reason for the timeout? (e.g., "The code uses a nested loop leading to O(n^2) complexity which is too slow for the given constraints," or "I forgot a termination condition in the while loop.")

2. Optimization Plan: How will you improve the code's efficiency or fix the loop?

3. Corrected Code: Provide the complete, corrected and optimized Python code inside a markdown block.
"""
                        else:
                            if score.lower() == "success":
                                answer_reward = ANSWER_REWARD * (REWARD_DECAY_FACTOR ** (turns - 1))
                                metrics["success_at_turn"] = turns
                                break
                            else:
                                stdout = meta_data.get("stdout", "")
                                stderr = meta_data.get("stderr", "")
                                c = error_message = f"""
### Instruction
Your previously generated code failed to pass the tests. Analyze the error, formulate a correction plan, and provide the updated code.

### Previous Code
```python
{extracted_code}
```

### Test Output (stdout & stderr)
{stdout}
{stderr}

### Your Task
1. Root Cause Analysis: Briefly explain why the previous code failed based on the test output.

2. Correction Plan: Describe the step-by-step changes you will make to fix the issue.

3. Corrected Code: Provide the complete, corrected Python code inside a markdown block.
"""
                    except Exception as e:
                        breakpoint()
                        logger.error(f"Error during reward calculation: {e}")
                        answer_reward = -0.2
                        break
                    finally:
                        if instance_id:
                            await self.code_tool.release(instance_id)
                if turns >= cur_max_turns:
                    break
                tool_messages = []
                tool_messages.append({
                    "role": "tool",
                    "content": error_message,
                })
                tool_response_ids = await self.loop.run_in_executor(
                    None,
                    lambda messages=tool_messages: self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True
                    ),
                )
                tool_response_ids = tool_response_ids[len(self.system_prompt) :]

                
                if len(response_mask) + len(tool_response_ids) >= self.response_length:
                    break
                prompt_ids += tool_response_ids
                response_mask += [0] * len(tool_response_ids)
            else:
                #logger.warning("**************No code extracted**************")
                metrics["No_code_extracted_count"] = 1
                answer_reward = -0.2
                break
        # if not is_validate:
        #     breakpoint()
        #breakpoint()
        response_ids = prompt_ids[-len(response_mask) :]
        prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]
        format_reward = FORMAT_REARD * (format_ok_turns / turns)
        #timeout_reward = metrics["timeout"] * TIMEOUTDECAY
        metrics["answer_reward"] = answer_reward
        metrics["format_reward"] = format_reward
        metrics["timeout_reward"] = timeout_reward
        reward = answer_reward + format_reward + timeout_reward + reward_decay

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            num_turns=turns, 
            metrics=metrics,
            reward=reward
        )
        return output