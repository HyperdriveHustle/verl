
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
REWARD_DECAY_FACTOR = 0.7
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

    def extract_code_from_answer_efficiently(self, solution_text: str) -> str | None:
        answer_tag_start = solution_text.find('<answer>')
        if answer_tag_start == -1:
            return None

        answer_tag_end = solution_text.find('</answer>', answer_tag_start)
        if answer_tag_end == -1:
            return None

        content_start_index = answer_tag_start + len('<answer>')
        answer_content = solution_text[content_start_index:answer_tag_end]
        code_block_start = solution_text.find('```', content_start_index, answer_tag_end)
        code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)
        match = code_pattern.search(answer_content)
        if match:
            return match.group(1).strip()
        else:
            return None
    
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
        response_mask, response_logprobs = [], []
        turns = 0
        assistant_turns = 0

        answer_reward = 0.0
        format_reward = 0.0
        timeout_reward = 0.0 
        reward_decay = 0.0
        format_ok_turns = 0
        progress_reward=0.0
        pre_pass_rate=0
        is_validate = sampling_params_w_test_code["validate"]
        cur_max_turns = 1 if is_validate else self.max_turns
        while turns < cur_max_turns:
            turns += 1
            with simple_timer("generate_sequence", metrics):
                output  = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
                )
            # if len(response_ids) > MAX_TURN_LEN:
            #     breakpoint()
            #     response_ids = response_ids[:MAX_TURN_LEN]
            #     reward_decay -= -0.1
            #     print("max_turn_len_exceed")
            response_ids = output.token_ids
            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)
            if output.log_probs:
                response_logprobs += output.log_probs
            assistant_turns += 1
            solution_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            format_ok_turns += 1 if self.validate_response_structure(solution_text) else 0

            extracted_code=self.extract_code_from_answer_efficiently(solution_text)
            if not extracted_code:
                code_match = re.search(r"```(?:python\n)?(.*?)```", solution_text, re.DOTALL)
                extracted_code = code_match.group(1).strip() if code_match else None
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
Code execution timeout, please reflect your answer and answer again to slove the problem based on the error message and your previous responses.

### Previous Code
```python
{extracted_code}
```
"""
                        else:
                            match_test_pass_rate = re.search(r"Pass rate: \*\*(.*?)\*\*", meta_data["stdout"])
                            pass_rate = float(match_test_pass_rate.group(1)) if match_test_pass_rate else 0.0
                            progress_reward += 0.5 * (pass_rate - pre_pass_rate) #if pass_rate > pre_pass_rate else 0 # it can be negative
                            pre_pass_rate=pass_rate
                            if score.lower() == "success" and pass_rate == 1.0:
                                answer_reward = ANSWER_REWARD * (REWARD_DECAY_FACTOR ** (turns - 1))
                                metrics["success_at_turn"] = turns
                                break
                            else:
                                stdout = meta_data.get("stdout", "")
                                stderr = meta_data.get("stderr", "")
                                c = error_message = f"""
### Instruction
Code test failed.\n\nPlease reflect your answer and asnwer again to slove the problem.

### Previous Code
```python
{extracted_code}
```

### ERROR
{stdout}
{stderr}
"""
                    except Exception as e:
                        #breakpoint()
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
                if response_logprobs:
                    response_logprobs += [0.0] * len(tool_response_ids)
            else:
                metrics["No_code_extracted_count"] = 1
                answer_reward = -0.2
                #breakpoint()
                break


        response_ids = prompt_ids[-len(response_mask) :]
        prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]
        format_reward = FORMAT_REARD * format_ok_turns if format_ok_turns else -0.2

        #timeout_reward = metrics["timeout"] * TIMEOUTDECAY
        metrics["answer_reward"] = answer_reward + progress_reward
        metrics["format_reward"] = format_reward
        metrics["timeout_reward"] = timeout_reward
        reward = answer_reward + format_reward + timeout_reward + reward_decay

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            num_turns=turns, 
            metrics=metrics,
            reward=reward,
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
        )
        return output