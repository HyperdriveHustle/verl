
import logging
import os
import re
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("code_execution_agent")
class CodeExecutionAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level CodeExecutionAgentLoop initialization")

        cls.tokenizer = tokenizer
        
        # 初始化用于奖励计算的工具
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        if not cls.tools:
            logger.warning("CodeExecutionAgentLoop initialized, but no reward tool was found in the config.")
        else:
            cls.reward_tool = next(iter(cls.tools.values()))
            print(f"Initialized reward tool: {cls.reward_tool.name}")

        cls.response_length = config.actor_rollout_ref.rollout.response_length

    @rollout_trace_op
    async def run(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any]) -> AgentLoopOutput:
        metrics = {}
        request_id = uuid4().hex
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            ),
        )

        with simple_timer("generate_sequence", metrics):
            response_ids = await self.server_manager.generate(
                request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params
            )

        solution_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # 这个正则表达式会查找被三个反引号包裹的代码块，并可选地匹配 'python' 标识符
        code_match = re.search(r"```(?:python\n)?(.*?)```", solution_text, re.DOTALL)
        extracted_code = code_match.group(1).strip() if code_match else None
        
        reward = 0.0
        
        if extracted_code and hasattr(self, 'reward_tool'):
            with simple_timer("reward_calculation", metrics):
                instance_id = None
                try:
                    instance_id = await self.reward_tool.create()
                    # 假设奖励工具的 execute 方法接受一个包含 "code" 的字典
                    # 并且返回 (result_string, score, metrics)
                    response, score, _ = await self.reward_tool.execute(instance_id, {"code": extracted_code})
                    reward = score if score is not None else 0.0
                except Exception as e:
                    logger.error(f"Error during reward calculation: {e}")
                    reward = 0.0 # 出错时给予默认奖励
                finally:
                    if instance_id:
                        await self.reward_tool.release(instance_id)

        # 6. 打包最终输出
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            # 因为所有内容都是LLM生成的，所以mask全是1
            response_mask=[1] * len(response_ids[: self.response_length]),
            reward=reward,
            num_turns=2,  # 1个用户轮次, 1个助手轮次
            metrics=metrics,
        )
        return output