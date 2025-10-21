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
import logging
import re
import pandas as pd
import tempfile
import os
from typing import Any
from time import perf_counter
import datasets

from verl.tools.base_tool import OpenAIFunctionToolSchema
from verl.tools.sandbox_fusion_tools import SandboxFusionTool
from verl.utils.dataset import RLHFDataset
from verl.utils.reward_score import math_dapo
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.reward_score.sandbox_fusion.utils import _process_single_case, check_correctness
logger = logging.getLogger(__name__)


class CustomSandboxFusionTool(SandboxFusionTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:

        # NOTE: some script may not explicitly print result, we need to add a print statement to the end of the script
        # lines = code.split("\n")
        # for i, line in reversed(list(enumerate(lines))):
        #     if line == "":
        #         continue
        #     if not lines[i].startswith("print"):
        #         lines[i] = f"print({line})"
        #     break
        # code = "\n".join(lines)

        code = parameters.get("code", "").strip()
        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("language", self.default_language)
        ground_truth = parameters.get("ground_truth", None)
        if not isinstance(code, str):
            code = str(code)
        actual_output, code_status, meta_data = await self.execution_pool.execute.remote(self.execute_code, instance_id, code, timeout, language, ground_truth, self.local_run)
        return actual_output, code_status, meta_data
    
    def execute_code(self, instance_id, code, timeout=30, language="python", ground_truth=None, local_run=False):
        if "functional" in ground_truth:
            code = code + "\n" + ground_truth["functional"]
            result_status, metadata = _process_single_case(
                0, None, None, self.sandbox_fusion_url, code, timeout, self.memory_limit_mb, language, local_run
            )
            if metadata["run_status"] == "Finished":
                actual_output = metadata["stdout"] + metadata["stderr"]
                code_status = metadata["api_status"]
                logger.debug(f"actual_output from sandbox fusion: {actual_output},{instance_id}")
                return actual_output, code_status, metadata
            else:
                return "no stdout here", "Not Finished", metadata
        elif "inputs" in ground_truth and "outputs" in ground_truth:
            result_status, metadata = check_correctness(
                self.sandbox_fusion_url, ground_truth,  code, timeout, self.memory_limit_mb, language, local_run
            )

            total_cases = len(result_status)
            if total_cases == 0:
                return "No test cases found.", "Success", {"status": "Success", "run_status": "Finished", "stdout": "Test cases pass rate:**0.00**\n No test cases found.", "stderr": "", "results": [], }
    
            passed_count = 0
            first_failure_meta = None
            final_code_status = "Success" # Assume success unless we find a failure
            for i, (status, meta) in enumerate(zip(result_status, metadata)):
                if status is True:
                    passed_count += 1
                else:
                    first_failure_meta = meta
                    final_code_status = meta.get("api_status", "Failed") 
                    break 
            
            final_metadata = {}
            if first_failure_meta is None:
                stdout_str = f"Test cases pass rate: **1.00**\nAll {total_cases} test cases passed."
                stderr_str = ""
                final_metadata = {
                    "run_status": "Finished",
                    "api_status": "Success",
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "exit_code": 0,
                    "status": "success"
                }
            else:
                pass_rate = passed_count / total_cases
                
                failed_input = first_failure_meta.get("input")
                actual_output_val = first_failure_meta.get("stdout")
                expected_output_val = first_failure_meta.get("expected_output")
                
                stdout_str = (
                    f"Test cases pass rate: **{pass_rate:.2f}**\n"
                    f"#Failed Test Case:\n"
                    f"  - Input: {failed_input}\n"
                    f"  - Your return value: {repr(actual_output_val)}\n"
                    f"  - Expected answer:  {repr(expected_output_val)}"
                )

                stderr_str = first_failure_meta.get("stderr", "")

                final_metadata = {
                    "run_status": first_failure_meta.get("run_status", "Finished"),
                    "api_status": first_failure_meta.get("api_status", "Failed"),
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "exit_code": first_failure_meta.get("exit_code", 1),
                    "status": first_failure_meta.get("status", "wrong_answer"),
                    "failed_case_index": first_failure_meta.get("case_index")
                }

            final_actual_output = final_metadata["stdout"]
            final_metadata["results"] = result_status
            pass_fail_list = [1 if status is True else 0 for status in result_status]
            final_metadata["pass_fail_list"] = pass_fail_list
            breakpoint()
            if final_metadata["stderr"]:
                final_actual_output += "\n#Error Log:\n" + final_metadata["stderr"]
                
            logger.debug(f"Aggregated actual_output: {final_actual_output}, {instance_id}")
            return final_actual_output, final_code_status, final_metadata
        # we should always expect this since we don't have correct answer



answer_format = """\nThe answer format must be: \\boxed{'The final answer goes here.'}"""

supported_datasets = ["leetcode", "lcbv5", "primeintellect", "taco", "codeforce", "codeforces", "livecodebench"]
class CustomRLHFDataset(RLHFDataset):
    """Custom dataset class to process Maxwell-Jia/AIME_2024, yentinglin/aime_2025 datasets."""
    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            data_source = "/".join(parquet_file.split("/")[-3:])
            try:
                dataframe = datasets.load_dataset(parquet_file)["test"]
            except:
                dataframe = datasets.load_dataset(parquet_file)["train"]
            print(dataframe)
            if any(keyword in data_source for keyword in supported_datasets):
                dataframe = dataframe.map(self.map_fn2, num_proc=16)
            else:
                raise ValueError(f"dataset: '{data_source}' not supported yet")
            
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

    def map_fn2(self, row: dict):
        row["agent_name"] = "code_execution_agent_multi_turn"
        return row

