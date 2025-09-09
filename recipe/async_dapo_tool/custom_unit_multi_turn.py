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
from typing import Any
from time import perf_counter
import datasets

from verl.tools.base_tool import OpenAIFunctionToolSchema
from verl.tools.sandbox_fusion_tools import SandboxFusionTool
from verl.utils.dataset import RLHFDataset
from verl.utils.reward_score import math_dapo
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)


class CustomSandboxFusionTool(SandboxFusionTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        code = parameters["code"].strip()

        # NOTE: some script may not explicitly print result, we need to add a print statement to the end of the script
        # lines = code.split("\n")
        # for i, line in reversed(list(enumerate(lines))):
        #     if line == "":
        #         continue
        #     if not lines[i].startswith("print"):
        #         lines[i] = f"print({line})"
        #     break
        # code = "\n".join(lines)

        timeout = parameters.get("timeout", self.default_timeout)
        language = parameters.get("language", self.default_language)
        if not isinstance(code, str):
            code = str(code)
        start = perf_counter()
        result = await self.execution_pool.execute.remote(self.execute_code, instance_id, code, timeout, language)
        end = perf_counter()
        breakpoint()
        actual_output, code_status, meta_data = result
        return actual_output, code_status, meta_data



answer_format = """\nThe answer format must be: \\boxed{'The final answer goes here.'}"""


class CustomRLHFDataset(RLHFDataset):
    """Custom dataset class to process Maxwell-Jia/AIME_2024, yentinglin/aime_2025 datasets."""

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            data_source = "/".join(parquet_file.split("/")[-2:])
            if "train" in data_source:
                dataframe = datasets.load_dataset(parquet_file)["train"]
            elif "test" in data_source:
                dataframe = datasets.load_dataset(parquet_file)["test"]
            if "leetcode" in data_source:
                dataframe = dataframe.map(self.map_fn2, num_proc=16)
            else:
                pass # other datasets are not supported yet
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

    def map_fn2(self, row: dict):
        row["agent_name"] = "code_execution_agent_multi_turn"
        return row

