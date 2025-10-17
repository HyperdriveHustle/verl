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
# from . import gsm8k, math, prime_math, prime_code

from verl.utils.import_utils import deprecated
import numpy as np

def default_compute_score(data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None, memory_limit_mb=None):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if isinstance(data_source, (list, tuple, np.ndarray)):
        if all(isinstance(ds, str) and (ds.startswith("math_judge") or ds.startswith("expert")) for ds in data_source):
            from . import remote_reward_batch
            return remote_reward_batch.compute_score_batched_from_response(data_source, solution_str, ground_truth, extra_info)
            # return remote_reward_batch.compute_score_batched_from_response(data_source, solution_str, ground_truth, extra_info)
        elif all(isinstance(ds, str) and (ds.startswith("dapo") or ds.startswith("train-math-numinamath")) for ds in data_source):
            from . import math_dapo
            print("use math_dapo")
            # 批量调用 math_dapo
            return [math_dapo.compute_score(s, g) for s, g in zip(solution_str, ground_truth)]
            # from . import remote_reward_batch
            # return remote_reward_batch.compute_score_batched(data_source, solution_str, ground_truth, extra_info)
        elif all(isinstance(ds, str) and (ds.startswith("boxed")) for ds in
                 data_source):
            from . import math_verify_boxed
            # 批量调用 math_dapo
            return [math_verify_boxed.compute_score(s, g) for s, g in zip(solution_str, ground_truth)]
        else:
            from . import remote_reward_batch
            return remote_reward_batch.compute_score_batched(data_source, solution_str, ground_truth, extra_info)

    elif data_source.startswith("math_verify"):
        from . import math_verify
        res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == "openai/gsm8k":
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        from . import math

        res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source in [
            'train-math-numinamath1.5_aops_forum', 'DeepScaleR_no_system', 'dapo_aime2025_s32_no_system', 'dapo_aime2024_s32_no_system',
            'train-math-numinamath1.5_aops_forum_int', 'train-math-numinamath1.5_aops_forum_total',
            'train-math-numinamath1.5_olympiads_int', 'train-math-numinamath1.5_olympiads_total', 
            'gpqa_diamond'
    ] or data_source.startswith("aime") or data_source.startswith("dapo"):
        from . import math_dapo
        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in ['codecontests', 'apps', 'codeforces', 'taco']:
        from . import prime_code
        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ['hiyouga/geometry3k']:
        from . import geo3k
        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in ['/nvfile-heatstorage/chatrl/users/hxh/data/rule_based_rl/math_train/reinforce_step150_wrong_answer/train_sample20_less_than_0d8.jsonl']:
        from . import self_developed
        res = self_developed.compute_score(solution_str, ground_truth)
    elif data_source in ['kk_logic']:
        from . import knight_and_knave
        res = knight_and_knave.compute_score(solution_str, ground_truth)
    elif data_source in ['count_down']:
        from . import count_down
        res = count_down.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (list, tuple)):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None, memory_limit_mb=None):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb)


__all__ = ["default_compute_score"]
