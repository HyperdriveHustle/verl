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


def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == 'openai/gsm8k':
        from . import gsm8k
        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
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
    ] or data_source.startswith("aime"):
        from . import math_dapo
        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source.startswith("math_judge"):
        from . import remote_reward
        # 调用judge model，
        # r"\\boxed\s*{([^}]*)}"匹配response中的pred 在路径/afs/chatrl/users/hwq/code/verl-req-sched/verl/utils/reward_score/remote_reward/__init__.py
        # qwen3默认配置nothinking模式，可在下面文件修改/afs/chatrl/users/hwq/code/verl-req-sched/verl/utils/reward_score/remote_reward/tools/api/base.py
        # extra_body={"chat_template_kwargs": {"enable_thinking": False}}, # nothinking，改成True为thinking
        # 有任何报错找胡文晴
        res = remote_reward.compute_score(data_source, solution_str, ground_truth, extra_info)
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
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
