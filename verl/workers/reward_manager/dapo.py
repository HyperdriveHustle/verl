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

from collections import defaultdict
import logging

import torch
import numpy as np

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

logger = logging.getLogger(__name__)


@register("dapo")
class DAPORewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # ========== DIAGNOSTIC CHECKS: Check if using llm-judge (rm_scores from rollout) ==========
        is_validation = data.meta_info.get("validate", False)
        has_rm_scores = "rm_scores" in data.batch.keys()
        
        # Check if reward_scores exist in non_tensor_batch (indicating llm-judge was used)
        has_reward_scores_in_non_tensor = False 
        reward_scores_samples = []
        if len(data) > 0:
            for i in range(min(3, len(data))):
                item_reward_scores = data[i].non_tensor_batch.get("reward_scores", None)
                if item_reward_scores is not None:
                    has_reward_scores_in_non_tensor = True
                    reward_scores_samples.append((i, item_reward_scores))
        
        reward_scores_info = []
        for i, r in reward_scores_samples:
            if isinstance(r, dict):
                reward_scores_info.append(f"sample_{i}:dict_keys={list(r.keys())}")
            else:
                reward_scores_info.append(f"sample_{i}:type={type(r).__name__},value={str(r)[:50]}")
        
        logger.warning(
            f"[REWARD_DIAGNOSTIC] ===== DAPORewardManager.__call__ =====\n"
            f"  is_validation: {is_validation}\n"
            f"  has_rm_scores: {has_rm_scores}\n"
            f"  has_reward_scores_in_non_tensor: {has_reward_scores_in_non_tensor}\n"
            f"  batch_keys: {list(data.batch.keys())}\n"
            f"  mode: {'VALIDATION' if is_validation else 'TRAINING'}\n"
            f"  using_llm_judge_in_rollout: {has_reward_scores_in_non_tensor}\n"
            f"  reward_scores_samples: {reward_scores_info}"
        )
        
        # ========== POTENTIAL ISSUE: If reward_scores exist but rm_scores don't ==========
        # This means llm-judge computed rewards in rollout, but they weren't converted to rm_scores
        if has_reward_scores_in_non_tensor and not has_rm_scores:
            logger.error(
                f"[REWARD_DIAGNOSTIC] ❌ CRITICAL ISSUE DETECTED: "
                f"reward_scores found in non_tensor_batch but rm_scores NOT in batch!\n"
                f"  This means llm-judge computed rewards in rollout, but they were not converted to rm_scores.\n"
                f"  DAPORewardManager will now recompute rewards using compute_score, which may be inconsistent!\n"
                f"  This could be the root cause of parameter update issues!"
            )

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            rm_scores = data.batch["rm_scores"]
            
            # ========== DIAGNOSTIC CHECK 1: Gradient check ==========
            if rm_scores.requires_grad:
                logger.error(
                    f"[REWARD_DIAGNOSTIC] ❌ CRITICAL: rm_scores requires grad! "
                    f"This could affect parameter updates. Detaching..."
                )
                rm_scores = rm_scores.detach().clone()
            else:
                logger.info(f"[REWARD_DIAGNOSTIC] ✓ rm_scores does not require grad (correct)")
            
            # ========== DIAGNOSTIC CHECK 2: Shape and dtype check ==========
            responses = data.batch["responses"]
            expected_shape = responses.shape
            actual_shape = rm_scores.shape
            
            logger.info(
                f"[REWARD_DIAGNOSTIC] Shape check:\n"
                f"  expected_shape (responses): {expected_shape}\n"
                f"  actual_shape (rm_scores): {actual_shape}\n"
                f"  dtype: {rm_scores.dtype}"
            )
            
            if actual_shape != expected_shape:
                logger.error(
                    f"[REWARD_DIAGNOSTIC] ❌ Shape mismatch! "
                    f"Expected {expected_shape}, got {actual_shape}"
                )
            else:
                logger.info(f"[REWARD_DIAGNOSTIC] ✓ Shape matches (correct)")
            
            # ========== DIAGNOSTIC CHECK 3: Reward placement check ==========
            response_mask = data.batch.get("response_mask", None)
            attention_mask = data.batch.get("attention_mask", None)
            
            if response_mask is not None or attention_mask is not None:
                # Check a few samples to verify reward placement
                num_samples_to_check = min(5, len(data))
                logger.info(f"[REWARD_DIAGNOSTIC] Checking reward placement for {num_samples_to_check} samples...")
                
                for i in range(num_samples_to_check):
                    sample_rm_scores = rm_scores[i]
                    
                    # Find non-zero positions
                    non_zero_positions = (sample_rm_scores != 0).nonzero(as_tuple=False).squeeze(-1)
                    non_zero_values = sample_rm_scores[non_zero_positions] if len(non_zero_positions) > 0 else torch.tensor([])
                    
                    # Find last valid token position using response_mask or attention_mask
                    if response_mask is not None:
                        sample_response_mask = response_mask[i]
                        last_valid_pos = (sample_response_mask != 0).nonzero(as_tuple=False)
                        if len(last_valid_pos) > 0:
                            last_valid_pos = last_valid_pos[-1].item()
                        else:
                            last_valid_pos = None
                    elif attention_mask is not None:
                        # For attention_mask, we need to find the last valid token in response part
                        sample_attention_mask = attention_mask[i]
                        prompt_length = data[i].batch.get("prompts", torch.tensor([])).shape[-1] if hasattr(data[i], 'batch') else 0
                        response_attention = sample_attention_mask[prompt_length:]
                        last_valid_pos = (response_attention != 0).nonzero(as_tuple=False)
                        if len(last_valid_pos) > 0:
                            last_valid_pos = last_valid_pos[-1].item()
                        else:
                            last_valid_pos = None
                    else:
                        last_valid_pos = None
                    
                    # Check if reward is at the correct position
                    if len(non_zero_positions) == 0:
                        logger.warning(
                            f"[REWARD_DIAGNOSTIC] ⚠️ Sample {i}: No non-zero rewards found!"
                        )
                    elif len(non_zero_positions) == 1:
                        reward_pos = non_zero_positions[0].item()
                        reward_val = non_zero_values[0].item()
                        if last_valid_pos is not None:
                            if reward_pos == last_valid_pos:
                                logger.info(
                                    f"[REWARD_DIAGNOSTIC] ✓ Sample {i}: Reward at correct position "
                                    f"(pos={reward_pos}, last_valid={last_valid_pos}, reward={reward_val:.4f})"
                                )
                            else:
                                logger.error(
                                    f"[REWARD_DIAGNOSTIC] ❌ Sample {i}: Reward at WRONG position! "
                                    f"reward_pos={reward_pos}, last_valid_pos={last_valid_pos}, reward={reward_val:.4f}"
                                )
                        else:
                            logger.info(
                                f"[REWARD_DIAGNOSTIC] Sample {i}: Reward at pos={reward_pos}, "
                                f"reward={reward_val:.4f} (cannot verify last_valid_pos)"
                            )
                    else:
                        logger.warning(
                            f"[REWARD_DIAGNOSTIC] ⚠️ Sample {i}: Multiple non-zero rewards found! "
                            f"positions={non_zero_positions.tolist()}, values={non_zero_values.tolist()}"
                        )
            
            # ========== DIAGNOSTIC CHECK 4: Reward value range check ==========
            non_zero_rewards = rm_scores[rm_scores != 0]
            if len(non_zero_rewards) > 0:
                min_reward = non_zero_rewards.min().item()
                max_reward = non_zero_rewards.max().item()
                mean_reward = non_zero_rewards.mean().item()
                std_reward = non_zero_rewards.std().item()
                num_non_zero = len(non_zero_rewards)
                total_tokens = rm_scores.numel()
                
                logger.info(
                    f"[REWARD_DIAGNOSTIC] Reward value statistics:\n"
                    f"  num_non_zero: {num_non_zero}/{total_tokens} ({100*num_non_zero/total_tokens:.2f}%)\n"
                    f"  min: {min_reward:.4f}\n"
                    f"  max: {max_reward:.4f}\n"
                    f"  mean: {mean_reward:.4f}\n"
                    f"  std: {std_reward:.4f}"
                )
                
                # Check if values are reasonable (typically 0-1 for accuracy, but could be other ranges)
                if min_reward < -10 or max_reward > 10:
                    logger.warning(
                        f"[REWARD_DIAGNOSTIC] ⚠️ Reward values seem unusually large: "
                        f"min={min_reward:.4f}, max={max_reward:.4f}"
                    )
            else:
                logger.error(
                    f"[REWARD_DIAGNOSTIC] ❌ CRITICAL: No non-zero rewards found in rm_scores!"
                )
            
            # ========== DIAGNOSTIC CHECK 5: Check if reward_scores exist in non_tensor_batch ==========
            if has_reward_scores_in_non_tensor:
                logger.info(
                    f"[REWARD_DIAGNOSTIC] ✓ reward_scores found in non_tensor_batch "
                    f"(indicating llm-judge was used in rollout)"
                )
                # Sample a few reward_scores
                for i in range(min(3, len(data))):
                    item_reward_scores = data[i].non_tensor_batch.get("reward_scores", {})
                    if item_reward_scores:
                        logger.info(
                            f"[REWARD_DIAGNOSTIC] Sample {i} reward_scores keys: {list(item_reward_scores.keys())}"
                        )
            else:
                logger.warning(
                    f"[REWARD_DIAGNOSTIC] ⚠️ reward_scores NOT found in non_tensor_batch. "
                    f"rm_scores might come from reward model, not llm-judge."
                )
            
            # ========== DIAGNOSTIC CHECK 6: Compare with computed rewards ==========
            # For first sample, compute reward using compute_score and compare
            if len(data) > 0 and not is_validation:
                try:
                    data_item = data[0]
                    prompt_ids = data_item.batch["prompts"]
                    prompt_length = prompt_ids.shape[-1]
                    valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                    valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                    response_ids = data_item.batch["responses"]
                    valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                    valid_response_ids = response_ids[:valid_response_length]
                    
                    prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                    response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                    eos_token = self.tokenizer.eos_token
                    if response_str.endswith(eos_token):
                        response_str = response_str[: -len(eos_token)]
                    
                    ground_truth = data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                    data_source = data_item.non_tensor_batch.get(self.reward_fn_key, None)
                    extra_info = data_item.non_tensor_batch.get("extra_info", {})
                    rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
                    extra_info["rollout_reward_scores"] = rollout_reward_scores
                    
                    if ground_truth is not None and data_source is not None:
                        computed_result = self.compute_score(
                            data_source=data_source,
                            solution_str=response_str,
                            ground_truth=ground_truth,
                            extra_info=extra_info,
                        )
                        computed_score = computed_result["score"] if isinstance(computed_result, dict) else computed_result
                        
                        # Get rm_scores value for this sample
                        sample_rm_scores = rm_scores[0]
                        non_zero_pos = (sample_rm_scores != 0).nonzero(as_tuple=False)
                        if len(non_zero_pos) > 0:
                            rm_score_value = sample_rm_scores[non_zero_pos[0]].item()
                            logger.info(
                                f"[REWARD_DIAGNOSTIC] Comparison for sample 0:\n"
                                f"  rm_scores value: {rm_score_value:.4f}\n"
                                f"  computed_score: {computed_score:.4f}\n"
                                f"  difference: {abs(rm_score_value - computed_score):.4f}\n"
                                f"  match: {abs(rm_score_value - computed_score) < 1e-3}"
                            )
                            if abs(rm_score_value - computed_score) > 1e-3:
                                logger.warning(
                                    f"[REWARD_DIAGNOSTIC] ⚠️ rm_scores value does not match computed_score! "
                                    f"This might indicate a problem with reward placement or calculation."
                                )
                except Exception as e:
                    logger.warning(
                        f"[REWARD_DIAGNOSTIC] Could not compare rewards: {e}"
                    )
            
            logger.warning(f"[REWARD_DIAGNOSTIC] ===== End of diagnostic checks =====\n")
            
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": rm_scores, "reward_extra_info": reward_extra_info}
            else:
                return rm_scores

        # ========== DIAGNOSTIC: Computing rewards (no rm_scores found) ==========
        logger.warning(
            f"[REWARD_DIAGNOSTIC] ===== Computing rewards (no rm_scores in batch) =====\n"
            f"  mode: {'VALIDATION' if is_validation else 'TRAINING'}\n"
            f"  This means we are computing rewards using compute_score function\n"
            f"  (NOT using llm-judge rewards from rollout)"
        )
        
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)

            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, None)

            extra_info = data_item.non_tensor_batch.get("extra_info", {})

            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            
            # ========== DIAGNOSTIC: Check if reward_scores exist ==========
            if rollout_reward_scores and i < 3:  # Log first 3 samples
                logger.info(
                    f"[REWARD_DIAGNOSTIC] Sample {i}: reward_scores found in non_tensor_batch: "
                    f"{list(rollout_reward_scores.keys()) if isinstance(rollout_reward_scores, dict) else 'non-dict'}"
                )
            elif i < 3:
                logger.warning(
                    f"[REWARD_DIAGNOSTIC] Sample {i}: reward_scores NOT found in non_tensor_batch. "
                    f"Using compute_score function instead."
                )

            extra_info["rollout_reward_scores"] = rollout_reward_scores

            if ground_truth is None or data_source is None:
                logger.warning(
                    f"[REWARD_DIAGNOSTIC] Sample {i}: ground_truth={ground_truth}, data_source={data_source}. "
                    f"Skipping reward computation."
                )
                score = 0.0
                result = None
            else:
                result = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
                
                score: float
                if isinstance(result, dict):
                    score = result["score"]
                    # Store the information including original reward
                    for key, value in result.items():
                        reward_extra_info[key].append(value)
                else:
                    score = result
                    reward_extra_info["acc"].append(score)

            reward = score

            if self.overlong_buffer_cfg and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source and data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if data_source and already_print_data_sources.get(data_source, 0) < self.num_examine:
                already_print_data_sources[data_source] = already_print_data_sources.get(data_source, 0) + 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if result is not None:
                    if isinstance(result, dict):
                        for key, value in result.items():
                            print(f"[{key}]", value)
                    else:
                        print("[score]", score)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
