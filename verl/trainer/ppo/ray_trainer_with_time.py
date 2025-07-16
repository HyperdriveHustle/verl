#opyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

# tp size for each worker
#MODEL_DEPLOYMENT = [1, 1, 1, 1, 1, 1, 1, 1]
#MODEL_DEPLOYMENT = [2, 1, 1, 1, 1, 1, 1]
#MODEL_DEPLOYMENT = [2,2,2,2]
MODEL_DEPLOYMENT = None

import os
import sys
import contextlib
import io
import uuid
from collections import defaultdict

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy
from time import time
import json
import heapq
import glob


import numpy as np
import pandas as pd
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from verl import DataProto
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
    process_validation_metrics,
)
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn, process_image
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


WorkerType = Type[Worker]


@contextlib.contextmanager
def suppress_stdout():
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


def unpad_responses(padded_tensor, pad_token_id):
    if isinstance(pad_token_id, list):
        # from all worker
        pid = pad_token_id[0]
        for worker_pad_token_id in pad_token_id:
            if worker_pad_token_id != pid:
                raise ValueError("pad_token_id is not the same across all workers")
        pad_token_id = pid

    padded_tensor = padded_tensor.cpu()
    # Convert tensor to list if it's a tensor
    if isinstance(padded_tensor, torch.Tensor):
        padded_list = padded_tensor.tolist()
    else:
        padded_list = padded_tensor
    
    # Reconstruct original responses by removing padding tokens
    unpadded_responses = []
    for padded_response in padded_list:
        # Find where padding starts (first occurrence of pad_token_id)
        try:
            pad_start_idx = padded_response.index(pad_token_id)
            # Get only the tokens before padding
            original_response = padded_response[:pad_start_idx]
        except ValueError:
            # No padding found, use the full response
            original_response = padded_response
        
        unpadded_responses.append(original_response)
    return unpadded_responses

class RLHFDatasetFilter(RLHFDataset):
    def __init__(self,
                parquet_files,
                tokenizer,
                processor,
                prompt_key='prompt',
                image_key='images',
                filter_prompts=True,
                cache_dir='~/.cache/verl/rlhf',
                chat_template_func=None,
                return_raw_chat=False,
                truncation='error',
                #
                filter_overlong_prompts=False, # NOTE: this will filter both prompt and responses
                min_prompt_length=None,
                max_prompt_length=1024,
                min_response_length=None,
                max_response_length=None,
                cap_dataset_size=None,
                req_scheduler=None,
        ):
        self.min_prompt_length = min_prompt_length
        self.max_prompt_length = max_prompt_length
        self.min_response_length = min_response_length
        self.max_response_length = max_response_length
        #
        self.filter_overlong_prompts = filter_overlong_prompts
        self.cap_dataset_size = cap_dataset_size
        self.req_scheduler = req_scheduler

        super().__init__(
            parquet_files=parquet_files,
            tokenizer=tokenizer,
            processor=processor,
            prompt_key=prompt_key,
            image_key=image_key,
            max_prompt_length=max_prompt_length,
            filter_prompts=filter_prompts,
            cache_dir=cache_dir,
            chat_template_func=chat_template_func,
            return_raw_chat=return_raw_chat,
            truncation=truncation,
            filter_overlong_prompts=filter_overlong_prompts,
        )

    def _generate_cache_id(self):
        import hashlib
        """Generate a unique identifier for the current dataset configuration"""
        # [RLHFDatasetFilter]: _generate_cache_id, self.parquet_files=['/afs/chatrl/users/hxh/data/rule_based_rl/DAPO-Math-17k/data/dapo-math-17k_dedup.parquet'].
        print(f'[RLHFDatasetFilter]: _generate_cache_id, {self.parquet_files=}.')          
        # Create a string containing all parameters that would affect filtering
        config_str = str(sorted(self.parquet_files)) + str(self.max_prompt_length) + str(self.min_prompt_length)
        # Hash the configuration string to create a shorter identifier
        return hashlib.md5(config_str.encode()).hexdigest()[:10]

    def _read_files_and_tokenize(self):
        import pickle
        # Create a cache identifier based on input files and filtering parameters
        cache_id = self._generate_cache_id()
        cache_dir = os.path.dirname(self.parquet_files[0])
        cache_file = os.path.join(cache_dir, f"filtered_data_{cache_id}.parquet")
        cache_metadata = os.path.join(cache_dir, f"metadata_{cache_id}.pkl")
        # [RLHFDatasetFilter]: cache_file='/afs/chatrl/users/hxh/data/rule_based_rl/DAPO-Math-17k/data/filtered_data_8aff3605b0.parquet'.
        print(f'[RLHFDatasetFilter]: {cache_file=}.')          
        
        # Check if cached filtered data exists
        if os.path.exists(cache_file) and os.path.exists(cache_metadata):
            try:
                # Load metadata to verify cache is valid
                with open(cache_metadata, 'rb') as f:
                    metadata = pickle.load(f)
                
                # Verify the cache is still valid for our current parameters
                if (metadata['max_prompt_length'] == self.max_prompt_length and
                    metadata['min_prompt_length'] == self.min_prompt_length):
                    print(f"[RLHFDatasetFilter] Loading pre-filtered dataset from cache: {cache_file}")
                    self.dataframe = pd.read_parquet(cache_file)
                    print(f'[RLHFDatasetFilter]: {len(self.dataframe)=} {list(self.dataframe.columns)}')
                    return

            except Exception as e:
                print(f"[RLHFDatasetFilter] Cache loading failed: {e}. Rebuilding cache.")

        # If cache doesn't exist or is invalid, process the data
        # Read and concatenate all input files
        print("[RLHFDatasetFilter] Processing data and building cache...")
        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)

        # XXX, for req scheduling, we need to build a table map from req_id to output len
        # the dataset is too large, we just hacky downsmaple for now
        self.dataframe = pd.concat(dataframes)
        full_len = len(self.dataframe)
        if self.cap_dataset_size is not None:
            self.dataframe = self.dataframe.iloc[:self.cap_dataset_size]
        print(f'[RLHFDatasetFilter]: {full_len=} {len(self.dataframe)=} {list(self.dataframe.columns)}')
        
        # apply filter
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key

            # add a new col for efficient filtering!
            new_key = 'applied_chat_template_prompts'
            self.dataframe[new_key] = self.dataframe[prompt_key].apply(lambda prompt: tokenizer.apply_chat_template(prompt, add_generation_prompt=True,))

            # filter prompt
            t1 = time()
            def filter_long(doc):
                return len(doc[new_key]) <= self.max_prompt_length
            def filter_short(doc):
                return len(doc[new_key]) >= self.min_prompt_length
            if self.min_prompt_length is not None:
                self.dataframe = self.dataframe[self.dataframe.apply(filter_short, axis=1)]
            if self.max_prompt_length is not None:
                self.dataframe = self.dataframe[self.dataframe.apply(filter_long, axis=1)]
            t2 = time()
            print(f'[RLHFDatasetFilter] filter prompt: {len(self.dataframe)=}, {self.min_prompt_length}:{self.max_prompt_length},  time cost: {t2-t1:.2f}s')

            # filter response
            t1 = time()
            def filter_long(doc):
                outlen = self.req_scheduler.lookup_table(doc[new_key])
                if outlen is None:
                    return False
                return outlen <= self.max_response_length
            def filter_short(doc):
                outlen = self.req_scheduler.lookup_table(doc[new_key])
                if outlen is None:
                    return False
                return outlen >= self.min_response_length
            # 
            if self.min_prompt_length is not None:
                self.dataframe = self.dataframe[self.dataframe.apply(filter_short, axis=1)]
            if self.max_prompt_length is not None:
                self.dataframe = self.dataframe[self.dataframe.apply(filter_long, axis=1)]
            t2 = time()
            print(f'[RLHFDatasetFilter] filter response: {len(self.dataframe)=}, {self.min_response_length}:{self.max_response_length},  time cost: {t2-t1:.2f}s')

    def __getitem__(self, item):
        # print(f'[RLHFDatasetFilter] items: {item} {type(item)}, end=')
        row_dict: dict = self.dataframe.iloc[item].to_dict()
        # 原数据格式
        # 输出结果：row_dict keys: dict_keys(['data_source', 'prompt', 'ability', 'reward_model', 'extra_info'])
        # print(f'[RLHFDatasetFilter] row_dict keys: {row_dict.keys()}')

        chat = row_dict.pop(self.prompt_key)
        # self.prompt_key='prompt' or message
        # print(f'[RLHFDatasetFilter] {self.prompt_key=}')

        prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

        assert self.image_key not in row_dict, 'multi-modal is not supported yet'
        raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation,
                                                                        )
        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict


class ReqScheduler:
    def __init__(self, config):
        self.config = config

        # prompt_ids -> len(reponse)
        self.table: dict[tuple[int], int] = self.load_table()
    
    def load_table(self):
        ''' 加载预存的 prompts 信息
        预存的 table 数据格式
        {
            "prompts": [
                [prompt_token_ids_1], 
                [prompt_token_ids_2], 
                ...
            ],
            "lengths": [
                [120, 88, 85, 92, 95, 100, 90, 110],  // prompt 1 对应 sample n 个 response 长度
                [105, 90, 95, 92, 100, 94, 90, 88],   // prompt 2
                ...
            ],
            "stats": [ // 初始预计算存储，仍可保留便于快速调用
                {"max": 120, "min": 85, "mean": 97.5, "std": 10.2, "sum": 780}, 
                {"max": 105, "min": 88, "mean": 94.3, "std": 5.6, "sum": 754}, 
                ...
            ]
        }
        '''
        if self.config.seq_dir is None:
            return {}

        # Find all JSON files in the directory
        json_files = glob.glob(os.path.join(self.config.seq_dir, "*.json"))
        print(f"[ReqScheduler] Found {len(json_files)} JSON files to process")

        # prompts -> list[responses]
        ans = {}
        for json_file in json_files:
            filename = os.path.basename(json_file)
            #if key not in filename:
            #    continue
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # [ReqScheduler] data keys = dict_keys(['prompts', 'response', 'reqs_idx', 'outlens'])
                print(f"[ReqScheduler] data keys = {data.keys()} in {filename}")
                # 按格式保存
                ps = data['prompts']
                ls = data['lengths']
                for p, l in zip(ps, ls):
                    p = tuple(p)
                    if p not in ans:
                        ans[p] = l
                print(f"[ReqScheduler] Processed {filename}, found {len(ans)} unique prompts")
            except Exception as e:
                print(f"[ReqScheduler] Error processing {filename}: {str(e)}")
                raise e
                
        # Aggregate prompts -> responses
        agg = self.config.get('agg', 'mean')
        if agg == 'max':
            ans = {k: max(v) for k, v in ans.items()}
        elif agg == 'min':
            ans = {k: min(v) for k, v in ans.items()}
        elif agg == 'mean':
            ans = {k: int(np.mean(v)) for k, v in ans.items()}
        elif agg =='median':
            ans = {k: int(np.median(v)) for k, v in ans.items()}
        elif agg == 'sum':
            ans = {k: sum(v) for k, v in ans.items()}
        else:
            raise ValueError(f"Unknown agg {agg}")
        print(f'[ReqScheduler] Table-Size: {len(ans)=}')
        return ans

    def lookup_table(self, prompt):
        ''' 根据 table 预存的信息 查找 prompt 的相关信息
        '''
        if isinstance(prompt, list):
            prompt = tuple(prompt)
        assert isinstance(prompt, tuple), f"prompt type {type(prompt)} is not supported"
        if prompt in self.table:
            # print(f"[ReqScheduler] Found prompt {len(prompt)} in table with response length {self.table[prompt]}")
            return self.table[prompt]
        return None

    def update_table(self, raw_prompt_ids, responses):
        new_table = {}
        for p, r in zip(raw_prompt_ids, responses):
            p = tuple(p)
            r = tuple(r)
            if p not in new_table:
                new_table[p] = []
            new_table[p].append(len(r))

        # Aggregate prompts -> responses
        agg = self.config.get('agg', 'mean')
        if agg == 'max':
            new_table = {k: max(v) for k, v in new_table.items()}
        elif agg == 'min':
            new_table = {k: min(v) for k, v in new_table.items()}
        elif agg == 'mean':
            new_table = {k: int(np.mean(v)) for k, v in new_table.items()}
        elif agg =='median':
            new_table = {k: int(np.median(v)) for k, v in new_table.items()}
        elif agg == 'sum':
            new_table = {k: sum(v) for k, v in new_table.items()}
        else:
            raise ValueError(f"Unknown agg {agg}")
        
        # add or overwrite
        for k, v in new_table.items():
            self.table[k] = v
        print(f'[ReqScheduler] in update_table, Table-Size: {len(self.table)=}')

    def log_seqlen(self, raw_prompt_ids, responses, prefix):
        print(f'[ReqScheduler] in log_seqlen, {type(raw_prompt_ids)}, {type(responses)}, {len(raw_prompt_ids)}, {len(responses)}')
        assert len(raw_prompt_ids) == len(responses), f'{len(raw_prompt_ids)}, {len(responses)}'
        prompts_dict = {}
        prompts, response = [], []
        for p, r in zip(raw_prompt_ids, responses):
            if tuple(p) not in prompts_dict:
                prompts_dict[tuple(p)] = []
            prompts_dict[tuple(p)].append(len(r))
        
        for pid in prompts_dict:
            prompts.append(list(pid))
            response.append(prompts_dict[pid])

        log_dir = self.config.log_dir
        os.makedirs(log_dir, exist_ok=True)
        data_files = glob.glob(f"{log_dir}/{prefix}_*.json")
        file_num = len(data_files) + 1
        output_file = f"{log_dir}/{prefix}_{file_num}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'prompts': prompts, 
                'lengths': response
            }, f)
    
    def restore_order(self,
                      gen_batch_output: DataProto,
                      reqs_idx,
                      n_samples,
                    ):
        # the output is permutated by req scheduler
        # this step store the original orders
        # 
        bs = len(gen_batch_output)
        assert bs % n_samples == 0, f'bs {bs} must be divisible by n_samples {n_samples}'
        assert bs//n_samples == len(reqs_idx), f'bs//n_samples {bs//n_samples} != len(reqs_idx) {len(reqs_idx)}'
        print(f"[ReqScheduler] restore_order, {bs=}, {n_samples=}, {len(reqs_idx)=}")

        # e.g. [1, 0] -> [16, 17, ..., 31, 0, 1, ... , 15]
        cnt = 0
        global_idx = [None for _ in range(bs)]
        group_idx = 0
        max_id = max(reqs_idx)
        while group_idx <= max_id:
            for i, idx in enumerate(reqs_idx):
                if idx == group_idx:
                    start_position = i * n_samples
                    end_position = start_position + n_samples
                    global_idx[start_position: end_position] = [j for j in range(cnt, cnt+n_samples)]
                    cnt += n_samples
            group_idx += 1

        assert len(global_idx) == bs, f'len(global_idx) {len(global_idx)} != bs {bs}'

        global_idx = torch.tensor(global_idx)
        gen_batch_output.reorder(global_idx)

    def sched(self, batch_dict: dict,
            world_size: int,
            config,
        ):
        print(f"[ReqScheduler] sched, {world_size=}, {config=}")
        # get OUT len
        outlens = []
        for raw_prompt_ids in batch_dict['raw_prompt_ids']:
            outlen = self.lookup_table(raw_prompt_ids)
            outlens.append(outlen)

        # sched
        tp_size = config.rollout.tensor_model_parallel_size
        assert world_size % tp_size == 0, f'world_size {world_size} must be divisible by tp_size {tp_size}'
        dp_size = world_size // tp_size
        res = self._sched(outlens, dp_size, tp_size)

        # idx -> dp group idx:
        batch_dict['reqs_idx'] = res
        batch_dict['outlens'] = np.array(outlens, dtype=np.int32)
        # len(batch_dict['outlens']) = train_prompt_bs
        # print(f"[ReqScheduler] calculate reqs_idx, outlens = {len(batch_dict['outlens'])}")
        
    
    def print_stats(self, outlens, res):
        longest = max(outlens)
        shortest = min(outlens)
        avg = np.mean(outlens)
        std = np.std(outlens)
        print(f"[ReqScheduler] Stats: {longest=}, {shortest=}, avg: {avg:.2f}, std: {std:.2f}")
        num_group = np.unique(res)
        group = [0 for _ in range(len(num_group))]
        for v in res:
            group[v] += 1
        print(f"[ReqScheduler] Group: {group}")
    
    def _sched(self, outlens, dp_size, tp_size):
        algo = self.config.algo

        # if has None, the prompt is not in table
        # so we use even_prompt
        has_none = False
        for outlen in outlens:
            if outlen is None:
                has_none = True
                break
        
        agg = self.config.get('agg', 'mean')
        if has_none:
            print(f"[ReqScheduler] has None, reset {algo} to even_prompt; {agg=}")
            algo = 'even_prompt'

            # so that print stats will not fail
            for i in range(len(outlens)):
                outlens[i] = -1
        else:
            print(f"[ReqScheduler] algo: {algo}, {agg=}")
        
        # get method
        method = getattr(self, algo)
        res = method(outlens, dp_size, tp_size, self.config)
        self.print_stats(outlens, res)
        return res
    
    def dummy(self, outlens, dp_size, tp_size, config):
        res = [0] * (len(outlens) - 1) + [1]
        res = np.array(res, dtype=np.int32)
        return res

    def even_prompt(self, outlens: list[int], dp_size, tp_size, config):
        per_dp = len(outlens) // dp_size
        res = []
        cnt = 0
        for i in range(0, len(outlens), per_dp):
            for j in range(per_dp):
                res.append(cnt)
            cnt += 1
        return np.array(res, dtype=np.int32)
    
    # def even_token(self, outlens, dp_size, tp_size, config):
    #     total_num_token = sum(outlens)
    #     per_dp = total_num_token // dp_size
    #     res = []
    #     group_idx = 0
    #     cnt = 0
    #     for i in range(0, len(outlens)):
    #         cnt += outlens[i]
    #         if cnt > per_dp:
    #             group_idx += 1
    #             cnt = 0
    #         res.append(group_idx)
    #     return np.array(res, dtype=np.int32)
    
    def even_token(self, outlens, dp_size, tp_size, config):
        prompt_indices = list(range(len(outlens)))
        sorted_pairs = sorted(zip(outlens, prompt_indices), reverse=True)
        heap = [(0, i) for i in range(dp_size)]
        heapq.heapify(heap)
        res = [None] * len(outlens)
        for token_len, orig_idx in sorted_pairs:
            total, group = heapq.heappop(heap)
            res[orig_idx] = group
            heapq.heappush(heap, (total + token_len, group))
        return np.array(res, dtype=np.int32)
    
    def long_short(self, outlens, dp_size, tp_size, config):
        p = np.percentile(outlens, config.percentile)
        long = set()
        for i in range(len(outlens)):
            if outlens[i] > p:
                long.add(i)

        # TODO assume only 1 long workers, the rest is short worker 
        # n_long_worker = dp_size//2
        # n_short_worker = dp_size - n_long_worker
        global MODEL_DEPLOYMENT
        if MODEL_DEPLOYMENT is None:
            n_short_worker = dp_size-1
        else:
            #n_short_worker = sum(MODEL_DEPLOYMENT) - MODEL_DEPLOYMENT[0] + 1
            n_short_worker = len(MODEL_DEPLOYMENT)-1

        # 1. even_prompt for the rest:
        #short_worker_cnt = 1
        #res = []
        #for i in range(len(outlens)):
        #    if i in long:
        #        # only one long worker
        #        res.append(0)
        #    else:
        #        # round-robin the rest prompts
        #        res.append(short_worker_cnt)
        #        short_worker_cnt += 1
        #        if short_worker_cnt > n_short_worker:
        #            short_worker_cnt = 1
        
        # 2. even_token for the rest
        res = [None for _ in range(len(outlens))]
        total_num_token_for_short = 0
        for i in range(len(outlens)):
            if i in long:
                # only one long worker
                res[i] = 0
            else:
                total_num_token_for_short += outlens[i]
                
        per_dp = total_num_token_for_short // n_short_worker + 1
        group_idx = 1
        cnt = 0
        for i in range(len(outlens)):
            if i not in long:
                res[i] = group_idx
                cnt += outlens[i]
                if cnt >= per_dp:
                    group_idx += 1
                    cnt = 0

        print(f"[ReqScheduler] p: {p}, {res=}")
        return np.array(res, dtype=np.int32)
    


############################ ############################
############################ ############################
############################ ############################
############################ ############################
############################ ############################
############################ ############################
############################ ############################


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = 'gae'
    GRPO = 'grpo'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REMAX = 'remax'
    RLOO = 'rloo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str,
                             RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items(
        ):
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes} cannot be satisfied in this ray cluster"
                )


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch["attention_mask"]
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"],
                                kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {
        "actor/reward_kl_penalty": current_kl,
        "actor/reward_kl_penalty_coeff": beta,
    }

    return data, metrics


def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(
                    kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(
                    init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                    target_kl=config.algorithm.kl_ctrl.target_kl,
                    horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO,
                AdvantageEstimator.REINFORCE_PLUS_PLUS,
                AdvantageEstimator.REMAX, AdvantageEstimator.RLOO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError
        
        # gh512 - init Req Scheduler
        self.req_scheduler = ReqScheduler(
            config=self.config.req_scheduler,
        )

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(
                    f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                    f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(
                    f"[{name}] You have set both '{name}.micro_batch_size' AND "
                    f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                    f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated)."
                )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.
                log_prob_micro_batch_size_per_gpu, "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size,
                config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size,
                config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get(
                'ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        print(
            "[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        #self.train_dataset = RLHFDataset(
        #    parquet_files=self.config.data.train_files,
        #    tokenizer=self.tokenizer,
        #    processor=self.processor,
        #    prompt_key=self.config.data.prompt_key,
        #    image_key=self.config.data.get('image_key', 'images'),
        #    max_prompt_length=self.config.data.max_prompt_length,
        #    filter_prompts=True,
        #    return_raw_chat=self.config.data.get('return_raw_chat', False),
        #    truncation='error',
        #    )
        self.train_dataset = RLHFDatasetFilter(
            parquet_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            image_key=self.config.data.get('image_key', 'images'),
            filter_prompts=True,
            return_raw_chat=self.config.data.get('return_raw_chat', False),
            truncation='left',
            # gh512
            filter_overlong_prompts=self.config.data.get('filter_overlong_prompts', False),
            min_prompt_length=self.config.data.min_prompt_length,
            max_prompt_length=self.config.data.max_prompt_length,
            min_response_length=self.config.data.min_response_length,
            max_response_length=self.config.data.max_response_length,
            cap_dataset_size=self.config.data.get('cap_dataset_size', None),
            req_scheduler=self.req_scheduler,
        )

        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(
                self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset,
                                    generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            num_workers=8,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler)

        #self.val_dataset = RLHFDataset(
        #    parquet_files=self.config.data.val_files,
        #    tokenizer=self.tokenizer,
        #    processor=self.processor,
        #    prompt_key=self.config.data.prompt_key,
        #    image_key=self.config.data.get('image_key', 'images'),
        #    max_prompt_length=self.config.data.max_prompt_length,
        #    filter_prompts=True,
        #    return_raw_chat=self.config.data.get('return_raw_chat', False),
        #    truncation='error',
        #    )
        self.val_dataset = RLHFDatasetFilter(
            parquet_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            image_key=self.config.data.get('image_key', 'images'),
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get('return_raw_chat', False),
            truncation=self.config.data.get("truncation", "left"),
            # xiaohui
            filter_overlong_prompts=self.config.data.get('filter_overlong_prompts', False),
            min_prompt_length=0,
            min_response_length=0,
            max_response_length=self.config.data.max_response_length * 10,
            cap_dataset_size=None,
            req_scheduler=self.req_scheduler,
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(
            self.val_dataloader
        ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'[RayPPOTrainer] in _create_dataloader, Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(
            self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'[RayPPOTrainer] in _create_dataloader, Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)
    
    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            # xiaohui: we need to schedule the val requests
            self.req_scheduler.sched(
                test_data, self.actor_rollout_wg.world_size, self.config.actor_rollout_ref,
            )
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                interleave=True,
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[
                    0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            # Store original inputs
            input_ids = test_batch.batch['input_ids']
            input_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in input_ids
            ]
            sample_inputs.extend(input_texts)

            if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=[
                        'raw_prompt_ids', 'multi_modal_data',
                        'multi_modal_inputs'
                    ],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    # 添加 resp 长度相关的信息
                    non_tensor_batch_keys=['raw_prompt_ids', 'reqs_idx', 'outlens'],
                )
            test_reqs_idx = test_gen_batch.non_tensor_batch['reqs_idx']
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            }
            
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")
            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(
                test_gen_batch_padded)
            self.req_scheduler.restore_order(
                test_output_gen_batch_padded, 
                test_reqs_idx, 
                n_samples=self.config.actor_rollout_ref.rollout.val_kwargs.n
            )

            # unpad
            test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch_padded, pad_size=pad_size)

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in output_ids
            ]
            sample_outputs.extend(output_texts)
            test_batch = test_batch.union(test_output_gen_batch)
            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), (f"{key_info}: {len(lst)=}, {len(sample_scores)=}")

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if ((var_name == core_var) and
                            any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and
                        (f"@{n_max}" in metric_name)):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {
            pool: {}
            for pool in self.resource_pool_manager.resource_pool_dict.values()
        }
        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role='actor_rollout',
                model_deployment=MODEL_DEPLOYMENT,
                )
            self.resource_pool_to_cls[resource_pool][
                'actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic],
                config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RewardModel],
                config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            print(f'{resource_pool=} | {class_dict=}')

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(
                class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()
        print("Critic model initialized.")
        print("=" * 100)
        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()
        print("Reference policy initialized.")
        print("=" * 100)
        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()
        print("Reward model initialized.")
        print("=" * 100)
        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()
        print("Actor rollout initialized.")
        print("=" * 100)

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir,
            f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir,
            f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path,
            actor_remote_path,
            self.global_steps,
            #remove_previous_ckpt=self.config.trainer.
            #remove_previous_ckpt_in_save)
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder,
                                             'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir,
                f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.global_steps,
                #remove_previous_ckpt=self.config.trainer.
                #remove_previous_ckpt_in_save)
            )

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder,
                                             'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir,
            'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir,
                                                 checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(
                checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path
                    and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode,
                                  str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir,
                                                      global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path,
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.
                                           trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(
                f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch"
            )

    def _balance_batch(self,
                       batch: DataProto,
                       metrics,
                       logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(
            batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor(
            [j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst,
            partitions=global_partition_lst,
            prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config,
                                                        resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get(
                'val_before_train', True):
            # XXX gh512 disable for now
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timings = []

        for epoch in range(self.config.trainer.total_epochs):
            print('*'*100)
            print('='*100)
            print(f"Epoch {epoch}: ")

            for bs_idx, batch_dict in enumerate(self.train_dataloader):
                # gh512; add sched results
                self.req_scheduler.sched(batch_dict,
                                        self.actor_rollout_wg.world_size,
                                        self.config.actor_rollout_ref,
                                    )

                metrics = {}
                timing_raw = {}
                # print(f'[BATCH] {len(batch_dict)} {batch_dict.keys()} {len(batch_dict["input_ids"])}')
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=[
                            'input_ids', 'attention_mask', 'position_ids'
                        ],
                        non_tensor_batch_keys=[
                            'raw_prompt_ids', 'multi_modal_data',
                            'multi_modal_inputs'
                        ],
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        # 添加 resp 长度相关的信息
                        non_tensor_batch_keys=['raw_prompt_ids', 'reqs_idx', 'outlens'],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                # gh512: data examine
                idx = gen_batch.batch['input_ids']  # (bs, prompt_length)
                attention_mask = gen_batch.batch['attention_mask']
                position_ids = gen_batch.batch['position_ids']
                raw_prompt_ids = gen_batch.non_tensor_batch['raw_prompt_ids'] # (bs, varlen)

                # NOTE: we put raw_prompt_ids back to batch for repeated-interleave purpose and log seq len
                # raw_prompt_ids 存储的是 prompts 的原始 token ids
                batch.non_tensor_batch['raw_prompt_ids'] = raw_prompt_ids
                reqs_idx = gen_batch.non_tensor_batch['reqs_idx']
                outlens = gen_batch.non_tensor_batch['outlens']
                # print(f'[BATCH INPUT]: reqs_idx = {reqs_idx[0]}, outlens = {len(outlens)}')
                print(
                    f'[BATCH INPUT]: {idx.shape}, {attention_mask.shape}, {position_ids.shape}, {gen_batch.non_tensor_batch.keys()} {type(raw_prompt_ids)}'
                )

                with _timer('step', timing_raw):
                    # generate a batch
                    # 这里传入的 batch 是所有的数据，到具体 rank 上再做分配
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(
                            gen_batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    with _timer('post_processing', timing_raw):
                        self.req_scheduler.restore_order(gen_batch_output, 
                                                         reqs_idx,
                                                         self.config.actor_rollout_ref.rollout.n,
                                                        )
                        batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                            dtype=object,
                        )
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                        #####################################
                        # gh512: data examine2 (union-ed)
                        seq = batch.batch['input_ids']
                        response = batch.batch['responses']
                        raw_prompt_ids = batch.non_tensor_batch['raw_prompt_ids']
                        print(f'[BATCH OUTPUT]: {seq.shape}, {response.shape} {len(batch)} {batch.batch.keys()} {batch.non_tensor_batch.keys()}')
                        # gh512: log
                        pad_ids = self.actor_rollout_wg.get_tokenizer_pad_id()
                        model = self.config.actor_rollout_ref.model.path.split('/')[-1]
                        dataset = self.config.data.train_files[0].split('/')[-1]
                        prefix = f'{dataset}_{model}_E{epoch}B{bs_idx}_data'
                        unpadded = unpad_responses(response, pad_ids)
                        self.req_scheduler.log_seqlen(
                            raw_prompt_ids, 
                            unpadded,
                            prefix, 
                        )
                        self.req_scheduler.update_table(
                            raw_prompt_ids,
                            unpadded,
                        )
                        # gh512: data examine2 (union-ed)
                        #####################################

                        batch.batch["response_mask"] = compute_response_mask(batch)
                        # balance the number of valid tokens on each dp rank.
                        # Note that this breaks the order of data inside the batch.
                        # Please take care when you implement group based adv computation such as GRPO and rloo
                        if self.config.trainer.balance_batch:
                            self._balance_batch(batch, metrics=metrics)

                        # compute global_valid tokens
                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()


                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        with suppress_stdout():
                            reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch[
                                'token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(
                            critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)
                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(
                                batch)
                        actor_output_metrics = reduce_metrics(
                            actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    # XXX gh512 disable
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or
                                                              self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                with _timer('collecting', timing_raw):
                    # collect metrics
                    metrics.update(
                        compute_data_metrics(batch=batch,
                                             use_critic=self.use_critic))
                    metrics.update(
                        compute_timing_metrics(batch=batch,
                                               timing_raw=timing_raw))
                    # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing


                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    progress_bar.close()
                    return
                progress_bar.update(1)
                self.global_steps += 1
            # gh512
            # print _timer
            print(f'{epoch=}: {bs_idx=}')
            print(timing_raw)
            print('*' * 100)
            timings.append(timing_raw)


        # print time
        keys = timings[0].keys()
        stats = {key: [] for key in keys}
        for timing in timings:
            for key in keys:
                stats[key].append(timing[key])

        print(f'timing: {len(timings)}')
        for key in keys:
            print(f'{key}: ')
            print(f'{np.mean(stats[key])} - {np.std(stats[key])}')
