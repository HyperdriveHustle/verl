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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import sys
import contextlib
import io
import heapq
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy
from time import time
import json
import glob


import numpy as np
import pandas as pd
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    _timer, 
    Role,
    WorkerType, 
    ResourcePoolManager,
    apply_kl_penalty,
    compute_advantage,
)

from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from contextlib import contextmanager
from codetiming import Timer
from typing import Type, Dict
from verl.trainer.ppo import core_algos
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.model import compute_position_id_with_mask
from verl.utils.tracking import ValidationGenerationsLogger
import verl.utils.torch_functional as verl_F


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
    

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """
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
        self.validation_generations_logger = ValidationGenerationsLogger()

        # define in-reward KL control
        # kl loss control currently not suppoorted
        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO,
                AdvantageEstimator.REINFORCE_PLUS_PLUS,
                AdvantageEstimator.REMAX,
                AdvantageEstimator.RLOO,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError
        self.req_scheduler = ReqScheduler(
            config=self.config.req_scheduler,
        )

        self._validate_config()
        self._create_dataloader()

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        # gh512: we use RLHFDatasetFilter to filter the dataset
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
        assert self.train_dataset.truncation == self.config.data.get("truncation", "error"), (
            f"dataset truncation {self.train_dataset.truncation} must be the same as config {self.config.data.get('truncation', 'error')}"
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
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=8,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler)

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

        print(f'[RayDAPOTrainer] in _create_dataloader, Size of train dataloader: {len(self.train_dataloader)}')

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
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            # gh512 disable for debug
            # val_metrics = self._validate()
            # pprint(f'Initial validation metrics: {val_metrics}')
            # logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        timings = [] # gh512 for logging
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
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
                # for time of req schedule

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if 'multi_modal_inputs' in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    gen_batch = new_batch.pop(
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
                new_batch.non_tensor_batch['raw_prompt_ids'] = raw_prompt_ids
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
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    with _timer('post_processing', timing_raw):
                        self.req_scheduler.restore_order(gen_batch_output, 
                                                         reqs_idx,
                                                         self.config.actor_rollout_ref.rollout.n,
                                                        )

                    new_batch.non_tensor_batch['uid'] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    #####################################
                    # gh512: data examine2 (union-ed)
                    seq = new_batch.batch['input_ids']
                    response = new_batch.batch['responses']
                    raw_prompt_ids = new_batch.non_tensor_batch['raw_prompt_ids']
                    print(f'[BATCH OUTPUT]: {seq.shape}, {response.shape} {len(new_batch)} {new_batch.batch.keys()} {new_batch.non_tensor_batch.keys()}')
                    # gh512: log and save
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

                    with _timer('reward', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            reward_tensor = reward_result['reward_tensor']
                            reward_extra_infos_dict = reward_result['reward_extra_info']
                        except Exception as e:
                            print(f'Error in reward_fn: {e}')
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch['token_level_scores'] = reward_tensor

                        print(f'{list(reward_extra_infos_dict.keys())=}')
                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update({
                                k: np.array(v) for k, v in reward_extra_infos_dict.items()
                            })

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(new_batch,
                                                                     kl_ctrl=self.kl_ctrl_in_reward,
                                                                     kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(
                                kl_metrics)  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch['token_level_rewards'] = new_batch.batch['token_level_scores']

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size, we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = new_batch.batch['token_level_rewards'].sum(
                                dim=-1).numpy()
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = new_batch.batch['token_level_scores'].sum(
                                dim=-1).numpy()

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(new_batch.non_tensor_batch['uid'],
                                                   new_batch.non_tensor_batch[metric_name]):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        if self.global_steps < 20:
                            print(f"apply filter_groups by std > 0")
                            kept_prompt_uids = [
                                uid for uid, std in prompt_uid2metric_std.items()
                                if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                            ]
                        else:  # @xiaohuihu 调整 keep prompt 的逻辑，改成 value 均值在 0.2-0.8 之间的留下
                            if self.config.algorithm.filter_groups.filter_score_high is not None \
                                    and self.config.algorithm.filter_groups.filter_score_low is not None:
                                print(f"apply filter_groups: {self.config.algorithm.filter_groups.filter_score_low=}, {self.config.algorithm.filter_groups.filter_score_high=}")
                                kept_prompt_uids = [
                                    uid for uid, metric_val in prompt_uid2metric_vals.items()
                                    if np.mean(metric_val) >= self.config.algorithm.filter_groups.filter_score_low \
                                        and np.mean(metric_val) <= self.config.algorithm.filter_groups.filter_score_high
                                ]
                            else:
                                # 原始逻辑，去掉方差为 0 的
                                print(f"apply filter_groups by std > 0")
                                kept_prompt_uids = [
                                    uid for uid, std in prompt_uid2metric_std.items()
                                    if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                                ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch['uid']):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        if batch is None:
                            batch = new_batch
                        else:
                            batch = DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f'{num_prompt_in_batch=} < {prompt_bsz=}')
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f'{num_gen_batches=}. Keep generating...')
                                continue
                            else:
                                raise ValueError(
                                    f'{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many. Please check your data.'
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

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
                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    # XXX gh512 disable for debug
                    # if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                    #         (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                    #     with _timer('testing', timing_raw):
                    #         val_metrics: dict = self._validate()
                    #         if is_last_step:
                    #             last_val_metrics = val_metrics
                    #     metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or
                                                              self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

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
