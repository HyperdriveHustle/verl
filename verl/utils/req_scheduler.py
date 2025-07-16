import glob
import os
import json
import heapq
import torch
import numpy as np
from verl import DataProto


class ReqScheduler:
    def __init__(self, config):
        self.config = config
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
    
