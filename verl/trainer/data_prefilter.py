# create a class DataProfiler
# each data is identified by a unique string id
# each data has a reward trace which is a list of tuples (epoch, reward)
# the class should have a method to add reward tuple. If the data is not in the list, it should be added
# the class should have a method to get the reward trace for a given data id
import torch
import numpy as np
import pandas as pd
import random
from collections import defaultdict

class DataPrefilter:
    def __init__(
        self, 
        reference_epoch,
        easy_acc_thresh,
        hard_acc_thresh,
        hard_keep_ratio,
        easy_keep_ratio,
        empty_keep_ratio,
        keep_empty_all,
        agg,
        strategy_version,
        init_history_acc_trace=None
    ):
        self.history_acc_trace = defaultdict(list) # (epoch, correct_num)
        self.easy_acc_thresh = easy_acc_thresh if easy_acc_thresh is not None else 1.0
        self.hard_acc_thresh = hard_acc_thresh if hard_acc_thresh is not None else 0.0
        self.hard_keep_ratio = hard_keep_ratio if hard_keep_ratio is not None  else 1.0
        self.easy_keep_ratio = easy_keep_ratio if easy_keep_ratio is not None else 1.0
        self.empty_keep_ratio = empty_keep_ratio if empty_keep_ratio is not None else 1.0
        self.agg = agg if agg else "none"
        self.reference_epoch = int(10e7) if reference_epoch is None else reference_epoch
        self.idx_adj = 0
        if strategy_version is None or strategy_version == "v0":
            self.filter_fn = self.simple_strategy_check
        elif strategy_version == "v1":
            self.filter_fn = self.simple_strategy_check_v1
        self.keep_empty_all = keep_empty_all
        if init_history_acc_trace is not None:
            self.history_acc_trace = init_history_acc_trace
            self.idx_adj = -1
        print("[dynamic filter] DataPrefilter initialized parameters:")
        print(f"[dynamic filter] reference_epoch={self.reference_epoch}")
        print(f"[dynamic filter] easy_acc_thresh={self.easy_acc_thresh}")
        print(f"[dynamic filter] hard_acc_thresh={self.hard_acc_thresh}")
        print(f"[dynamic filter] easy_keep_ratio={self.easy_keep_ratio}")
        print(f"[dynamic filter] hard_keep_ratio={self.hard_keep_ratio}")
        print(f"[dynamic filter] empty_keep_ratio={self.empty_keep_ratio}")
        print(f"[dynamic filter] agg_method={self.agg}")
        print(f"[dynamic filter] keep_empty_all={self.keep_empty_all}")
        
        self.easy_example_cnt = 0
        self.hard_example_cnt = 0
        self.empty_example_cnt = 0
        self.easy_keep_cnt = 0
        self.hard_keep_cnt = 0
        self.empty_keep_cnt = 0
        self.epoch_skip_cnt = 0
        self.epoch_easy_skip_cnt = 0
        self.epoch_hard_skip_cnt = 0
        self.epoch_empty_skip_cnt = 0
        self.both_easy_hard_example_cnt = 0

    def epoch_init(self, current_epoch):
        print("------Epoch stat-------")
        print(f"[dynamic filter] empty_example_cnt: {self.empty_example_cnt}")
        print(f"[dynamic filter] easy_example_cnt: {self.easy_example_cnt}")
        print(f"[dynamic filter] hard_example_cnt: {self.hard_example_cnt}")
        print(f"[dynamic filter] easy_keep_cnt: {self.easy_keep_cnt}")
        print(f"[dynamic filter] hard_keep_cnt: {self.hard_keep_cnt}")
        print(f"[dynamic filter] empty_keep_cnt: {self.empty_keep_cnt}")
        print(f"[dynamic filter] epoch_skip_cnt: {self.epoch_skip_cnt}")
        print(f"[dynamic filter] epoch_easy_skip_cnt: {self.epoch_easy_skip_cnt}")
        print(f"[dynamic filter] epoch_hard_skip_cnt: {self.epoch_hard_skip_cnt}")
        print(f"[dynamic filter] epoch_empty_skip_cnt: {self.epoch_empty_skip_cnt}")
        print("------Epoch stat end-------")
        if self.keep_empty_all and current_epoch == 1:
            self.keep_empty_all = False
            print("[dynamic filter] keep_empty_all = False")
        self.start_epoch = max(current_epoch-self.reference_epoch, self.idx_adj)
        print(f"Epoch {current_epoch} init, start_epoch: {self.start_epoch}")
        self.empty_example_cnt, self.easy_example_cnt, self.hard_example_cnt = 0, 0, 0
        self.empty_keep_cnt, self.easy_keep_cnt, self.hard_keep_cnt = 0, 0, 0
        self.both_easy_hard_example_cnt = 0
        self.epoch_skip_cnt, self.epoch_easy_skip_cnt, self.epoch_hard_skip_cnt, self.epoch_empty_skip_cnt = 0, 0, 0, 0
        
    def _get_agg_acc(self, history_acc_trace, fill_empty=False):
        all_acc = []
        for item in history_acc_trace[::-1]:
            if item[0] < self.start_epoch:
                if fill_empty:
                    all_acc.append(item[1])
                break
            all_acc.append(item[1])
        if len(all_acc):
            if self.agg == 'max':
                return max(all_acc)
            elif self.agg == 'min':
                return min(all_acc)
            elif self.agg == "mean":
                return sum(all_acc) / len(all_acc)
        else:
            return None

    def simple_strategy_check(self, current_epoch, data_id):
        history_acc_trace = self.history_acc_trace.get(data_id, [])
        is_empty, is_easy, is_hard = len(history_acc_trace) == 0, False, False
        is_skip = True
        
        if not is_empty:
            is_empty = True
            if self.agg == 'none':
                for item in history_acc_trace[::-1]:
                    if item[0] < self.start_epoch:
                        break
                    is_empty = False
                    if item[1] > self.easy_acc_thresh:
                        is_easy = True
                        break
                    if item[1] < self.hard_acc_thresh:
                        is_hard = True
            else:
                _acc = self._get_agg_acc(history_acc_trace, fill_empty=False)
                if _acc is None:
                    is_empty = True
                else:
                    is_empty = False
                    if _acc > self.easy_acc_thresh:
                        is_easy = True
                    if _acc < self.hard_acc_thresh:
                        is_hard = True
        if is_empty and self.keep_empty_all:
            is_skip = False
        else:
            is_skip = is_empty or is_easy or is_hard
            self.empty_example_cnt += 1 if is_empty else 0
            self.easy_example_cnt += 1 if is_easy else 0
            self.hard_example_cnt += 1 if is_hard else 0
            self.both_easy_hard_example_cnt = 1 if is_easy and is_hard else 0
            if is_skip:
                if is_empty and random.random() <= self.empty_keep_ratio:
                    self.empty_keep_cnt += 1
                    is_skip = False
                if is_easy and random.random() <= self.easy_keep_ratio:
                    self.easy_keep_cnt += 1
                    is_skip = False
                if is_hard and random.random() <= self.hard_keep_ratio:
                    self.hard_keep_cnt += 1
                    is_skip = False
        return is_empty, is_easy, is_hard, is_skip
    
    def simple_strategy_check_v1(self, current_epoch, data_id):
        history_acc_trace = self.history_acc_trace.get(data_id, [])
        is_empty, is_easy, is_hard = len(history_acc_trace) == 0, False, False
        is_skip = True
        if not is_empty:
            if self.agg == 'none':
                for item in history_acc_trace[::-1]:
                    if item[0] < self.start_epoch:
                        break
                    is_empty = False
                    if item[1] > self.easy_acc_thresh:
                        is_easy = True
                        break
                    if item[1] < self.hard_acc_thresh:
                        is_hard = True
            else:
                _acc = self._get_agg_acc(history_acc_trace, fill_empty=True)
                if _acc is None:
                    is_empty = True
                else:
                    is_empty = False
                    if _acc > self.easy_acc_thresh:
                        is_easy = True
                    if _acc < self.hard_acc_thresh:
                        is_hard = True
        if is_empty and self.keep_empty_all:
            is_skip = False
        else:
            is_skip = is_empty or is_easy or is_hard
            self.empty_example_cnt += 1 if is_empty else 0
            self.easy_example_cnt += 1 if is_easy else 0
            self.hard_example_cnt += 1 if is_hard else 0
            self.both_easy_hard_example_cnt = 1 if is_easy and is_hard else 0
            if is_skip:
                if is_empty and random.random() <= self.empty_keep_ratio:
                    self.empty_keep_cnt += 1
                    is_skip = False
                if is_easy and random.random() <= self.easy_keep_ratio:
                    self.easy_keep_cnt += 1
                    is_skip = False
                if is_hard and random.random() <= self.hard_keep_ratio:
                    self.hard_keep_cnt += 1
                    is_skip = False
        return is_empty, is_easy, is_hard, is_skip

    def filter_examples_easy(self, current_epoch, batch_dict, return_log=False):
        selected_index, skiped_data_index = [], []
        df = pd.DataFrame(batch.non_tensor_batch)
        df["data_id"] = df['data_source'] + "_" + df['index'].astype(str)
        empty_example_cnt, easy_example_cnt, hard_example_cnt, skip_example_cnt = 0, 0, 0, 0
        easy_skip_example_cnt, hard_skip_example_cnt, empty_skip_example_cnt = 0, 0, 0
        both_easy_hard_example_cnt = 0
        for index, data_id in enumerate(df['data_id'].unique()):
            is_empty, is_easy, is_hard, is_skip = self.filter_fn(current_epoch, data_id)
            empty_example_cnt += 1 if is_empty else 0
            easy_example_cnt += 1 if is_easy else 0
            hard_example_cnt += 1 if is_hard else 0
            easy_skip_example_cnt += 1 if is_easy and is_skip else 0
            hard_skip_example_cnt += 1 if is_hard and is_skip else 0
            empty_skip_example_cnt += 1 if is_empty and is_skip else 0
            skip_example_cnt += 1 if is_skip else 0
            
            both_easy_hard_example_cnt = 1 if is_easy and is_hard else 0
            if is_skip:
                skiped_data_index.append(data_id.rsplit("_", 1)[-1])
                continue
            else:
                selected_index.append(index)
        
        self.epoch_skip_cnt += skip_example_cnt
        self.epoch_easy_skip_cnt += easy_skip_example_cnt
        self.epoch_hard_skip_cnt += hard_skip_example_cnt
        self.epoch_empty_skip_cnt += empty_skip_example_cnt
        
        batch = batch.select_by_index(selected_index)
        print(f"[dynamic filter] batch empty examples count: {empty_example_cnt}")
        print(f"[dynamic filter] batch easy examples count: {easy_example_cnt}")
        print(f"[dynamic filter] batch hard examples count: {hard_example_cnt}")
        print(f"[dynamic filter] batch easy skip examples count: {easy_skip_example_cnt}")
        print(f"[dynamic filter] batch hard skip examples count: {hard_skip_example_cnt}")
        print(f"[dynamic filter] batch empty skip examples count: {empty_skip_example_cnt}")
        print(f"[dynamic filter] batch easy-hard examples count: {both_easy_hard_example_cnt}")
        print(f"[dynamic filter] batch skip examples count: {skip_example_cnt}")
        print(f"[dynamic filter] epoch skip examples count: {self.epoch_skip_cnt}")
        print(f"[dynamic filter] epoch skip easy examples count: {self.epoch_easy_skip_cnt}")
        print(f"[dynamic filter] epoch skip hard examples count: {self.epoch_hard_skip_cnt}")
        print(f"[dynamic filter] epoch skip empty examples count: {self.epoch_empty_skip_cnt}")
        
        if return_log:
            return batch, skiped_data_index
        return batch

    def filter_examples_easy_dict(self, current_epoch, batch_dict, return_log=False):
        keys = list(batch_dict.keys())
        data_len = len(batch_dict[keys[0]])
        
        empty_example_cnt, easy_example_cnt, hard_example_cnt, skip_example_cnt = 0, 0, 0, 0
        easy_skip_example_cnt, hard_skip_example_cnt, empty_skip_example_cnt = 0, 0, 0
        both_easy_hard_example_cnt = 0
        keep_indices, skiped_data_index = [], []

        for i in range(data_len):
            data_source = batch_dict['data_source'][i]
            index = batch_dict['index'][i]
            data_id = f"{data_source}_{index}"
            is_empty, is_easy, is_hard, is_skip = self.filter_fn(current_epoch, data_id)
            empty_example_cnt += 1 if is_empty else 0
            easy_example_cnt += 1 if is_easy else 0
            hard_example_cnt += 1 if is_hard else 0
            easy_skip_example_cnt += 1 if is_easy and is_skip else 0
            hard_skip_example_cnt += 1 if is_hard and is_skip else 0
            empty_skip_example_cnt += 1 if is_empty and is_skip else 0
            skip_example_cnt += 1 if is_skip else 0
            
            both_easy_hard_example_cnt = 1 if is_easy and is_hard else 0
            if is_skip:
                skiped_data_index.append(data_id.rsplit("_", 1)[-1])
                continue
            else:
                keep_indices.append(i)
            # if self.check_is_filter(index, data_source):
            #     keep_indices.append(i)

        self.epoch_skip_cnt += skip_example_cnt
        self.epoch_easy_skip_cnt += easy_skip_example_cnt
        self.epoch_hard_skip_cnt += hard_skip_example_cnt
        self.epoch_empty_skip_cnt += empty_skip_example_cnt
        print(f"[dynamic filter] batch empty examples count: {empty_example_cnt}")
        print(f"[dynamic filter] batch easy examples count: {easy_example_cnt}")
        print(f"[dynamic filter] batch hard examples count: {hard_example_cnt}")
        print(f"[dynamic filter] batch easy skip examples count: {easy_skip_example_cnt}")
        print(f"[dynamic filter] batch hard skip examples count: {hard_skip_example_cnt}")
        print(f"[dynamic filter] batch empty skip examples count: {empty_skip_example_cnt}")
        print(f"[dynamic filter] batch easy-hard examples count: {both_easy_hard_example_cnt}")
        print(f"[dynamic filter] batch skip examples count: {skip_example_cnt}")
        print(f"[dynamic filter] epoch skip examples count: {self.epoch_skip_cnt}")
        print(f"[dynamic filter] epoch skip easy examples count: {self.epoch_easy_skip_cnt}")
        print(f"[dynamic filter] epoch skip hard examples count: {self.epoch_hard_skip_cnt}")
        print(f"[dynamic filter] epoch skip empty examples count: {self.epoch_empty_skip_cnt}")

        filtered_batch = {}
        for key, value in batch_dict.items():
            filtered_batch[key] = value[keep_indices]
            # if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
            #     filtered_batch[key] = value[keep_indices]
            # else:
            #     filtered_batch[key] = [value[i] for i in keep_indices]
        if return_log:
            return filtered_batch, skiped_data_index
        return filtered_batch

    # def add_reward(self, epoch: int, data_id: str, reward: float):
    #     self.history_acc_trace[data_id].append((epoch, reward))
        
    def update(self, epoch, acc_info: tuple):
        for data_id, acc in acc_info:
            try:
                if self.history_acc_trace.get(data_id, [])[-1][0] == epoch:
                    print(f"{epoch} {data_id} dup, skip")
                    continue
            except:
                pass
            self.history_acc_trace[data_id].append((epoch, acc))
            # if acc >= self.easy_acc_thresh:
            #     self.easy_example_cnt += 1
            # if acc <= self.hard_acc_thresh:
            #     self.hard_example_cnt += 1

    def get_reward_trace(self, data_id: str):
        return self.history_acc_trace.get(data_id, [])

    def get_all_data_ids(self):
        return list(self.history_acc_trace.keys())

    def save(self, path: str):
        print(f"save history_acc_trace to {path}")
        torch.save(self.history_acc_trace, path)
        print(f"Saved history_acc_trace with {len(self.history_acc_trace)} data ids.")

    def load(self, path: str, init_mode: str):
        print(f"Loading history_acc_trace from {path}")
        if init_mode == "directly":
            self.history_acc_trace = torch.load(path, weights_only=False)
            print(f"Loaded history_acc_trace with {len(self.history_acc_trace)} data ids.")
        elif init_mode == "only_last":
            tmp_history_acc_trace = torch.load(path, weights_only=False)
            for k, v in tmp_history_acc_trace.items():
                self.history_acc_trace[k] = [(-1, v[-1][1])]
            print(f"Loaded history_acc_trace with {len(self.history_acc_trace)} data ids.")

    def __len__(self):
        return len(self.history_acc_trace)

# 示例用法：
# acc_batch_dict = {}
# while True:
#     batch_dict = get_next_batch()  # 自定义获取函数
#     final_batch, acc_batch_dict = filter_batch(batch_dict, acc_batch_dict, batch_size=64)
#     if final_batch is not None:
#         process_batch(final_batch)  # 自定义处理逻辑