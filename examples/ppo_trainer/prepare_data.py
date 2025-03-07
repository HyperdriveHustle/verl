import os
import json
from verl.utils.hdfs_io import copy, makedirs
from datasets import load_dataset
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/nvfile-heatstorage/teleai-infra/wxe/verl/data/math')
    # parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = '/nvfile-heatstorage/chatrl/users/hxh/data/rule_based_rl/math_train/reinforce_step150_wrong_answer/train_sample20_less_than_0d8.jsonl'
    
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    train_dataset = load_dataset("json", data_files=data_source, split="train")
    def make_map_fn(split):

        def process_fn(example, idx):
            id=example.pop('id')
            question=example.pop('input')
            question = question + ' ' + instruction_following

            answer=example.pop('ground_truth_answer')
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': id
                }
            }
            return data

        return process_fn
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)

    local_dir = args.local_dir
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    # with open(data_source, "r", encoding="utf-8") as f:
    #     try:
    #         for line in f:
    #             data = json.loads(line.strip())  # 解析 JSON 行
    #             if isinstance(data, list):
    #                 # with open(output_file, "w", encoding="utf-8") as out_f:
    #                 #     for item in data:
    #                 #         out_f.write(json.dumps(item, ensure_ascii=False) + "\n")  # 逐行写入 JSONL
    #                 print(f"文件已修复，保存为")
    #             else:
    #                 id=data.pop('id')
    #                 question=data.pop('input')
    #                 question = question + ' ' + instruction_following

    #                 answer=data.pop('ground_truth_answer')
    #                 line = {
    #                         "data_source": data_source,
    #                         "prompt": [{
    #                             "role": "user",
    #                             "content": question
    #                         }],
    #                         "ability": "math",
    #                         "reward_model": {
    #                             "style": "rule",
    #                             "ground_truth": answer
    #                         },
    #                         "extra_info": {
    #                             'split': 'train',
    #                             'index': id
    #                         }
    #                     }
    #     except json.JSONDecodeError as e:
    #         print(f"JSON 解析错误: {e}")



    #     # 或者逐行读取处理：
    #     # for line in f:
    #     #     item = json.loads(line)
    #     #     print(item)  # 逐行处理

    # # print(data)  # data 是一个列表，每个元素是一个 JSON 对象

    

