""" Preprocess dataset for knights and knaves logic task """

import os
from datasets import Dataset, load_dataset
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json

def make_prefix(dp, template_type):
    quiz = dp['quiz']
    messages = []
    if template_type == 'base':
        content = f"""The user asks a question, and the Assistant solves it.The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. List the identity of each person one by one, for example, <answer> (1) Zoey is a knight\n(2) Oliver is a knight\n(3)... </answer>.\n\nUser:{quiz}\nAssistant: """
        messages.append(
            {"role": "user", "content": content}
        )
    elif template_type == 'qwen-instruct':
        prefix = f"""You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>."""
        messages.append({"role": "system", "content": prefix})
        messages.append({"role": "user", "content": quiz})
    return messages

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/home/t2vg-a100-G4-43/data/kk/instruct/3ppl', help="save dir")
    parser.add_argument('--hdfs_dir', default=None, help="target copy dir")
    parser.add_argument('--train_data_path', default='/home/t2vg-a100-G4-43/mem-kk-logic/kk_train_data/3ppl.jsonl')
    parser.add_argument('--test_data_path', default='/home/t2vg-a100-G4-43/mem-kk-logic/kk_train_data/3ppl.jsonl')
    parser.add_argument('--train_size', type=int, default=900)
    parser.add_argument('--test_size', type=int, default=100)
    parser.add_argument('--template_type', type=str, default='qwen-instruct')
    
    args = parser.parse_args()
    
    data_source = 'kk_logic'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    # Load custom JSONL dataset
    def gen_from_jsonl(path):
        with open(path) as f:
            for line in f:
                yield json.loads(line)
    
    # 加载训练集
    raw_train_dataset = Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': args.train_data_path})
    assert len(raw_train_dataset) >= TRAIN_SIZE
    print(len(raw_train_dataset))
    # 加载测试集
    raw_test_dataset = Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': args.test_data_path})
    print(len(raw_test_dataset))

    train_dataset = raw_train_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_test_dataset.select(range(TEST_SIZE))

    def make_map_fn(split):
        def process_fn(example, idx):
            messages = make_prefix(example, template_type=args.template_type)
            solution = {
                "solution_text_format": example['solution_text_format'],
                "statements": example['statements']
            }
            data = {
                "data_source": data_source,
                "prompt": messages,
                "ability": "kk_logic",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Create local directory if not exists
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    print(f"save parquet to {local_dir}")
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)