"""
Preprocess LeetCode problems (newfacade/LeetCodeDataset) to parquet format.
"""

import os
import json
import re
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset, concatenate_datasets, Dataset
from rich.rule import Rule
import rich

from verl.utils.hdfs_io import copy, makedirs

N_TESTSET_PER_DATASET = 512  # per dataset
_EMPTY_RETURN_ = {
    "data_source": None,
    "prompt": None,
    "ability": None,
    "reward_model": None,
    "extra_info": None,
}


def minimize_stdio(inputs, outputs, max_n_tests=8):
    stdin_list = []
    stdout_list = []
    for stdin, stdout in zip(inputs, outputs):
        if isinstance(stdin, list):
            stdin = "\n".join(stdin)
        if isinstance(stdout, list):
            stdout = "\n".join(stdout)
        if sys.getsizeof(stdin) > 4 * 1024:
            continue
        stdout.replace("\r\n", "\n")
        stdin_list.append(stdin)
        stdout_list.append(stdout)

    zipped = sorted(zip(stdin_list, stdout_list), key=lambda x: sys.getsizeof(x[0]))

    if not zipped:
        print("No tests found!")
        return [], []

    sorted_stdin, sorted_stdout = zip(*zipped)
    return list(sorted_stdin[:max_n_tests]), list(sorted_stdout[:max_n_tests])


SYSTEM_PROMPT = """"""

PY_IMPORTS = "import heapq\nfrom math import floor, gcd\nimport random\nimport sys\nfrom typing import *\nfrom functools import *\nimport collections\nfrom collections import *\nfrom itertools import *\nfrom heapq import *\nfrom bisect import *\nfrom string import *\nimport math\nimport datetime\ninf = float('inf')\n"


def leetcode2k(train_path, test_path):
    rich.print(Rule("Loading LeetCodeDataset..."))
    test_dataset = load_dataset("json",
                                data_files=test_path)["train"]
    print("Test set:", test_dataset)

    train_dataset = load_dataset("json",
                                data_files=train_path)["train"]
    print("Before deduplication - Training set:", train_dataset)
    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        meta_instruction = (
            "1.  Think how to solve the task through reasoning and then provides the user with the final answer. The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags, respectively. \n"
            "2.  Your solution will be tested against a series of test cases. The execution will stop upon the test case that fails and report error information. Use this information to refine your code in subsequent attempts. \n"
        )
        def process_fn(example, idx):
            prompt = (
                f"You are a helpful programming assistant. Please solve the programming task below.\n\n"   
                f"{example['query'].strip()}\n\n"
                f"Follow these principle to respond:\n{meta_instruction}\n\n"
            )
            original_test_assertions = example['test'].strip()
            all_lines = original_test_assertions.split('\n')
            assert_lines = [line.strip() for line in all_lines if line.strip().startswith('assert')]
            test_cases_list_str = repr(assert_lines)
            new_check_function_template = f"""
def check(candidate): 
    test_cases = {test_cases_list_str} 
    total_count = len(test_cases) 

    if total_count == 0: 
        # Edge case: No test cases to run. 
        print("Test cases pass rate: **0.0**") 
        return 

    for i, test_case in enumerate(test_cases):  
            try:
                parts = test_case.split('==', 1)
                call_part_str = parts[0].replace('assert', '').strip()
                expected_part_str = parts[1].strip()

                actual_output = eval(call_part_str)
                expected_output = eval(expected_part_str)

                if actual_output == expected_output:
                    continue # 测试通过，继续下一个
                else:
                    pass_rate = i / total_count
                    print(f"Test cases pass rate: **{{pass_rate:.2f}}**")
                    print(f"Failed Test Case: {{test_case}}")
                    print(f"- Your return value: {{repr(actual_output)}}")
                    print(f"- Expected answer:  {{repr(expected_output)}}")
                    return # 立即退出

            except Exception as e:
                pass_rate = i / total_count
                print(f"Test cases pass rate: **{{pass_rate:.2f}}**")
                print(f"Failed Test Case: {{test_case}}")
                print(f"### ERROR: Code failed during execution.")
                error_details = traceback.format_exc()
                print(f"Traceback:\\n{{error_details}}")
                return # 立即退出

    print("Test cases pass rate: **1.0**")
"""
            return {
                "data_source": "code",
                "prompt": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "ability": "coding",
                "reward_model": {
                    "style":
                        "rule",
                    "ground_truth":
                        json.dumps({"functional": f"{new_check_function_template}\n\ncheck({example['entry_point'].strip()})"}),
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "reference": example["completion"],  # C++?
                    "prompt": prompt,
                    "dataset": "LeetCodeDataset",
                },
            }

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    return train_dataset, test_dataset

def process_generic_dataset(
    dataset_name: str, 
    file_path: str, 
    file_format: str = "parquet", # Default to parquet
    test_split_ratio: float = 0.05
    ):

    rich.print(Rule(f"Loading '{dataset_name}' dataset from {file_format.upper()} file..."))
    
    
    dataset = load_dataset(file_format, data_files=file_path)["train"]
    print(f"Full '{dataset_name}' dataset:", dataset)

    def make_map_fn(split):
        meta_instruction = (
            "1.  Think how to solve the task through reasoning and then provides the user with the final answer. The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags, respectively. \n"
            "2.  Your solution will be tested against a series of test cases. The execution will stop upon the test case that fails and report error information. Use this information to refine your code in subsequent attempts. \n"
        )
        def process_fn(example, idx):
            problem_description = example['problem'].strip()
            
            solutions = example.get("solutions")
            solution_code = solutions[0] if solutions else ""
            
            prompt = (
                f"You are a helpful programming assistant. Please solve the programming task below.\n\n"   
                f"{example['problem'].strip()}\n\n"
                f"Follow these principle to respond:\n{meta_instruction}\n\n"
            )
            
            try:
                # The 'tests' field could be a JSON string, which needs parsing.
                # If it's already a dict (common in Parquet), this will still work.
                test_field = example['tests']
                if isinstance(test_field, str):
                    test_data = json.loads(test_field)
                else:
                    test_data = test_field # Assume it's already a dict
                standardized_tests = {}
                if isinstance(test_data, list):
                    #print(f"Example {idx} has 'tests' as a list with {len(test_data)} items.")
                    inputs = []
                    outputs = []
                    for test_case in test_data:
                        # 从每个字典中提取 input 和 output
                        if isinstance(test_case, dict) and 'input' in test_case and 'output' in test_case:
                            inputs.append(test_case['input'])
                            outputs.append(test_case['output'])
                    standardized_tests = {'inputs': inputs, 'outputs': outputs}
                elif isinstance(test_data, dict) and 'inputs' in test_data and 'outputs' in test_data:
                    standardized_tests = test_data
                else:
                    print(f"Skipping example {idx} due to unrecognized 'tests' format: {type(test_data)}")
                    return None
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"Skipping example {idx} due to malformed 'tests' field: {e}")
                return None
            return {
                "data_source": "code",
                "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                "ability": "coding",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps(standardized_tests)
                },
                "extra_info": {"split": split, "index": idx, "reference": solution_code, "prompt": prompt, "dataset": dataset_name},
            }
        return process_fn
    processed_dataset = dataset.map(function=make_map_fn("train"), with_indices=True, writer_batch_size=50)
    processed_dataset = processed_dataset.filter(lambda example: example is not None)
    return processed_dataset



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./data/unified_coding_dataset_v1/")
    args = parser.parse_args()
    local_dir = args.output_dir

    train_datasets = []
    test_datasets = []

    datasets_to_process = [
        {
            "name": "LeetCode", 
            "enabled": False, # Set to False to skip
            "func": leetcode2k,
            "params": {
                "train_path": "/afs/chatrl/users/lyy/data/code_raw/leetcode2k/LeetCodeDataset-v0.3.1-train.jsonl",
                "test_path": "/afs/chatrl/users/lyy/data/code_raw/leetcode2k/LeetCodeDataset-v0.3.1-test.jsonl"
            }
        },
        {
            "name": "APPS", 
            "enabled": True, # Set to False to skip
            "func": process_generic_dataset,
            "params": {
                "dataset_name": "APPS",
                "file_path": "/afs/chatrl/users/lyy/data/code_raw/DeepCoder-Preview-Dataset/primeintellect/train-*.parquet", # <-- IMPORTANT: SET YOUR PATH
                "file_format": "parquet"
            }
        },
    ]

    for d_info in datasets_to_process:
        if not d_info["enabled"]:
            continue
        
        print("\n" + "="*80)
        print(f"Processing dataset: {d_info['name']}")
        print("="*80)

        func = d_info["func"]
        params = d_info.get("params", {})
        #train_data, test_data = func(**params)
        train_data = func(**params)
        if train_data:
            train_datasets.append(train_data)
            #test_datasets.append(test_data)
            print(f"Successfully processed and added '{d_info['name']}'.")
        else:
            print(f"Failed to process '{d_info['name']}', it was skipped.")

    if not train_datasets:
        print("\nNo datasets were processed successfully. Exiting.")
        sys.exit(0)

    final_train_dataset = concatenate_datasets(train_datasets)#.shuffle(seed=42)
    #final_test_dataset = concatenate_datasets(test_datasets)

    rich.print(Rule("Saving the final unified dataset"))
    print("Final Train set:", final_train_dataset)
    #print("Final Test set:", final_test_dataset)

    os.makedirs(local_dir, exist_ok=True)
    rich.print(f"[bold green]Saving to {local_dir}...")
    final_train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    #final_test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    print(f"\nProcessing complete! Unified dataset saved to {local_dir}")

