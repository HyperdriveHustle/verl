"""
Preprocess LeetCode problems (newfacade/LeetCodeDataset) to parquet format.
"""

import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset, concatenate_datasets
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


SYSTEM_PROMPT = """You are an expert Python programming assistant who provides clean, production-ready code. \
The user will ask you a question and you as the assistant solve it. \

Follow these principle to respond:
1.  **Think**: First, reason about the problem step-by-step inside a <think>...</think> block.
2.  **Answer**: After thinking, provide your final code inside an <answer>...</answer> block.
3.  The code must be a complete, executable Python solution with all necessary imports.

Now, solve the following user request.
"""

PY_IMPORTS = "import heapq\nfrom math import floor, gcd\nimport random\nimport sys\nfrom typing import *\nfrom functools import *\nimport collections\nfrom collections import *\nfrom itertools import *\nfrom heapq import *\nfrom bisect import *\nfrom string import *\nimport math\nimport datetime\ninf = float('inf')\n"


def leetcode2k():
    rich.print(Rule("Loading LeetCodeDataset..."))
    test_dataset = load_dataset("json",
                                data_files="/nvfile-heatstorage/chatrl/users/hxh/data/rule_based_rl/LeetCodeDataset/LeetCodeDataset-test.jsonl")["train"]
    print("Test set:", test_dataset)

    train_dataset = load_dataset("json",
                                data_files="/nvfile-heatstorage/chatrl/users/hxh/data/rule_based_rl/LeetCodeDataset/LeetCodeDataset-train.jsonl")["train"]
    print("Before deduplication - Training set:", train_dataset)
    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        meta_instruction = (
            "Your solution will be tested against a series of test cases that run in a specific order. "
            "The execution will stop immediately upon the first test case that fails. "
            "The reported pass rate will be the fraction of test cases you passed before the first failure. "
            "Use this information to refine your code in subsequent attempts."
        )
        def process_fn(example, idx):
            prompt = (
                f"You are a helpful programming assistant. Please solve the programming task below.\n\n"
                f"IMPORTANT NOTE ON TESTING:\n{meta_instruction}\n\n"   
                f"{example['query'].strip()}"
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="./data/")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    root_dir = args.root_dir
    hdfs_dir = args.hdfs_dir

    train_datasets = []
    test_datasets = []

    # dataset_makes = [leetcode2k, taco]
    dataset_makes = [leetcode2k]
    names = "-".join([make.__name__ for make in dataset_makes])

    for train, test in [make() for make in dataset_makes]:
        train_datasets.append(train)
        test_datasets.append(test)

    train_dataset = concatenate_datasets(train_datasets)#.shuffle(seed=666)
    test_dataset = concatenate_datasets(test_datasets)

    rich.print(Rule("Saving the final dataset"))
    print("Train set:", train_dataset)
    print("Test set:", test_dataset)

    local_dir = os.path.join(root_dir, f"code-r1-{round(len(train_dataset) / 1000)}k-{names}")
    rich.print(f"[bold green]Saving to {local_dir}...")
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=root_dir, dst=hdfs_dir)
