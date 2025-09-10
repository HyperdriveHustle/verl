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

Follow these steps to respond:
1.  **Think**: First, reason about the problem step-by-step inside a <think>...</think> block.
2.  **Answer**: After thinking, provide your final code inside an <answer>...</answer> block.

**OUTPUT RULES**:
1.  The code must be a complete, executable Python solution with all necessary imports.
2.  Your response should **ONLY** include the function or class definition. Do not provide any usage examples or test cases, the user already has their own test cases which will be used to execute your code automatically.

"Here is an example of the desired format:\n<think>\nI need to solve problem X. First I will do A, then B.\n</think>\n<answer>\n```python\n# final code here\n```\n</answer>"

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

        def process_fn(example, idx):
            prompt = f"You are a helpful programming assistant. Please solve the programming task below.\n\n{example['query'].strip()}"
            original_test_assertions = example['test'].strip()
            all_lines = original_test_assertions.split('\n')
            assert_lines = [line.strip() for line in all_lines if line.strip().startswith('assert')]
            test_cases_list_str = repr(assert_lines)
            new_check_function_template = f"""
def check(candidate):
    passed_count = 0
    test_cases = {test_cases_list_str}
    total_count = len(test_cases)
    for test_case in test_cases:
        try:
            # `exec` will run the assert statement.
            # `candidate` is available in this local scope.
            exec(test_case)
            passed_count += 1
        except Exception as e:
            # This catches AssertionError and any other runtime errors from the candidate code
            pass # We just count it as a failure and continue

    if total_count == 0:
        pass_rate = 0.0
    else:
        pass_rate = passed_count / total_count

    # This is the final output the sandbox will capture.
    # It provides a float reward signal.
    print(f"Pass rate: **{{pass_rate}}**")
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
