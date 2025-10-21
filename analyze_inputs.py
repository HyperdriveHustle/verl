import json
import numpy as np
import argparse
from datasets import load_dataset
import os
import glob

def analyze_input_use_cases(data_paths, max_inputs=100):
    """
    Analyze the number of input test cases from the 'inputs' field inside reward_model.ground_truth.
    Expected structure:
        row["reward_model"] = {
            "style": "rule",
            "ground_truth": '{"inputs": [...], "outputs": [...]}'
        }
    Args:
        data_paths: path to parquet file or dir
        max_inputs: threshold to compute percentage of samples with <= max_inputs test cases
    """
    if isinstance(data_paths, str):
        if os.path.isdir(data_paths):
            parquet_files = glob.glob(os.path.join(data_paths, "*.parquet"))
        else:
            parquet_files = [data_paths]
    else:
        parquet_files = data_paths

    if not parquet_files:
        raise ValueError("No parquet files found!")

    all_num_inputs = []

    for file_path in parquet_files:
        print(f"Loading {file_path}...")
        try:
            ds = load_dataset("parquet", data_files=file_path)
            split = list(ds.keys())[0]
            dataset = ds[split]
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            continue

        print(f"Processing {len(dataset)} samples from {file_path}...")

        for row in dataset:
            try:
                reward_model = row["reward_model"]
                ground_truth_str = reward_model["ground_truth"]
                ground_truth = json.loads(ground_truth_str)
                inputs = ground_truth.get("inputs", [])
                num_inputs = len(inputs)
                all_num_inputs.append(num_inputs)
            except Exception as e:
                continue

    if not all_num_inputs:
        print("No valid 'reward_model.ground_truth.inputs' found in the dataset!")
        return

    arr = np.array(all_num_inputs)
    mean_val = np.mean(arr)
    min_val = np.min(arr)
    max_val = np.max(arr)
    std_val = np.std(arr)
    total_samples = len(arr)

    # 新增：计算 <= max_inputs 的比例
    count_le = np.sum(arr <= max_inputs)
    pct_le = count_le / total_samples * 100

    unique, counts = np.unique(arr, return_counts=True)
    dist = dict(zip(unique, counts))

    print("\n" + "="*60)
    print("📊 Input Use Case (inputs) Statistics")
    print("="*60)
    print(f"Total samples analyzed: {total_samples}")
    print(f"Mean number of inputs per sample: {mean_val:.2f}")
    print(f"Min: {min_val}, Max: {max_val}")
    print(f"Std: {std_val:.2f}")
    print(f"\n[Threshold Analysis] Samples with ≤ {max_inputs} inputs: {count_le} / {total_samples} ({pct_le:.2f}%)")
    print("="*60)

    sorted_items = sorted(dist.items(), key=lambda item: item[1], reverse=True)
    print("\nDistribution (num_inputs → count → percentage) [sorted by % desc]:")
    for num_inputs, count in sorted_items:
        pct = count / total_samples * 100
        print(f"  {num_inputs:4d} → {count:5d} → {pct:5.1f}%")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze input test case counts in coding datasets.")
    parser.add_argument("data_path", type=str, help="Path to a parquet file or directory.")
    parser.add_argument("--max_inputs", type=int, default=10, help="Threshold to compute percentage of samples with <= N inputs (default: 100).")
    args = parser.parse_args()

    analyze_input_use_cases(args.data_path, max_inputs=args.max_inputs)