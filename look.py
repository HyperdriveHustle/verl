from datasets import load_dataset
from pathlib import Path

parquet_dir = "/afs/chatrl/users/lyy/data/code/livecodebench/sp_none_prompt_none/2408_2502"

# 加载整个目录的所有 parquet 文件为一个 dataset
dataset = load_dataset(
    "parquet",
    data_files=[str(p) for p in Path(parquet_dir).glob("*.parquet")]
)["train"]  # 默认 split 是 "train"

# 打印前 5 个样本
print("First 5 samples:")
for i in range(min(5, len(dataset))):
    print(f"--- Sample {i} ---")
    print(dataset[i])