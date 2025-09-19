import pandas as pd
import os

# --- 请在这里修改为您的数据文件夹路径 ---
# 这个路径是上一个脚本运行时打印出来的，例如 './data/code-r1-14k-leetcode2k'
local_dir = '/afs/chatrl/users/lyy/data/code_train/DeepCoder-Preview-Dataset_wlw/primeintellect/' 
# -----------------------------------------

# 构造训练和测试文件的完整路径
train_file_path = os.path.join(local_dir, 'train.parquet')
#test_file_path = os.path.join(local_dir, 'test.parquet')

# 检查文件是否存在
if not os.path.exists(train_file_path): #or not os.path.exists(test_file_path):
    print(f"错误：在 '{local_dir}' 目录下找不到 parquet 文件。")
    print("请确保您已将脚本中的 'local_dir' 变量设置为正确的路径。")
else:
    print(f"正在从 '{local_dir}' 读取文件...")

    # 使用 pandas 读取 Parquet 文件
    train_df = pd.read_parquet(train_file_path)
    #test_df = pd.read_parquet(test_file_path)

    print("\n" + "="*50)
    print("          训练集 (train.parquet)          ")
    print("="*50)
    
    # 打印训练集的基本信息（列名、非空值数量、数据类型）
    print("\n[训练集信息 .info()]")
    train_df.info()
    train_df.head().to_html("train_head.html", index=False)
    # 打印训练集的前5行数据来查看具体内容
    print("\n[训练集前5行 .head()]")
    # 设置 pandas 显示选项以避免内容被截断
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 120)
    pd.set_option('display.max_colwidth', 80) # 设置每列最大宽度
    print(train_df.head())


    print("\n" + "="*50)
    print("          测试集 (test.parquet)           ")
    print("="*50)

    # 打印测试集的基本信息
    print("\n[测试集信息 .info()]")
    #test_df.info()

    # 打印测试集的前5行数据
    print("\n[测试集前5行 .head()]")
    #print(test_df.head())