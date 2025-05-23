#!/bin/bash

# 打印环境变量信息
env
echo "-----------------------"
echo "MASTER_ADDR = ${MASTER_ADDR}"
echo "MASTER_PORT = ${MASTER_PORT}"
echo "RANK = ${RANK}"

# 进入项目目录
cd /afs/chatrl/users/hxh/code/verl

# 安装当前目录下的Python包
pip install -v -e .

# 根据环境变量RANK执行不同的任务
if [ "$RANK" = "0" ]; then
    # 主节点启动Ray集群
    echo "Starting Ray head node..."
    ray start --head --port=$MASTER_PORT &

    # 等待Ray启动完成
    sleep 60

    # 检查Ray状态
    echo "Checking Ray status:"
    ray status

    # 执行训练脚本，并记录日志
    echo "Running training script..."
    ./examples_sensecore/grpo_trainer/run_qwen2-32b-4node-dapo-math-rolloutn16-bs512-minibs32-tp2-maxlen8k-vllm_0_8_2.sh 2>&1 | tee logs_sensecore/run_qwen2-32b-4node-dapo-math-rolloutn16-bs512-minibs32-tp2-maxlen8k-vllm_0_8_2.log
else
    # 工作节点连接到Ray主节点
    echo "Starting Ray worker node..."
    ray start --address=${MASTER_ADDR}:${MASTER_PORT} --block
fi
