#!/bin/bash

# 打印环境变量信息
env
echo "-----------------------"
echo "MASTER_ADDR = ${MASTER_ADDR}"
echo "MASTER_PORT = ${MASTER_PORT}"
echo "RANK = ${RANK}"

# 进入项目目录
cd /afs/chatrl/users/hxh/code/verl

pip install re -i https://mirrors.aliyun.com/pypi/simple/
pip install math_verify -i https://mirrors.aliyun.com/pypi/simple/
pip install sympy -i https://mirrors.aliyun.com/pypi/simple/
# 安装当前目录下的Python包
pip install -v -e . -i https://mirrors.aliyun.com/pypi/simple/

# 根据环境变量RANK执行不同的任务
if [ "$RANK" = "0" ]; then
    # 主节点启动Ray集群
    echo "Starting Ray head node..."
    ray start --head --port=$MASTER_PORT &

    # 等待Ray启动完成
    while true; do
        echo "Checking Ray cluster status..."
        ray status
        NODE_COUNT=$(ray status | grep -c '^ 1 node_')
        EXPECTED_NODE_COUNT=${WORLD_SIZE}  # 获得任务的节点数

        echo "Current alive nodes: ${NODE_COUNT}/${EXPECTED_NODE_COUNT}"
        
        if [ "$NODE_COUNT" -eq "$EXPECTED_NODE_COUNT" ]; then
            echo "All Ray nodes are ready."
            break
        else
            echo "Waiting for all Ray nodes to be ready..."
            sleep 10
        fi
    done

    # 检查Ray状态
    echo "Checking Ray status:"
    ray status

    # 执行训练脚本，并记录日志
    echo "Running training script..."
    ./examples_sensecore/grpo_scripts_req_sched/qwen32b-dapo-req-sched-dapo-trick.sh
else
    # 工作节点连接到Ray主节点
    echo "Starting Ray worker node..."
    ray start --address=${MASTER_ADDR}:${MASTER_PORT} --block
fi
