# 设置环境变量 
export NCCL_IB_HCA=${NCCL_IB_HCA:-""}
export FIRST_HCA=$(echo $NCCL_IB_HCA | awk -F',' '{print $1}')
export NCCL_IB_HCA=$(echo $NVIDIA_VISIBLE_DEVICES | awk -F"," '{if(NF<8){print ENVIRON["FIRST_HCA"]}else{print ENVIRON["NCCL_IB_HCA"]}}')
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-2}
export NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT:-7}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}

# 设置多线程数量
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

# 设置单POD的GPU数量
GPUS_PER_NODE=8

# 设置分布式训练DDP所需的环境变量
MASTER_ADDR=${MASTER_ADDR:-'127.0.0.1'}
TROUBLESHOOT_PORT=$MASTER_PORT
NNODES=${WORLD_SIZE:-'1'}
NODE_RANK=${RANK:-'0'}
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $TROUBLESHOOT_PORT
"
# 设置NCCL的allreduce算法为RING
export NCCL_ALGO=RING

timeout 5m torchrun $DISTRIBUTED_ARGS nccl_pytorch.py -b 128M -e 1G -f 2

