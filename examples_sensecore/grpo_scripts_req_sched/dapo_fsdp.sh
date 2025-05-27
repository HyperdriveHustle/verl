set -x

export dapo_train_path=${dapo_train_path:-/afs/chatrl/users/hxh/data/rule_based_rl/DAPO-Math-17k/data/dapo-math-17k_dedup.parquet}
export aime2024_test_path=${aime2024_test_path:-/afs/chatrl/users/hxh/data/rule_based_rl/AIME-2024/dapo_aime2024_sample8.parquet}

train_files=("$dapo_train_path")
test_files=("$aime2024_test_path")

export project_name=verl_dapo_math_grpo_req_sched

use_dynamic_bsz=True
infer_micro_batch_size=null

min_prompt_length=$((1 * 1))
max_prompt_length=$((1024 * 2))
min_response_length=$((1 * 1))
max_response_length=$((1024 * 14))
max_tokens=$((max_prompt_length + max_response_length))

grpo_rollout_n=16
train_prompt_batch_size=512
real_train_batch_size=$((train_prompt_batch_size * grpo_rollout_n))

lr=1e-6
shuffle=False

# tp=1 -> OOM??
export vllm_tp=${vllm_tp:-2}
export nnode=${WORLD_SIZE:-1}


vllm_mem=0.7

train_prompt_batch_size=$((real_train_batch_size / grpo_rollout_n))

offload=False
model=/afs/chatrl/public/models/Qwen2.5-7B-Instruct-1M

seq_dir=/afs/chatrl/users/hxh/data/req_sched_seq_dir/Qwen2.5-7B-Instruct-1M_dapo_seq/seq_init_new
log_dir=/afs/chatrl/users/hxh/data/req_sched_seq_dir/Qwen2.5-7B-Instruct-1M_dapo_seq/seq_log
cap_dataset_size=$((1024 * 80000))
filter_overlong_prompts=False

#req_algo="long_short"
#req_algo="even_prompt"
req_algo="even_token"
percentile=90

agg="sum" # sum / max

echo "real_train_batch_size = $real_train_batch_size, train_prompt_batch_size = $train_prompt_batch_size, nnode = $nnode, offload = $offload, max_tokens = $max_tokens, model = $model, vllm_tp = $vllm_tp, vllm_mem = $vllm_mem, seq_dir = $seq_dir, log_dir = $log_dir, cap_dataset_size = $cap_dataset_size, filter_overlong_prompts = $filter_overlong_prompts, min_prompt_length = $min_prompt_length max_prompt_length = $max_prompt_length, max_response_length = $max_response_length, min_response_length = $min_response_length, req_algo = $req_algo, percentile = $percentile, agg = $agg"

sleep 1

export experiment_name=Qwen2.5-7B-1M-Instruct_dapo_math_grpo_vllm_0_8_2_${nnode}node_rollout${grpo_rollout_n}_bs${train_prompt_batch_size}_lr${lr}_tp${vllm_tp}_maxlen${max_response_length}
#export TENSORBOARD_DIR=/nvfile-heatstorage/chatrl/users/hxh/models/verl_rl_models/${project_name}/${experiment_name}/tensorboard_log
# export TENSORBOARD_DIR=/workspace/tmp

rm -rf /workspace/tmp_tensorboard/*

#data.max_batch_size=${train_prompt_batch_size} \
#python3 -u -m verl.trainer.main_ppo \
python3 -u -m verl.trainer.main_ppo_with_time \
    --config-path=config \
    --config-name='hgl_fsdp.yaml' \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.00 \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=${train_prompt_batch_size} \
    data.shuffle=False \
    data.filter_overlong_prompts=${filter_overlong_prompts} \
    data.cap_dataset_size=${cap_dataset_size} \
    data.min_prompt_length=${min_prompt_length} \
    data.max_prompt_length=${max_prompt_length} \
    data.min_response_length=${min_response_length} \
    data.max_response_length=${max_response_length} \
    req_scheduler.seq_dir="$seq_dir" \
    req_scheduler.log_dir="$log_dir" \
    req_scheduler.agg="$agg" \
    req_scheduler.algo="$req_algo" \
    req_scheduler.percentile=$percentile \
    actor_rollout_ref.model.path=${model} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=null \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0. \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.shuffle=${shuffle} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_tokens} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${vllm_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${vllm_mem} \
    actor_rollout_ref.rollout.n=${grpo_rollout_n} \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_tokens} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_tokens} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    trainer.critic_warmup=0 \
    trainer.logger=['tensorboard'] \
    trainer.default_local_dir=/workspace/tmp_tensorboard \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${nnode} \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=30