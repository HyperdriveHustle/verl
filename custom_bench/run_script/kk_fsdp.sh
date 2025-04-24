set -x
export VLLM_ATTENTION_BACKEND=XFORMERS

kk_train_path=/workspace/datasets/kk/train.parquet
kk_test_path=/workspace/datasets/kk/test.parquet

train_files=("$kk_train_path")
test_files=("$kk_test_path")

export project_name=verl_dapo_math_grpo_vllm082

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
vllm_tp=1
vllm_mem=0.7

train_prompt_batch_size=$((real_train_batch_size / grpo_rollout_n))

nnode=1
offload=True
#model=/nvfile-heatstorage/chatrl/public/models/Qwen2.5-7B-Instruct-1M
#model=/workspace/models/llama7b
#model=/workspace/models/DeepSeek-R1-Distill-Llama-8B
model=/workspace/models/Qwen2.5-7B-Instruct-1M

seq_dir=/workspace/tmp_seq
log_dir=/workspace/tmp_log_seq
cap_dataset_size=10000
filter_overlong_prompts=True

echo "real_train_batch_size = $real_train_batch_size, train_prompt_batch_size = $train_prompt_batch_size, nnode = $nnode, offload = $offload, max_tokens = $max_tokens, model = $model, vllm_tp = $vllm_tp, vllm_mem = $vllm_mem, seq_dir = $seq_dir, log_dir = $log_dir, cap_dataset_size = $cap_dataset_size, filter_overlong_prompts = $filter_overlong_prompts, min_prompt_length = $min_prompt_length max_prompt_length = $max_prompt_length, max_response_length = $max_response_length, min_response_length = $min_response_length"

sleep 1

export experiment_name=Qwen2.5-7B-1M-Instruct_dapo_math_grpo_vllm_0_8_2_${nnode}node_rollout${grpo_rollout_n}_bs${train_prompt_batch_size}_lr${lr}_tp${vllm_tp}_maxlen${max_response_length}
#export TENSORBOARD_DIR=/nvfile-heatstorage/chatrl/users/hxh/models/verl_rl_models/${project_name}/${experiment_name}/tensorboard_log
export TENSORBOARD_DIR=/workspace/tmp

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
    req_scheduler.agg="mean" \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${vllm_tp} \
    actor_rollout_ref.rollout.name=vllm \
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
    trainer.total_epochs=5 