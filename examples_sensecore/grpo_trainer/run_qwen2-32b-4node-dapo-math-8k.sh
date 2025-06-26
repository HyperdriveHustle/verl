set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1


export nnode=${WORLD_SIZE:-1}
export lr=${lr:-1e-6}
export vllm_tp=${vllm_tp:-8}
export clip_ratio_high=${clip_ratio_high:-0.28}
export grpo_rollout_n=${grpo_rollout_n:-16}
export train_prompt_batch_size=${train_prompt_batch_size:-512}
export ppo_mini_batch_size=${ppo_mini_batch_size:-32}


dapo_train_path=/afs/chatrl/users/hxh/data/rule_based_rl/DAPO-Math-17k/data/dapo-math-17k.parquet
aime2024_test_path=/afs/chatrl/users/hxh/data/rule_based_rl/DAPO-AIME-2024/dapo_aime2024_sample8.parquet
aime2025_test_path=/afs/chatrl/users/hxh/data/rule_based_rl/AIME-2025/dapo_aime2025_sample8.parquet
# train_files="['$gsm8k_train_path', '$math_train_path']"
# test_files="['$gsm8k_test_path', '$math_test_path']"
train_files="['$dapo_train_path']"
test_files="['$aime2024_test_path', '$aime2025_test_path']"

export project_name=verl_dapo_math_grpo_dapo

kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2

loss_agg_mode="token-mean"

use_dynamic_bsz=True
infer_micro_batch_size=null

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0


gen_prompt_bsz=$((train_prompt_batch_size * 3))
real_train_batch_size=$((train_prompt_batch_size * grpo_rollout_n))

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

shuffle=True

clip_ratio_low=0.2

echo "real_train_batch_size = ${real_train_batch_size}"

export experiment_name=Qwen25-32B-Base_grpo_vllm_0_8_2_${nnode}node_reward_rollout${grpo_rollout_n}_bs${train_prompt_batch_size}_minibatch${ppo_mini_batch_size}_lr${lr}_tp${vllm_tp}_maxlen${max_response_length}_clip_high${clip_ratio_high}
export TENSORBOARD_DIR=/afs/chatrl/users/hxh/models/verl_rl_models/${project_name}/${experiment_name}/tensorboard_log

offload=False
max_tokens=$((max_prompt_length  + max_response_length))


python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=${train_prompt_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.model.path=/afs/chatrl/public/models/Qwen2.5-32B \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=null \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.shuffle=${shuffle} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${vllm_tp} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=${grpo_rollout_n} \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_tokens} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    algorithm.kl_ctrl.kl_coef=0 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_tokens} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${max_tokens} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_tokens} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    trainer.critic_warmup=0 \
    trainer.logger=['tensorboard'] \
    trainer.default_local_dir=/afs/chatrl/users/hxh/models/verl_rl_models/${project_name}/${experiment_name} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${nnode} \
    trainer.save_freq=50 \
    trainer.test_freq=20 \
    trainer.total_epochs=1 2>&1 | tee /afs/chatrl/users/hxh/code/verl/logs_sensecore/$experiment_name.log