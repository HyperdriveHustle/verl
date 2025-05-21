set -x

export VLLM_ATTENTION_BACKEND=XFORMERS


dapo_train_path=/nvfile-heatstorage/chatrl/users/hxh/data/rule_based_rl/DAPO-Math-17k/data/dapo-math-17k.parquet
dapo_test_path=/nvfile-heatstorage/chatrl/users/hxh/data/rule_based_rl/DAPO-AIME-2024/data/aime-2024.parquet
# train_files="['$gsm8k_train_path', '$math_train_path']"
# test_files="['$gsm8k_test_path', '$math_test_path']"
train_files="['$dapo_train_path']"
test_files="['$dapo_test_path']"

export project_name=verl_dapo_math_grpo_vllm082


use_dynamic_bsz=True
infer_micro_batch_size=null

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 15))
grpo_rollout_n=16
train_prompt_batch_size=512
real_train_batch_size=$((train_prompt_batch_size * grpo_rollout_n))
ppo_mini_batch_size=32

lr=1e-6

nnode=4
shuffle=True
vllm_tp=4

echo "real_train_batch_size = ${real_train_batch_size}"

export experiment_name=Qwen25-32B-Base_grpo_vllm_0_8_2_${nnode}node_5reward_rollout${grpo_rollout_n}_bs${train_prompt_batch_size}_minibatch${ppo_mini_batch_size}_lr${lr}_tp${vllm_tp}_maxlen${max_response_length}
export TENSORBOARD_DIR=/nvfile-heatstorage/chatrl/users/hxh/models/verl_rl_models/${project_name}/${experiment_name}/tensorboard_log

offload=False
max_tokens=$((max_prompt_length  + max_response_length))

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=${train_prompt_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.model.path=/nvfile-heatstorage/chatrl/public/models/Qwen25-32B-Base \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size=null \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.shuffle=${shuffle} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${vllm_tp} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=${grpo_rollout_n} \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_tokens} \
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
    trainer.default_local_dir=/nvfile-heatstorage/chatrl/users/hxh/models/verl_rl_models/${project_name}/${experiment_name} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${nnode} \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 $@
