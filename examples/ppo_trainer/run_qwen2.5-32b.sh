set -x

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet
chatrl_train_path=/nvfile-heatstorage/teleai-infra/wxe/verl/data/math/train.parquet
chatrl_test_path=/nvfile-heatstorage/teleai-infra/wxe/verl/data/math/train.parquet
# train_files="['$gsm8k_train_path', '$math_train_path']"
# test_files="['$gsm8k_test_path', '$math_test_path']"
train_files="['$chatrl_train_path']"
test_files="['$chatrl_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=/nvfile-heatstorage/chatrl/users/hxh/models/rl_models/Qwen25-7B-Base-change-chat-template-math-hard-rloo-mix-reward-0220/global_step150_hf \
    actor_rollout_ref.actor.optim.lr=7e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
<<<<<<< HEAD
    actor_rollout_ref.rollout.prompt_length=4096 \
    actor_rollout_ref.rollout.response_length=4096 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
=======
>>>>>>> gm-tyx/puffin/main
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.ulysses_sequence_parallel_size=2 \
    critic.model.path=/nvfile-heatstorage/chatrl/users/hxh/models/rl_models/Qwen25-7B-Base-change-chat-template-math-hard-rloo-mix-reward-0220/global_step150_hf \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=8 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_example' \
    trainer.experiment_name='Qwen2.5-7B-Instruct_function_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 $@
