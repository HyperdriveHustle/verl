set -x


export dapo_math_17k=${dapo_math_17k:-/afs/chatrl/users/hxh/data/rule_based_rl/DAPO-Math-17k/data/dapo-math-17k_dedup.parquet}

export aime2024_test_path=${aime2024_test_path:-/afs/chatrl/users/hxh/data/rule_based_rl/AIME-2024/data/aime-2024.parquet}

export train_files=${train_files:-"['$dapo_math_17k']"}
export test_files=${test_files:-"['$aime2024_test_path']"}

# resume configGSPO-8H800-run_dapo_1020_dapo17k_for_format_off_policy.sh

export resume_mode=${resume_mode:-auto} # resume_path
export resume_from_path=${resume_from_path:-null} # 

export model_path=${model_path:-/afs/chatrl/public/models/DeepSeek-R1-Distill-Qwen-7B}

# project config
model_save_dir=/afs/chatrl/users/hxh/models/verl_rl_models_telechat3
export project_name=${project_name:-verl_expert}
export total_epochs=${total_epochs:-50}
export vllm_tp=${vllm_tp:-1}
export train_prompt_batch_size=${train_prompt_batch_size:-256}
export grpo_rollout_n=${grpo_rollout_n:-8}

export trust_remote_code=${trust_remote_code:-False}


export max_response_length=${max_response_length:-16384}
export prompt_key=${prompt_key:-prompt}
export resume_type=${resume_type:-no_resume}
export nnode=${WORLD_SIZE:-1}
export ulysses_sequence_parallel_size=${ulysses_sequence_parallel_size:-1}

# === 改动：clip参数与loss_mode适配GSPO ===

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.0003
clip_ratio_high=0.0004
loss_agg_mode="seq-mean-token-mean"

enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=10

use_dynamic_bsz=True
infer_micro_batch_size=null
max_prompt_length=$((1024 * 2))

enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

export gen_prompt_bsz=${gen_prompt_bsz:-$((train_prompt_batch_size * 1))}
real_train_batch_size=$((train_prompt_batch_size * grpo_rollout_n))
ppo_mini_batch_size=32
lr=1e-6

temperature=1.0
top_p=1.0
top_k=-1
shuffle=False
offload=False

max_tokens=$((max_prompt_length  + max_response_length))
gen_max_tokens=$((max_tokens * 2))
log_prob_max_tokens=$((max_tokens * 2))


cap_dataset_size=$((1024 * 80000))
filter_overlong_prompts=False
export req_algo=${req_algo:-even_token}
export agg=${agg:-max}

percentile=90
export TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
reward_manager=${reward_manager:-dapo}

echo "real_train_batch_size = $real_train_batch_size, train_prompt_batch_size = $train_prompt_batch_size, nnode = $nnode"

sleep 1
export base_model_suffix=${base_model_suffix:-telechat3}
export experiment_name=GSPO-${base_model_suffix}_${nnode}node_rollout${grpo_rollout_n}_temp${temperature}_bs${train_prompt_batch_size}_minibatch${ppo_mini_batch_size}_lr${lr}_sp${ulysses_sequence_parallel_size}_maxlen${max_response_length}

rm -rf /workspace/tmp_tensorboard/*
export TENSORBOARD_DIR=${model_save_dir}/${project_name}/${experiment_name}
# 如果路径不存在，则创建（-p 会自动逐级创建）
if [ ! -d "$TENSORBOARD_DIR" ]; then
    mkdir -p "$TENSORBOARD_DIR"
    echo "Created directory: $TENSORBOARD_DIR"
else
    echo "Directory already exists: $TENSORBOARD_DIR"
fi
export rollout_data_dir=${model_save_dir}/${project_name}/${experiment_name}/rollout_data_dir
mkdir -p "$rollout_data_dir"

cd /afs/chatrl/users/hxh/code/verl_gspo/verl

export save_freq=${save_freq:-20}
export test_freq=${test_freq:-20}

# === 改动：主入口由DAPO脚本改为标准PPO入口 ===

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.prompt_key=${prompt_key}  \
    data.train_batch_size=${train_prompt_batch_size} \
    actor_rollout_ref.rollout.n=${grpo_rollout_n} \
    data.shuffle=True \
    data.filter_overlong_prompts=${filter_overlong_prompts} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.truncation='left' \
    data.trust_remote_code=${trust_remote_code} \
    actor_rollout_ref.model.trust_remote_code=${trust_remote_code} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_tokens} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${log_prob_max_tokens} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${log_prob_max_tokens} \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ulysses_sequence_parallel_size} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.nccl_timeout=6000000000 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${vllm_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${gen_max_tokens} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    reward_model.reward_manager=${reward_manager} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.resume_mode=${resume_mode} \
    trainer.resume_from_path=${resume_from_path} \
    trainer.logger=['tensorboard'] \
    trainer.default_local_dir=${model_save_dir}/${project_name}/${experiment_name} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${nnode} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.val_before_train=True \
    trainer.rollout_data_dir=${rollout_data_dir}  \
    trainer.total_epochs=${total_epochs} 2>&1 | tee /afs/chatrl/users/hxh/logs/verl_logs/${project_name}-${experiment_name}.log
