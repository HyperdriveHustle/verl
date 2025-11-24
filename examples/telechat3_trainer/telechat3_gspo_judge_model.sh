set -x

# export HF_CACHE_DIR="/afs/chatrl/users/hxh/huggingface"
# export PYTHONPATH="${HF_CACHE_DIR}/modules:${PYTHONPATH}"

echo "✅ HF_CACHE_DIR: $HF_CACHE_DIR"
echo "✅ PYTHONPATH: $PYTHONPATH"

# 1. 异步训练核心配置：设置rollout模式与raw_chat返回
rollout_mode="async"
rollout_name="vllm"  # 异步训练推荐使用sglang，也可根据需求改为vllm
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1  # 异步模式下启用VLLM V1接口
    return_raw_chat="True"  # 异步训练必须返回原始chat格式数据
fi

# 2. 原有数据路径与模型配置（保持不变）
export dapo_math_17k=/afs/chatrl/users/hxh/data/math_data/dapo-math/prompts/dapo-math-17k_dedup_no_prompt_math_verify.parquet
export aime2024_test_path=/afs/chatrl/users/hxh/data/rule_based_rl/AIME-2024/math_verify_aime2024_sample32_no_prompt.parquet

export train_files=${train_files:-"['$dapo_math_17k']"}
export  test_files="['$aime2024_test_path']"


# 3. 原有 resume、模型、项目配置（保持不变）
export resume_mode=${resume_mode:-auto}
export resume_from_path=${resume_from_path:-null}
export model_path=${model_path:-/afs/chatrl/public/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B}
export model_name=$(basename "$model_path")

export project_name=${project_name:-verl_remote_judge_debug}

export total_epochs=${total_epochs:-50}
export vllm_tp=${vllm_tp:-2}

export train_prompt_batch_size=${train_prompt_batch_size:-32}
export ppo_mini_batch_size=${ppo_mini_batch_size:-32}

export grpo_rollout_n=${grpo_rollout_n:-8}

export max_response_length=${max_response_length:-10000}
export prompt_key=${prompt_key:-messages}

export resume_type=${resume_type:-no_resume}
export nnode=${WORLD_SIZE:-1}

export ulysses_sequence_parallel_size=${ulysses_sequence_parallel_size:-1}

export trust_remote_code=${trust_remote_code:-False}

export save_freq=${save_freq:-10}
export test_freq=${test_freq:-10}
export val_before_train=${val_before_train:-True}

# 4. 原有 GSPO 核心参数（clip、loss_mode等，保持不变）
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.0003
clip_ratio_high=0.0004
loss_agg_mode="seq-mean-token-mean"

enable_filter_groups=False
filter_groups_metric=acc

use_dynamic_bsz=True
infer_micro_batch_size=null
max_prompt_length=$((1024 * 2))

enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

export gen_prompt_bsz=${gen_prompt_bsz:-$((train_prompt_batch_size * 1))}
real_train_batch_size=$((train_prompt_batch_size * grpo_rollout_n))

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

export base_url=${base_url:-http://app-2abf503c001748c4967f6b495c322ffc.ns-bjdianxin-cb517126.svc.cluster.local:6669/v1}
export api_key=${api_key:-EMPTY}
export judge_model_name=${judge_model_name:-Qwen3-30B-A3B}
percentile=90


reward_manager=${reward_manager:-dapo}

echo "real_train_batch_size = $real_train_batch_size, train_prompt_batch_size = $train_prompt_batch_size, nnode = $nnode"

sleep 1
export root_dir=${root_dir:-/afs/chatrl/users/hxh/models/verl_rl_models_telechat3}
export base_model_suffix=${base_model_suffix:-Base}
export experiment_name=GSPO-judge-verl061-${base_model_suffix}_${resume_type}${nnode}node_tp${vllm_tp}_rollout${grpo_rollout_n}_temp${temperature}_bs${train_prompt_batch_size}_minibs${ppo_mini_batch_size}_lr${lr}_sp${ulysses_sequence_parallel_size}_maxlen${max_response_length}

rm -rf /workspace/tmp_tensorboard/*
export TENSORBOARD_DIR=${root_dir}/${project_name}/${experiment_name}
# 如果路径不存在，则创建（-p 会自动逐级创建）
if [ ! -d "$TENSORBOARD_DIR" ]; then
    mkdir -p "$TENSORBOARD_DIR"
    echo "Created directory: $TENSORBOARD_DIR"
else
    echo "Directory already exists: $TENSORBOARD_DIR"
fi
# 定义存储 rollout 数据的目录
export rollout_data_dir=${root_dir}/${project_name}/${experiment_name}/rollout_data_dir
if [ ! -d "$rollout_data_dir" ]; then
    mkdir -p "$rollout_data_dir"
    echo "Created directory: $rollout_data_dir"
else
    echo "Directory already exists: $rollout_data_dir"
fi


cd /afs/chatrl/users/hxh/code/verl_gspo/verl

export HYDRA_FULL_ERROR=1

# 5. 异步训练核心命令：添加return_raw_chat、rollout.mode=async，适配rollout_name
python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.prompt_key=${prompt_key}  \
    data.train_batch_size=${train_prompt_batch_size} \
    data.return_raw_chat=$return_raw_chat \
    data.shuffle=False \
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
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${vllm_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.trust_remote_code=${trust_remote_code} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${gen_max_tokens} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.n=${grpo_rollout_n} \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    reward_model.reward_manager=${reward_manager} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.resume_mode=${resume_mode} \
    trainer.resume_from_path=${resume_from_path} \
    trainer.logger=['tensorboard'] \
    trainer.default_local_dir=${root_dir}/${project_name}/${experiment_name} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${nnode} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.val_before_train=${val_before_train} \
    trainer.total_epochs=${total_epochs} 2>&1 | tee /afs/chatrl/users/hxh/logs/verl_logs/${project_name}-${experiment_name}.log
    # trainer.rollout_data_dir=${rollout_data_dir}  \
