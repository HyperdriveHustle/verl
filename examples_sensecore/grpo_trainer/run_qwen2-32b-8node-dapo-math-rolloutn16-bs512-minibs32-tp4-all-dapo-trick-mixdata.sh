set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

# data
export dapo_math_train_path=${dapo_math_train_path:-/afs/chatrl/users/hxh/data/rule_based_rl/DAPO-Math-17k/data/dapo-math-17k_dedup_r1_sys_prompt.parquet}
export deepmath_openqa_train_path=${deepmath_openqa_train_path:-/afs/chatrl/users/hxh/data/rule_based_rl/DeepMath-103k/open_qa_trainset_question_with_sys.parquet}

export aime2024_test_path=${aime2024_test_path:-/afs/chatrl/users/hxh/data/rule_based_rl/AIME-2024/aime2024_sample8_r1_sys_prompt.parquet}
export aime2025_test_path=${aime2025_test_path:-/afs/chatrl/users/hxh/data/rule_based_rl/AIME-2025/aime2025_sample8_r1_sys_prompt.parquet}

train_files="['$dapo_math_train_path', '$deepmath_openqa_train_path']"
test_files="['$aime2024_test_path', '$aime2025_test_path']"

# env config
export nnode=${WORLD_SIZE:-4}

export model_path=${model_path:-/afs/chatrl/public/models/Qwen2.5-32B}
# resume config
export resume_mode=${resume_mode:-auto}
export resume_from_path=${resume_from_path:-null}

# train params
export total_epochs=${total_epochs:-10}
export vllm_tp=${vllm_tp:-4}
export train_prompt_batch_size=${train_prompt_batch_size:-512}
export grpo_rollout_n=${grpo_rollout_n:-16}
export clip_ratio_c=${clip_ratio_c:-3.0}
export clip_ratio_high=${clip_ratio_high:-0.28}
export filter_score_high=${filter_score_high:-null}
export filter_score_low=${filter_score_low:-null}
export data_shuffle=${data_shuffle:-True}
export gen_prompt_bsz=${gen_prompt_bsz:-256}

# model params
export max_response_length=${max_response_length:-8192}
export prompt_key=${prompt_key:-prompt}
# project config
export project_name=${project_name:-verl_grpo_dapomath_and_deepmath}
export resume_type=${resume_type:-resume_none}
export suffix_name=${suffix_name:-all_dapo_trick}

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10


use_dynamic_bsz=True
infer_micro_batch_size=null

max_prompt_length=$((1024 * 2))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

real_train_batch_size=$((train_prompt_batch_size * grpo_rollout_n))
ppo_mini_batch_size=32

lr=1e-6

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

echo "real_train_batch_size = ${real_train_batch_size}"

export experiment_name=Qwen25-32B-Base_grpo_${nnode}node_${resume_type}_neg_reward_rollout${grpo_rollout_n}_bs${train_prompt_batch_size}_minibatch${ppo_mini_batch_size}_lr${lr}_genbatch${gen_prompt_bsz}_tp${vllm_tp}_maxlen${max_response_length}_${suffix_name}
export TENSORBOARD_DIR=/nvfile-heatstorage/chatrl/users/hxh/models/verl_rl_models/${project_name}/${experiment_name}/tensorboard_log

offload=False
max_tokens=$((max_prompt_length  + max_response_length))
gen_max_tokens=$((max_tokens * 2))
log_prob_max_tokens=$((max_tokens * 2))

python3 -u -m recipe.dapo.src.main_dapo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.prompt_key=${prompt_key} \
    data.train_batch_size=${train_prompt_batch_size} \
    data.shuffle=${data_shuffle} \
    actor_rollout_ref.rollout.n=${grpo_rollout_n} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.truncation='left' \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.filter_score_high=${filter_score_high} \
    algorithm.filter_groups.filter_score_low=${filter_score_low} \
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
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${vllm_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=${gen_max_tokens} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.resume_mode=${resume_mode} \
    trainer.resume_from_path=${resume_from_path} \
    trainer.logger=['tensorboard'] \
    trainer.default_local_dir=/afs/chatrl/users/hxh/models/verl_rl_models/${project_name}/${experiment_name} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${nnode} \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=${total_epochs} 2>&1 | tee /afs/chatrl/users/hxh/code/verl/logs_sensecore/$experiment_name.log
