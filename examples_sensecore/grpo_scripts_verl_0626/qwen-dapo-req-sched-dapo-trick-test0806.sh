pip install langdetect
pip install math-verify sympy
set -x

# export dapo_train_path=${dapo_train_path:-/afs/chatrl/users/kzl/data/rule_based_rl/DAPO-Math-17k/data/dapo-math-17k_dedup.parquet}
# export aime2024_test_path=${aime2024_test_path:-/afs/chatrl/users/kzl/data/rule_based_rl/AIME-2024/dapo_aime2024_sample8.parquet}
# export dapo_train_path=${dapo_train_path:-/afs/chatrl/users/kzl/data/rule_based_rl/filter_by_32b_cold_start_20250614/filtered_dapo-math-17k_by_acc_0.2_0.7.parquet}
# export deepmath_train_path=${deepmath_train_path:-/afs/chatrl/users/kzl/data/rule_based_rl/filter_by_32b_cold_start_20250614/filtered_deepmath_by_acc_0.2_0.7.parquet}
# export math7d5k_train_path=${math7d5k_train_path:-/afs/chatrl/users/kzl/data/rule_based_rl/filter_by_32b_cold_start_20250614/filtered_math_train_by_acc_0_0.7.parquet}

export aime2024_test_path=${aime2024_test_path:-/afs/chatrl/users/hxh/data/rule_based_rl/AIME-2024/dapo_aime2024_sample8_no_prompt.parquet}
export aime2025_test_path=${aime2025_test_path:-/afs/chatrl/users/hxh/data/rule_based_rl/AIME-2025/dapo_aime2025_sample8_no_prompt.parquet}

# export aime2024_test_path_from_lyy=${aime2024_test_path_from_lyy:-/afs/chatrl/users/kzl/data/eval/aime2024_dapo_sample32_new.parquet}
# export aime2025_test_path_from_lyy=${aime2025_test_path_from_lyy:-/afs/chatrl/users/lyy/data/eval/aime2025_dapo_sample32.parquet}

export dapo_train_path=/afs/chatrl/users/hxh/data/math_data/dapo-math/rule_based_rl/dapo-math-17k_dedup_no_prompt_sft_0614_acc_0d1-0d7.parquet
export math7d5k_train_path=/afs/chatrl/users/hxh/data/math_data/MATH_train/rule_based_rl/train_7d5k_math_verify_sft_0614_acc_0-0d7.parquet

export math_zh=/afs/chatrl/users/kzl/code/awesome_scripts/math_zh_test/processed_data/0730-250627-0710_science_label_question_combined-12121_single_no_repeat_30.parquet

export math_dapo_zh=/afs/chatrl/users/kzl/data/math_data/dapo-math/prompts/dapo-math-17k_zh.parquet
export math_dapo_en=/afs/chatrl/users/kzl/data/math_data/dapo-math/prompts/dapo-math-17k_en.parquet
# export train_files="['$math7d5k_train_path', '$dapo_train_path']"

# export train_files="['$math7d5k_train_path', '$dapo_train_path']"
export train_files="['$math_dapo_zh','$math_dapo_en']"

# train_files="['$math7d5k_train_path', '$dapo_train_path', '$deepmath_train_path']"
# export train_files=${train_files:-"['$math7d5k_train_path', '$dapo_train_path', '$deepmath_train_path']"}

# test_files="['$aime2024_test_path_from_lyy', '$aime2025_test_path_from_lyy']"
test_files="['$aime2024_test_path', '$aime2025_test_path']"

# resume config
export resume_mode=${resume_mode:-auto}
export resume_from_path=${resume_from_path:-null}
export model_path=/afs/chatrl/public/models/DeepSeek-R1-Distill-Qwen-7B
export model_name=$(basename "$model_path")


# project config
export project_name=${project_name:-verl_qwen_7b_dapo_req_sched_v0626}
# train params
export total_epochs=${total_epochs:-50}
export vllm_tp=${vllm_tp:-2}

export train_prompt_batch_size=${train_prompt_batch_size:-32}
export grpo_rollout_n=${grpo_rollout_n:-16}
# model params
export max_response_length=${max_response_length:-20000}
export prompt_key=${prompt_key:-messages}
export resume_type=${resume_type:-no_resume}
# env config
export nnode=${WORLD_SIZE:-1}

export ulysses_sequence_parallel_size=${ulysses_sequence_parallel_size:-1}

export filter_score_high=${filter_score_high:-1.1}
export filter_score_low=${filter_score_low:--1}
# export filter_score_high=${filter_score_high:-0.7}
# export filter_score_low=${filter_score_low:-0.2}

export save_freq=${save_freq:-20}
export test_freq=${test_freq:-20}

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

loss_agg_mode="token-mean"

enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=10


use_dynamic_bsz=True
infer_micro_batch_size=null

max_prompt_length=$((1024 * 2))

export val_before_train=${val_before_train:-True}
export trust_remote_code=${trust_remote_code:-False}

export enable_overlong_buffer=${enable_overlong_buffer:-False}
export overlong_buffer_len=${overlong_buffer_len:-$((1024 * 4))}
overlong_penalty_factor=1.0

export gen_prompt_bsz=${gen_prompt_bsz:-$((train_prompt_batch_size * 1))}


real_train_batch_size=$((train_prompt_batch_size * grpo_rollout_n))
ppo_mini_batch_size=32


lr=1e-6

# Algorithm
export temperature=${temperature:-1.0}
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

shuffle=False

offload=False
max_tokens=$((max_prompt_length  + max_response_length))
gen_max_tokens=$((max_tokens * 2))
log_prob_max_tokens=$((max_tokens * 2))


export seq_dir=${seq_dir:-/afs/chatrl/users/kzl/data/req_sched_seq_dir/filter_by_32b_cold_start_20250614/init}
export log_dir=${log_dir:-/afs/chatrl/users/kzl/data/req_sched_seq_dir/filter_by_32b_cold_start_20250614/log}

cap_dataset_size=$((1024 * 80000))
filter_overlong_prompts=False

#req_algo="long_short"
# req_algo="even_prompt"
# req_algo="even_token"
# agg="max" # sum / max

export req_algo=${req_algo:-even_token}
export agg=${agg:-max}

export entropy_coeff=${entropy_coeff:-0}
export entropy_max=${entropy_max:-null}

percentile=90
export TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")


echo "real_train_batch_size = $real_train_batch_size, train_prompt_batch_size = $train_prompt_batch_size, nnode = $nnode, offload = $offload, max_tokens = $max_tokens, model = $model_name, vllm_tp = $vllm_tp, vllm_mem = $vllm_mem, seq_dir = $seq_dir, log_dir = $log_dir, cap_dataset_size = $cap_dataset_size, filter_overlong_prompts = $filter_overlong_prompts, max_prompt_length = $max_prompt_length, max_response_length = $max_response_length, req_algo = $req_algo, percentile = $percentile, agg = $agg"

sleep 1
export base_model_suffix=${base_model_suffix:-Base}
export experiment_name=${model_name}-${base_model_suffix}_dapo-${req_algo}-${agg}_${nnode}node_rollout${grpo_rollout_n}_bs${train_prompt_batch_size}_minibatch${ppo_mini_batch_size}_lr${lr}_sp${ulysses_sequence_parallel_size}_tp${vllm_tp}_maxlen${max_response_length}_overlong_punish_${enable_overlong_buffer}_all_dapo_trick_${resume_type}_dapo_mix_zh_en

rm -rf /workspace/tmp_tensorboard/*
export TENSORBOARD_DIR=/afs/chatrl/users/kzl/models/verl_rl_models/${project_name}/${experiment_name}

#data.max_batch_size=${train_prompt_batch_size} \
#python3 -u -m verl.trainer.main_ppo \
# python3 -u -m verl.trainer.main_ppo_with_time \
python3 -u -m  recipe.dapo.main_dapo \
    --config-path=config \
    --config-name='dapo_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.prompt_key=${prompt_key} \
    data.train_batch_size=${train_prompt_batch_size} \
    actor_rollout_ref.rollout.n=${grpo_rollout_n} \
    data.shuffle=True \
    data.filter_overlong_prompts=${filter_overlong_prompts} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    req_scheduler.seq_dir="$seq_dir" \
    req_scheduler.log_dir="$log_dir" \
    req_scheduler.agg="$agg" \
    req_scheduler.algo="$req_algo" \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.truncation='left' \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.profiler.all_ranks=True \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    data.trust_remote_code=${trust_remote_code} \
    actor_rollout_ref.model.trust_remote_code=${trust_remote_code} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.filter_score_low=${filter_score_low} \
    algorithm.filter_groups.filter_score_high=${filter_score_high} \
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
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${vllm_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
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
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.resume_mode=${resume_mode} \
    trainer.resume_from_path=${resume_from_path} \
    trainer.logger=['tensorboard'] \
    trainer.default_local_dir=/afs/chatrl/users/kzl/models/verl_rl_models/${project_name}/${experiment_name} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=${val_before_train} \
    trainer.nnodes=${nnode} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} 2>&1 | tee /afs/chatrl/users/kzl/code/verlmix/logs_sensecore/$experiment_name.log