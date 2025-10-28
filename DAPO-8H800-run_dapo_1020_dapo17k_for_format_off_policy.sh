set -x


export aops_difficulty_1_15_dapo_verify=${aops_difficulty_1_15_dapo_verify:-/afs/chatrl/users/hwq/data/expert/aops/numinamath1.5_aops_forum_format_with_16.parquet}
export aops_with_expert_cot=${aops_with_expert_cot:-/afs/chatrl/users/hwq/data/numina_cot/aops_forum_sky_with_solution_cot_fuzzy_filtered_acc_Distill-7B_16_prompt_format_filtered_add_answer_tag.parquet}
export dapo_math_17k=${dapo_math_17k:-/afs/chatrl/users/hxh/data/rule_based_rl/DAPO-Math-17k/data/dapo-math-17k_dedup.parquet}

export aime2024_test_path_from_lyy=${aime2024_test_path_from_lyy:-/afs/chatrl/users/hxh/data/rule_based_rl/AIME-2024/data/aime-2024.parquet}
export aime2025_test_path_from_lyy=${aime2025_test_path_from_lyy:-/afs/chatrl/users/hwq/data/aime/aime2025_dapo_sample64.parquet}
# {"data_source":"dapo_aime2024_s8","prompt":[{"content":"Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\nThere exist real numbers $x$ and $y$, both greater than 1, such that $\\log_x\\left(y^x\\right)=\\log_y\\left(x^{4y}\\right)=10$. Find $xy$.\n\nRemember to put your answer on its own line after \"Answer:\".","role":"user"}],"ability":"MATH","reward_model":{"ground_truth":"25","style":"rule-lighteval\/MATH_v2"},"extra_info":{"index":24,"raw_problem":"There exist real numbers $x$ and $y$, both greater than 1, such that $\\log_x\\left(y^x\\right)=\\log_y\\left(x^{4y}\\right)=10$. Find $xy$.","split":null}}

#export train_files=${train_files:-"['$deepmath_openqa_math_judge_path', '$train_7d5k_with_refined_answers_math_judge_path', '$aops_forum_sky_processed_math_judge_path', '$olympiads_sky_processed_math_judge_path']"}

#export train_files=${train_files:-"['$train_7d5k_with_refined_answers_math_judge_path', '$aops_forum_sky_processed_math_judge_path', '$olympiads_sky_processed_math_judge_path']"}
export aime2024_with_math_verify_boxed_path=${aime2024_with_math_verify_boxed_path:-/afs/chatrl/users/hwq/data/sky_work_data/aime_from_lyy/aime2024_math_verify_boxed_sample32.parquet}
export aime2025_with_math_verify_boxed_path=${aime2025_with_math_verify_boxed_path:-/afs/chatrl/users/hwq/data/sky_work_data/aime_from_lyy/aime2025_math_verify_boxed_sample32.parquet}
export math500_with_math_verify_boxed_path=${math500_with_math_verify_boxed_path:-/afs/chatrl/users/hwq/data/math500/math500_test_converted_int_idx.parquet}

export train_files=${train_files:-"['$dapo_math_17k']"}
#export train_files=${train_files:-"['$aops_with_judge_path']"}
train_data=${train_data:-dapo_17k}
#train_data=${train_data:-deepmath}
test_files="['$aime2024_test_path_from_lyy']"

# test_files="['$aime2024_test_path_from_lyy', '$aime2025_test_path_from_lyy']"
#test_files="['$aime2024_with_math_verify_boxed_path', '$aime2025_with_math_verify_boxed_path', '$math500_with_math_verify_boxed_path']"




# resume config
export resume_mode=${resume_mode:-auto}
export resume_from_path=${resume_from_path:-null}
export model_path=${model_path:-/afs/chatrl/public/models/DeepSeek-R1-Distill-Qwen-7B}
export model_name=$(basename "$model_path")

# project config
export project_name=${project_name:-verl_expert}
# train params
export total_epochs=${total_epochs:-50}
export vllm_tp=${vllm_tp:-1}

export train_prompt_batch_size=${train_prompt_batch_size:-256}
export grpo_rollout_n=${grpo_rollout_n:-8}
# model params
export max_response_length=${max_response_length:-16384}
export prompt_key=${prompt_key:-prompt}
export resume_type=${resume_type:-no_resume}
# env config
export nnode=${WORLD_SIZE:-1}

export ulysses_sequence_parallel_size=${ulysses_sequence_parallel_size:-1}


use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.27

loss_agg_mode="token-mean"

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

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

shuffle=False

offload=False
max_tokens=$((max_prompt_length  + max_response_length))
gen_max_tokens=$((max_tokens * 2))
log_prob_max_tokens=$((max_tokens * 2))


export seq_dir=${seq_dir:-/afs/chatrl/users/zhr/data/req_sched_seq_dir/filter_by_32b_cold_start_20250614/init}
export log_dir=${log_dir:-/afs/chatrl/users/zhr/data/req_sched_seq_dir/filter_by_32b_cold_start_20250614/log}

cap_dataset_size=$((1024 * 80000))
filter_overlong_prompts=False

#req_algo="long_short"
# req_algo="even_prompt"/afs/chatrl/users/hwq/log/verl/logs_sensecore/${experiment_name}.log
# req_algo="even_token"
# agg="max" # sum / max

export req_algo=${req_algo:-even_token}
export agg=${agg:-max}

export base_url=${base_url:-http://111.31.225.52:6669/v1}
export api_key=${api_key:-EMPTY}
export judge_model_name=${judge_model_name:-Qwen3-30B-A3B}

percentile=90
export TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
reward_manager=${reward_manager:-dapo}

echo "real_train_batch_size = $real_train_batch_size, train_prompt_batch_size = $train_prompt_batch_size, nnode = $nnode, offload = $offload, max_tokens = $max_tokens, model = $model, vllm_tp = $vllm_tp, vllm_mem = $vllm_mem, seq_dir = $seq_dir, log_dir = $log_dir, cap_dataset_size = $cap_dataset_size, filter_overlong_prompts = $filter_overlong_prompts, answer_injection_ratio = $answer_injection_ratio, max_prompt_length = $max_prompt_length, max_response_length = $max_response_length, req_algo = $req_algo, percentile = $percentile, agg = $agg"

sleep 1
export base_model_suffix=${base_model_suffix:-Base}
export experiment_name=DAPO-8H800-test_reward_from_response_total_${expert_inject_enable}_answer_injection_ratio_${answer_injection_ratio}_${solution_field}_DeepSeek-R1-Distill-Qwen-7B_rollout${grpo_rollout_n}_bs${train_prompt_batch_size}_minibatch${ppo_mini_batch_size}_lr${lr}_sp${ulysses_sequence_parallel_size}_tp${vllm_tp}_maxlen${max_response_length}_TRAIN_DATA_${train_data}_reward_manager_${reward_manager}_${TIMESTAMP}
rm -rf /workspace/tmp_tensorboard/*
export TENSORBOARD_DIR=/afs/chatrl/users/zhr/models/verl_rl_models/${project_name}/${experiment_name}
export SAVE_NUM_EXAMINE_PATH_REMOTE_BATCH=/afs/chatrl/users/zhr/log/save_batch_reward/${experiment_name}.jsonl
export SAVE_NUM_EXAMINE_PATH_DAPO=/afs/chatrl/users/zhr/log/save_dapo_reward/${experiment_name}.jsonl
#data.max_batch_size=${train_prompt_batch_size} \
#python3 -u -m verl.trainer.main_ppo \
# python3 -u -m verl.trainer.main_ppo_with_time \
cd /afs/chatrl/users/zhr/code/verl060

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
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
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
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${vllm_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${gen_max_tokens} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=${reward_manager} \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.resume_mode=${resume_mode} \
    trainer.resume_from_path=${resume_from_path} \
    trainer.logger=['tensorboard'] \
    trainer.default_local_dir=/afs/chatrl/users/zhr/models/verl_rl_models/${project_name}/${experiment_name} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${nnode} \
    trainer.save_freq=6 \
    trainer.test_freq=3 \
    trainer.val_before_train=True \
    trainer.total_epochs=${total_epochs} 2>&1 | tee /afs/chatrl/users/zhr/log/verl/logs_sensecore/${experiment_name}.log