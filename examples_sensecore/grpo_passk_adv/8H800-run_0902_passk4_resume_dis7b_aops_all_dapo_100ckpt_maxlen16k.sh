set -x

export aops_difficulty_10_16_dapo_verify=${aops_difficulty_10_16_dapo_verify:-/afs/chatrl/users/hwq/data/numina_cot/aops_forum_sky_processed_difficulty_10_16_prompt.parquet}
export aops_difficulty_all=${aops_difficulty_all:-/afs/chatrl/users/hwq/data/numina_cot/aops_forum_sky_processed_difficulty_prompt.parquet}
export aime2024_test_path_from_lyy=${aime2024_test_path_from_lyy:-/afs/chatrl/users/hwq/data/aime/aime2024_dapo_sample64.parquet}
export aime2025_test_path_from_lyy=${aime2025_test_path_from_lyy:-/afs/chatrl/users/hwq/data/aime/aime2025_dapo_sample64.parquet}
# {"data_source":"dapo_aime2024_s8","prompt":[{"content":"Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\nThere exist real numbers $x$ and $y$, both greater than 1, such that $\\log_x\\left(y^x\\right)=\\log_y\\left(x^{4y}\\right)=10$. Find $xy$.\n\nRemember to put your answer on its own line after \"Answer:\".","role":"user"}],"ability":"MATH","reward_model":{"ground_truth":"25","style":"rule-lighteval\/MATH_v2"},"extra_info":{"index":24,"raw_problem":"There exist real numbers $x$ and $y$, both greater than 1, such that $\\log_x\\left(y^x\\right)=\\log_y\\left(x^{4y}\\right)=10$. Find $xy$.","split":null}}

#export train_files=${train_files:-"['$deepmath_openqa_math_judge_path', '$train_7d5k_with_refined_answers_math_judge_path', '$aops_forum_sky_processed_math_judge_path', '$olympiads_sky_processed_math_judge_path']"}

#export train_files=${train_files:-"['$train_7d5k_with_refined_answers_math_judge_path', '$aops_forum_sky_processed_math_judge_path', '$olympiads_sky_processed_math_judge_path']"}

export train_files=${train_files:-"['$aops_difficulty_all']"}
#export train_files=${train_files:-"['$aops_with_judge_path']"}
train_data=${train_data:-aops_difficulty_all}
#train_data=${train_data:-deepmath}


test_files="['$aime2024_test_path_from_lyy', '$aime2025_test_path_from_lyy']"
#test_files="['$aime2024_with_math_verify_boxed_path', '$aime2025_with_math_verify_boxed_path', '$math500_with_math_verify_boxed_path']"

#test_files="['$aime2024_test_path_from_lyy']"


# resume config
export resume_mode=${resume_mode:-resume_path}
export resume_from_path=${resume_from_path:-/afs/chatrl/users/hwq/models/verl_rl_models/verl_expert/8H800-expert_False_answer_injection_ratio_0_rephrased_cot_DeepSeek-R1-Distill-Qwen-7B_rollout8_bs32_minibatch32_lr1e-6_sp1_tp1_maxlen16384_TRAIN_DATA_dapo_17k_reward_manager_dapo_2025-08-18_01-34-21/global_step_100}
export model_path=${model_path:-/afs/chatrl/users/hwq/models/verl_rl_models/verl_expert/8H800-expert_False_answer_injection_ratio_0_rephrased_cot_DeepSeek-R1-Distill-Qwen-7B_rollout8_bs32_minibatch32_lr1e-6_sp1_tp1_maxlen16384_TRAIN_DATA_dapo_17k_reward_manager_dapo_2025-08-18_01-34-21/global_step_100/actor/huggingface}


# project config
export project_name=${project_name:-verl_passk_adv}
# train params
export total_epochs=${total_epochs:-50}
export vllm_tp=${vllm_tp:-1}

export train_prompt_batch_size=${train_prompt_batch_size:-128}
export grpo_rollout_n=${grpo_rollout_n:-32}
# model params
export max_response_length=${max_response_length:-16384}
export prompt_key=${prompt_key:-prompt}
export resume_type=${resume_type:-resume_expert_cot}
# env config
export nnode=${WORLD_SIZE:-1}

export ulysses_sequence_parallel_size=${ulysses_sequence_parallel_size:-1}

export filter_score_high=${filter_score_high:-1.1}
export filter_score_low=${filter_score_low:--0.1}


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


export seq_dir=${seq_dir:-/afs/chatrl/users/hwq/data/req_sched_seq_dir/filter_by_32b_cold_start_20250614/init}
export log_dir=${log_dir:-/afs/chatrl/users/hwq/data/req_sched_seq_dir/filter_by_32b_cold_start_20250614/log}

cap_dataset_size=$((1024 * 80000))
filter_overlong_prompts=True
answer_injection_ratio=0

#req_algo="long_short"
# req_algo="even_prompt"/afs/chatrl/users/hwq/log/verl/logs_sensecore/${experiment_name}.log
# req_algo="even_token"
# agg="max" # sum / max
export grpo_passk=${grpo_passk:-4}

export req_algo=${req_algo:-even_token}
export agg=${agg:-max}

export base_url=${base_url:-http://111.31.225.52:6669/v1}
export api_key=${api_key:-EMPTY}
export judge_model_name=${judge_model_name:-Qwen3-30B-A3B}
reward_manager=${reward_manager:-remote_batch}
export expert_inject_enable=${expert_inject_enable:-False}

percentile=90
export TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")


echo "real_train_batch_size = $real_train_batch_size, train_prompt_batch_size = $train_prompt_batch_size, nnode = $nnode, offload = $offload, max_tokens = $max_tokens, model = $model, vllm_tp = $vllm_tp, vllm_mem = $vllm_mem, seq_dir = $seq_dir, log_dir = $log_dir, cap_dataset_size = $cap_dataset_size, filter_overlong_prompts = $filter_overlong_prompts, answer_injection_ratio = $answer_injection_ratio, max_prompt_length = $max_prompt_length, max_response_length = $max_response_length, req_algo = $req_algo, percentile = $percentile, agg = $agg"

sleep 1
export experiment_name=8H800-grpo_passk_adv_K_${grpo_passk}_DeepSeek-R1-Distill-Qwen-7B_rollout${grpo_rollout_n}_bs${train_prompt_batch_size}_minibatch${ppo_mini_batch_size}_lr${lr}_sp${ulysses_sequence_parallel_size}_tp${vllm_tp}_maxlen${max_response_length}_TRAIN_DATA_${train_data}_${TIMESTAMP}
rm -rf /workspace/tmp_tensorboard/*
export TENSORBOARD_DIR=/afs/chatrl/users/hwq/models/verl_rl_models/${project_name}/${experiment_name}
export SAVE_NUM_EXAMINE_PATH_REMOTE_BATCH=/afs/chatrl/users/hwq/log/save_batch_reward/${experiment_name}.jsonl
export SAVE_NUM_EXAMINE_PATH_DAPO=/afs/chatrl/users/hwq/log/save_dapo_reward/${experiment_name}.jsonl
#data.max_batch_size=${train_prompt_batch_size} \
#python3 -u -m verl.trainer.main_ppo \
# python3 -u -m verl.trainer.main_ppo_with_time \
cd /afs/chatrl/users/hwq/code/verl-req-sched

python3 -u -m  recipe.dapo.main_dapo \
    --config-path=config \
    --config-name='dapo_trainer.yaml' \
    algorithm.adv_estimator=grpo_passk_adv \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.prompt_key=${prompt_key} \
    data.train_batch_size=${train_prompt_batch_size} \
    actor_rollout_ref.rollout.n=${grpo_rollout_n} \
    data.shuffle=True \
    data.filter_overlong_prompts=${filter_overlong_prompts} \
    data.answer_injection_ratio=${answer_injection_ratio} \
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
    algorithm.grpo_passk=${grpo_passk} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.filter_score_low=${filter_score_low} \
    algorithm.filter_groups.filter_score_high=${filter_score_high} \
    algorithm.expert_inject.enable=${expert_inject_enable} \
    algorithm.expert_inject.gold_cot_file=${aops_with_expert_cot} \
    algorithm.expert_inject.tokenizer_name=${model_path} \
    algorithm.expert_inject.max_len=${max_tokens} \
    algorithm.expert_inject.num_repeat=${grpo_rollout_n} \
    algorithm.expert_inject.solution_field=${solution_field} \
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
    actor_rollout_ref.rollout.val_kwargs.temperature=1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=${reward_manager} \
    remote_reward.base_url=${base_url} \
    remote_reward.api_key=EMPTY \
    remote_reward.model_name=${judge_model_name} \
    remote_reward.save_judge_path=/afs/chatrl/users/hwq/log/verl/logs_sensecore/${experiment_name}_remote_reward_output.jsonl \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.resume_mode=${resume_mode} \
    trainer.resume_from_path=${resume_from_path} \
    trainer.logger=['tensorboard'] \
    trainer.default_local_dir=/afs/chatrl/users/hwq/models/verl_rl_models/${project_name}/${experiment_name} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${nnode} \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.val_before_train=False \
    trainer.total_epochs=${total_epochs} 2>&1 | tee /afs/chatrl/users/hwq/log/verl/logs_sensecore/${experiment_name}.log