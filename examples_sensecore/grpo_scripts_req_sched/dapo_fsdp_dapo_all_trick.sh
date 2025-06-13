set -x

# export dapo_train_path=${dapo_train_path:-/afs/chatrl/users/hxh/data/rule_based_rl/DAPO-Math-17k/data/dapo-math-17k_dedup.parquet}
# export aime2024_test_path=${aime2024_test_path:-/afs/chatrl/users/hxh/data/rule_based_rl/AIME-2024/dapo_aime2024_sample8.parquet}
export dapo_train_path=${dapo_train_path:-/nvfile-heatstorage/chatrl/users/hxh/data/rule_based_rl/DAPO-Math-17k/data/dapo-math-17k_dedup.parquet}
export aime2024_test_path=${aime2024_test_path:-/nvfile-heatstorage/chatrl/users/hxh/data/rule_based_rl/DAPO-AIME-2024/data/aime-2024.parquet}
# export aime2025_test_path=${aime2025_test_path:-/nvfile-heatstorage/chatrl/users/hxh/data/rule_based_rl/AIME-2025/dapo_aime2025_sample8.parquet}
train_files="['$dapo_train_path']"
test_files="['$aime2024_test_path']"

# resume config
export resume_mode=${resume_mode:-auto}
export resume_from_path=${resume_from_path:-null}
export model_path=${model_path:-/nvfile-heatstorage/chatrl/public/models/Qwen25-7B-Instruct}
# project config
export project_name=${project_name:-verl_dapo_math_grpo_dapo_req_sched}
# train params
export total_epochs=${total_epochs:-15}
export vllm_tp=${vllm_tp:-1}

export train_prompt_batch_size=${train_prompt_batch_size:-512}
export grpo_rollout_n=${grpo_rollout_n:-8}
# model params
export max_response_length=${max_response_length:-20000}
export prompt_key=${prompt_key:-prompt}
export resume_type=${resume_type:-resume_step230}
# env config
export nnode=${WORLD_SIZE:-1}

export ulysses_sequence_parallel_size=${ulysses_sequence_parallel_size:-1}


use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10

use_dynamic_bsz=True
infer_micro_batch_size=null

min_prompt_length=$((1 * 1))
max_prompt_length=$((1024 * 2))
min_response_length=$((1 * 1))

enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

gen_prompt_bsz=$((train_prompt_batch_size * 1))
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


export seq_dir=${seq_dir:-/nvfile-heatstorage/teleai-infra/wlw/workspace/Qwen2.5-32B-seq/seq_init}
export log_dir=${log_dir:-/nvfile-heatstorage/teleai-infra/wlw/workspace/Qwen2.5-32B-seq/seq_log}

cap_dataset_size=$((1024 * 80000))
filter_overlong_prompts=False

#req_algo="long_short"
# req_algo="even_prompt"
# req_algo="even_token"
# req_algo="even_token_kk"
# agg="max" # sum / max

export req_algo=${req_algo:-even_token}
export agg=${agg:-max}
export suffix_name=${suffix_name:-reqsched}

percentile=90


echo "real_train_batch_size = $real_train_batch_size, train_prompt_batch_size = $train_prompt_batch_size, nnode = $nnode, offload = $offload, max_tokens = $max_tokens, model = $model, vllm_tp = $vllm_tp, vllm_mem = $vllm_mem, seq_dir = $seq_dir, log_dir = $log_dir, cap_dataset_size = $cap_dataset_size, filter_overlong_prompts = $filter_overlong_prompts, min_prompt_length = $min_prompt_length max_prompt_length = $max_prompt_length, max_response_length = $max_response_length, min_response_length = $min_response_length, req_algo = $req_algo, percentile = $percentile, agg = $agg"

sleep 1

export experiment_name=Qwen25-32B-Base_grpo_math_${suffix_name}-${req_algo}-${agg}_${nnode}node_reward_rollout${grpo_rollout_n}_bs${train_prompt_batch_size}_minibatch${ppo_mini_batch_size}_lr${lr}_sp${ulysses_sequence_parallel_size}_tp${vllm_tp}_maxlen${max_response_length}_all_dapo_trick_${resume_type}_filter_data
mkdir /nvfile-heatstorage/teleai-infra/wlw/workspace/logs_sensecore
rm -rf /workspace/tmp_tensorboard/*

#data.max_batch_size=${train_prompt_batch_size} \
#python3 -u -m verl.trainer.main_ppo \
# python3 -u -m verl.trainer.main_ppo_with_time \
python3 -u -m  recipe.dapo.src.main_dapo \
    --config-path=config \
    --config-name='dapo_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.prompt_key=${prompt_key} \
    data.train_batch_size=${train_prompt_batch_size} \
    actor_rollout_ref.rollout.n=${grpo_rollout_n} \
    data.shuffle=False \
    data.filter_overlong_prompts=${filter_overlong_prompts} \
    data.min_prompt_length=${min_prompt_length} \
    data.min_response_length=${min_response_length} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    req_scheduler.seq_dir="$seq_dir" \
    req_scheduler.log_dir="$log_dir" \
    req_scheduler.agg="$agg" \
    req_scheduler.algo="$req_algo" \
    req_scheduler.percentile=$percentile \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.truncation='left' \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
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
    trainer.default_local_dir=/nvfile-heatstorage/teleai-infra/wlw/workspace/${project_name}/${experiment_name} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${nnode} \
    trainer.save_freq=10 \
    trainer.test_freq=20 \
    trainer.total_epochs=${total_epochs} 2>&1 | tee /nvfile-heatstorage/teleai-infra/wlw/workspace/logs_sensecore/$experiment_name.log