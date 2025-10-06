set -x

# ================= data/model/tool =================
HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}
echo $HDFS_ROOT
#train data:
leetcode2k=/afs/chatrl/users/lyy/data/code_train/leetcode2k_wlw
lcbv5_230501_240731=/afs/chatrl/users/lyy/data/code_train/DeepCoder-Preview-Dataset_wlw/lcbv5-230501-240731
primeintellect=/afs/chatrl/users/lyy/data/code_train/DeepCoder-Preview-Dataset_wlw/primeintellect
taco=/afs/chatrl/users/lyy/data/code_train/DeepCoder-Preview-Dataset_wlw/taco

#test data:
lcbv5_test=/afs/chatrl/users/lyy/data/code_test/DeepCoder-Preview-Dataset_wlw/lcbv5-240801-250102
codeforces=/afs/chatrl/users/lyy/data/code_test/DeepCoder-Preview-Dataset_wlw/codeforces
leetcode2k_test=/afs/chatrl/users/lyy/data/code_test/leetcode2k_wlw

model_path=/afs/chatrl/public/models/Qwen3-8B
# model_path=/model/Qwen2.5-3B
# model_path=/model/Qwen25-32B-Instruct
train_files="['$taco']"
test_files="['$codeforces']"

# tool
tool_config_path=$DATA_ROOT/recipe/async_dapo_tool/sandbox_fusion_tool_config.yaml

# wandb
project_name=wlw_multi_turn
export TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
#experiment_name=wlw_multi_turn_Qwen3-4B-Instruct_2025-08-31_23-17-12 #Qwen3-14B 
#experiment_name=wlw_multi_turn_Qwen3-4B-Instruct_2025-08-31_23-17-12_500step #Qwen3-14B 500step
#experiment_name=wlw_multi_turn_Qwen25-7B-Instruct_2025-08-20_18-21-26
# experiment_name=${project_name}_Qwen25-7B-Instruct_2025-08-20_18-21-26_650step_8k
# experiment_name=wlw_multi_turn_Qwen3-4B-Instruct_2025-08-28_10-26-16
#experiment_name=wlw_multi_turn_Qwen3-8B-16k_TISfalse_reward_v3_grpo_bs32_minibs32_overlongfilter_2025-09-21_15-23-32

# ================= algorithm =================
adv_estimator=d_gigpo_ungrouped


use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

tis_imp_ratio_cap=-1 #TIS SAMPLING, if tis_imp_ratio_cap != -1, you should set actor_rollout_ref.rollout.calculate_log_probs=True
calculate_log_probs=False # if tis_imp_ratio_cap != -1, you should set calculate_log_probs=True

max_turns=4
max_prompt_length=2548
max_response_length=16384
overlong_filter=False # whether to filter out overlong samples in the Rollout(mask out)

actor_lr=1e-6

train_batch_size=32
ppo_mini_batch_size=32
n_resp_per_prompt=10 ## adjust to 10 for sample test
n_resp_per_prompt_val=1

# ================= perfomance =================
infer_tp=1 # vllm
train_sp=1 # train
offload=True
#export VLLM_USE_V1=1
actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 4 ))

experiment_name=${project_name}_Qwen3-8B-$(( max_response_length / 1024 ))k_TIS${tis_imp_ratio_cap}_reward_01reward_${adv_estimator}xiangtongT_bs${train_batch_size}_minibs${ppo_mini_batch_size}_overlongfilter${overlong_filter}_n${n_resp_per_prompt}_${TIMESTAMP}
default_local_dir=/afs/chatrl/users/wlw/ckpt/$experiment_name
export TENSORBOARD_DIR=/afs/chatrl/users/wlw/worklog/tensorboard_log/${project_name}/${experiment_name}
export VERL_LOGGING_LEVEL=INFO
#python3 -m verl.trainer.main_ppo \
#python3 -m recipe.async_dapo_tool.main_dapo \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=$HDFS_ROOT/recipe/async_dapo_tool/custom_unit_multi_turn.py \
    data.custom_cls.name=CustomRLHFDataset \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.actor.tis_imp_ratio_cap=$tis_imp_ratio_cap \
    actor_rollout_ref.rollout.calculate_log_probs=$calculate_log_probs \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.overlong_filter=$overlong_filter \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    trainer.logger=['console, tensorboard'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=True \
    trainer.val_only=False\
    trainer.log_val_generations=100 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.default_local_dir=$default_local_dir \
    trainer.test_freq=5 \
    trainer.total_epochs=1 $@ 2>&1 | tee -a /afs/chatrl/users/wlw/worklog/log/agent_multi_turn/$experiment_name.log

sleep inf
