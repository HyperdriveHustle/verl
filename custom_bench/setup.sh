
# nvfile-heatstorage -> shared storage; can install python package, but read might be slow
# /data02 -> local storage to pods, cp models here or other read-heavy tasks

# my dir
# /nvfile-heatstorage/teleai-infra/heguoliang



# cp e2e/run_qwen_megatron.sh .
# chmod +x run_qwen_megatron.sh

cp examples/ppo_trainer/run_qwen2.5-32b.sh . 

# model path: /data02 or /nvfile-heatstorage/chatrl/public/models/

# data path: verl/data/math