from transformers import AutoModel

model_path_1 = "/nvfile-heatstorage/chatrl/public/models/Qwen25-32B-Base"
model_path_2 = "/nvfile-heatstorage/chatrl/users/hxh/models/rl_models/Qwen25-7B-Base-change-chat-template-math-hard-rloo-mix-reward-0220/global_step150_hf"

model_1 = AutoModel.from_pretrained(model_path_1)
model_2 = AutoModel.from_pretrained(model_path_2)

def print_model_layers(state_dict):
    """
    逐层打印模型的 state_dict 结构
    """
    layer_dict = {}
    
    for key in state_dict.keys():
        parts = key.split(".")
        if parts[0] not in layer_dict:
            layer_dict[parts[0]] = []
        layer_dict[parts[0]].append(key)

    for layer, params in layer_dict.items():
        print(f"🔹 Layer: {layer} ({len(params)} parameters)")
        for param in params[:]:  # 只打印前 5 个参数，防止输出太长
            print(f"   - {param}")

print("📌 Qwen25-72B-Instruct Model Layers:")
print_model_layers(model_1.state_dict())
print("="*100)
print("\n📌 Qwen25-7B-Instruct Model Layers:")
print_model_layers(model_2.state_dict())
