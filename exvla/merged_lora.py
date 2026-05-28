import json
import torch
from peft import PeftModel
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM

lora_path = "/home/ren9/yidong-code/exendovla/exvla/models/grpo_5090_10percent_image"
save_path = "/home/ren9/yidong-code/exendovla/exvla/models/grpo_5090_10percent_image_newDora_Ultimate_Merged"

print("="*50)
print("HuggingFace 模式，使用 GPU 融合...")

# 自动读取底座模型名字
with open(f"{lora_path}/adapter_config.json", "r") as f:
    config = json.load(f)
base_model_name = config["base_model_name_or_path"]

print(f"读取底座: {base_model_name} (加载到 GPU)")

# 原生加载底座到 GPU (device_map="cuda")，绕开 Unsloth
try:
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda", 
        trust_remote_code=True
    )
except Exception:
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True
    )

processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)

print("\n" + "="*50)
print("对接 DoRA 权重并进行 GPU 矩阵融合...")
peft_model = PeftModel.from_pretrained(base_model, lora_path)
merged_model = peft_model.merge_and_unload()

print("\n" + "="*50)
print("将融合后的全新纯净大模型保存至硬盘...")
merged_model.save_pretrained(save_path)
processor.save_pretrained(save_path)

print(f"融合成！")
print(f"保存在: {save_path}")
print("="*50 + "\n")