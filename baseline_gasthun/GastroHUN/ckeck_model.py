import torch
import timm

ckpt_path = "/home/ren9/yidong-code/exendovla/baseline_gasthun/GastroHUN/models/best-model-val_f1_macro.ckpt"
checkpoint = torch.load(ckpt_path, map_location="cpu")
state_dict = checkpoint.get("state_dict", checkpoint)

print("🔍 论文权重文件中的前 15 个 Key:")
for k in list(state_dict.keys())[:15]:
    print(k)

print("\n" + "="*50 + "\n")

model = timm.create_model('convnext_tiny', num_classes=23)
print("🤖 我们的 timm 模型需要的前 15 个 Key:")
for k in list(model.state_dict().keys())[:15]:
    print(k)