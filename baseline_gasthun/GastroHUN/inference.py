import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import json
import time
import pandas as pd
from tqdm import tqdm

# ================= 1. 配置路径与网络参数 =================
# 💡 填入你下载的 ConvNeXt 权重文件的绝对路径
BASELINE_WEIGHTS_PATH = "/home/ren9/yidong-code/exendovla/baseline_gasthun/GastroHUN/models/best-model-val_f1_macro1.ckpt" 

TEST_JSONL_PATH = "/home/ren9/yidong-code/exendovla/exvla/output_dir/gastrohun_llm_en_images_train_sft_10.jsonl"
OUTPUT_EXCEL_PATH = "/home/ren9/yidong-code/exendovla/exvla/baseline_evaluation_clinical_metrics.xlsx"

# 严格按照字母顺序排列的 23 个分类
CLASSES = [
    "A1", "A2", "A3", "A4", "A5", "A6", 
    "G1", "G2", "G3", "G4", 
    "L1", "L2", "L3", "L4", "L5", "L6", 
    "NA", 
    "P1", "P2", "P3", "P4", "P5", "P6"
]

print("="*50)
print(f"正在加载 GastroHUN 官方 Baseline 模型")

# ================= 2. 加载纯视觉 Baseline 模型 =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🚨 核心修改：使用 torchvision 构建 ConvNeXt Tiny 架构
# model = models.convnext_tiny(weights=None)
model = models.convnext_large(weights=None)
# 替换分类头，将默认的 1000 类改为 GastroHUN 的 23 类
in_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(in_features, len(CLASSES))

# 加载论文权重并进行字典匹配
if os.path.exists(BASELINE_WEIGHTS_PATH):
    checkpoint = torch.load(BASELINE_WEIGHTS_PATH, map_location=device)
    
    # 兼容 PyTorch Lightning
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    clean_state_dict = {}
    for k, v in state_dict.items():
        # 去除前缀 'model.' 等，使之与 torchvision 的键名完美匹配
        new_k = k.replace('module.', '').replace('model.', '').replace('base_model.', '')
        clean_state_dict[new_k] = v
        
    try:
        missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict, strict=False)
        print("✅ 论文 ConvNeXt 权重加载成功！")
        
        if missing_keys:
            print(f"⚠️ 提示：仍有 {len(missing_keys)} 个参数未找到。")
    except Exception as e:
        print(f"⚠️ 权重加载遇到错误: {e}")
else:
    print(f"⚠️ 警告：未找到 {BASELINE_WEIGHTS_PATH}，使用随机权重测试。")

model.to(device)
model.eval()

# =========================================================
# 🚀 预处理：ConvNeXt 通常使用 224x224 尺寸
# =========================================================
transform = transforms.Compose([
    transforms.Resize(256),         
    transforms.CenterCrop(224),     
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
print("="*50)

# ================= 3. 加载测试数据 =================
test_samples = []
with open(TEST_JSONL_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            test_samples.append(json.loads(line))

print(f"📊 共加载了 {len(test_samples)} 条测试数据。即将开始 Baseline 推理...")

# ================= 4. 开启推理与多维对齐评测 =================
results_list = []
depth_correct = 0
depth_tolerant_correct = 0
wall_correct = 0
feature_hit_correct = 0
total_valid_count = 0

for sample in tqdm(test_samples, desc="Baseline 推理进度"):
    img_path = sample.get("image_path")
    gt_answer = str(sample.get("answer", "")).strip().upper() 
    sample_id = sample.get("id", "Unknown")
    
    if not os.path.exists(img_path):
        continue

    try:
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        start_time = time.time()

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            pred_label = CLASSES[predicted_idx.item()] 

        end_time = time.time()
        latency = end_time - start_time

        # ==========================================
        # 🚀 临床多维评测逻辑 (已同步新标尺)
        # ==========================================
        # 统一清理 gt_answer 中的空格，防止出现 "OTHER CLASS" 导致匹配失败
        gt_clean = gt_answer.replace(" ", "")
        
        is_depth = False
        is_depth_tolerant = False
        is_wall = False
        is_feature_hit = False

        # 1. 严格拦截：如果真实标签是 OTHERCLASS，只有预测出 NA 才算对，否则全错
        if gt_clean == "OTHERCLASS":
            if pred_label == "NA" or pred_label == "OTHERCLASS":
                is_depth = is_depth_tolerant = is_wall = is_feature_hit = True
            else:
                is_depth = is_depth_tolerant = is_wall = is_feature_hit = False
                
        # 2. 完全匹配 (预测和真实标签完全一致)
        elif pred_label == gt_clean or pred_label == gt_answer:
            is_depth = is_depth_tolerant = is_wall = is_feature_hit = True
            
        # 3. SSS 标签的拆解与多维判定
        elif len(gt_clean) == 2 and len(pred_label) == 2 and gt_clean != "NA" and pred_label != "NA":
            gt_wall, gt_num = gt_clean[0], int(gt_clean[1])
            pred_wall, pred_num = pred_label[0], int(pred_label[1])
            
            # 纵向深度判断
            if gt_num == pred_num:
                is_depth = True
            # 深度 ±1 级容错判断
            if abs(gt_num - pred_num) <= 1:
                is_depth_tolerant = True
            # 圆周方位判断
            if gt_wall == pred_wall:
                is_wall = True
            # 核心特征捕获（方位或深度有一个对上了就行）
            is_feature_hit = (is_depth or is_wall)

        # 累计分数
        if is_depth: depth_correct += 1
        if is_depth_tolerant: depth_tolerant_correct += 1
        if is_wall: wall_correct += 1
        if is_feature_hit: feature_hit_correct += 1
        total_valid_count += 1

        results_list.append({
            "样本 ID": sample_id,
            "图片路径": img_path,
            "原本正确的部位": gt_answer,
            "模型推理的部位": pred_label,
            "纵向深度命中": is_depth,
            "±1级深度容错": is_depth_tolerant,
            "圆周方位命中": is_wall,
            "特征有效捕获": is_feature_hit,
            "推理延迟 (秒)": round(latency, 4)
        })

    except Exception as e:
        results_list.append({
            "样本 ID": sample_id,
            "图片路径": img_path,
            "原本正确的部位": gt_answer,
            "模型推理的部位": f"ERROR: {str(e)}",
            "纵向深度命中": False,
            "±1级深度容错": False,
            "圆周方位命中": False,
            "特征有效捕获": False,
            "推理延迟 (秒)": 0.0
        })

# ================= 5. 数据统计与导出 =================
acc_depth = (depth_correct / total_valid_count) if total_valid_count > 0 else 0
acc_depth_tol = (depth_tolerant_correct / total_valid_count) if total_valid_count > 0 else 0
acc_wall = (wall_correct / total_valid_count) if total_valid_count > 0 else 0
acc_feature = (feature_hit_correct / total_valid_count) if total_valid_count > 0 else 0
average_latency = sum(item["推理延迟 (秒)"] for item in results_list) if results_list else 0
average_latency = average_latency / len(results_list) if results_list else 0

df_results = pd.DataFrame(results_list)

summary_row = {
    "样本 ID": "【Baseline 对比评估】",
    "图片路径": f"总计测试: {total_valid_count} 张",
    "原本正确的部位": "",
    "模型推理的部位": f"Baseline (torchvision_tiny) 汇总 ->",
    "纵向深度命中": f"{acc_depth * 100:.2f}%",
    "±1级深度容错": f"{acc_depth_tol * 100:.2f}%",
    "圆周方位命中": f"{acc_wall * 100:.2f}%",
    "特征有效捕获": f"{acc_feature * 100:.2f}%",
    "推理延迟 (秒)": f"平均: {average_latency:.4f} 秒"
}

df_results = pd.concat([df_results, pd.DataFrame([summary_row])], ignore_index=True)
df_results.to_excel(OUTPUT_EXCEL_PATH, index=False)

print("\n" + "="*50)
print("🎉 Baseline 评测完成！结果已完美对齐你的多维评价标准：")
print(f"📏 纵向深度精准率: {acc_depth*100:.2f}%")
print(f"⚖️ ±1级深度容错率: {acc_depth_tol*100:.2f}%")
print(f"🧭 圆周方位精准率: {acc_wall*100:.2f}%")
print(f"🎯 核心特征捕获率: {acc_feature*100:.2f}%")
print(f"💾 报表已保存至: {OUTPUT_EXCEL_PATH}")
print("="*50)