import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import os
import json
import time
import pandas as pd
from tqdm import tqdm
import re

# ================= 0. 导入训练配置 =================
try:
    from config import IMAGE_WIDTH, IMAGE_HEIGHT
except ImportError:
    IMAGE_WIDTH, IMAGE_HEIGHT = 1024, 768 

# ================= 1. 配置路径 =================
model_path = "/home/rennc1/Documents/Yidong_code/exvla/models/grpo_5090_10percent_image_newDora_Ultimate_Merged"
test_jsonl_path = "/home/rennc1/Documents/Yidong_code/exvla/output_dir/gastrohun_llm_en_images_train_sft_10.jsonl"
output_excel_path = "/home/rennc1/Documents/Yidong_code/exvla/evaluation_report_clinical_metrics.xlsx"

print("="*50)
print("🚀 正在加载模型与处理器，请稍候...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True
)
model.eval() 
print("✅ 模型加载成功！")
print("="*50)

# ================= 3. 加载测试数据 =================
test_samples = []
with open(test_jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            test_samples.append(json.loads(line))

print(f"📊 共加载了 {len(test_samples)} 条测试数据。即将开始批量推理...")

# ================= 4. 开启批量推理与记录 =================
results_list = []
depth_correct = 0
depth_tolerant_correct = 0
wall_correct = 0
feature_hit_correct = 0
total_valid_count = 0

for sample in tqdm(test_samples, desc="推理进度"):
    img_path = sample.get("image_path")
    gt_answer = str(sample.get("answer", "")).strip().upper() 
    sample_id = sample.get("id", "Unknown")
    
    sys_instruction = sample.get("system_instruction", "")
    user_instruction = sample.get("user_instruction", "")
    
    if not os.path.exists(img_path):
        continue

    try:
        image = Image.open(img_path).convert("RGB")
        if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
            image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)
        
        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_instruction}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_instruction}
            ]}
        ]

        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to("cuda")

        start_time = time.time()

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip().upper()

        end_time = time.time()
        latency = end_time - start_time

        # ==========================================

        # ==========================================
        match = re.search(r'\b([AGLP][1-6]|NA|OTHERCLASS)\b', output_text)
        pred_label = match.group(1) if match else output_text

        is_depth = False
        is_depth_tolerant = False
        is_wall = False
        is_feature_hit = False

        # 1. 特例：OTHERCLASS 或 完全相等的 NA
        if gt_answer == "OTHERCLASS" or pred_label == gt_answer:
            is_depth = is_depth_tolerant = is_wall = is_feature_hit = True
            
        # 2. SSS 标签的拆解与多维判定
        elif len(gt_answer) == 2 and len(pred_label) == 2 and gt_answer != "NA" and pred_label != "NA":
            gt_wall, gt_num = gt_answer[0], int(gt_answer[1])
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

# ================= 5. 数据统计与生成最后一行汇总 =================
acc_depth = (depth_correct / total_valid_count) if total_valid_count > 0 else 0
acc_depth_tol = (depth_tolerant_correct / total_valid_count) if total_valid_count > 0 else 0
acc_wall = (wall_correct / total_valid_count) if total_valid_count > 0 else 0
acc_feature = (feature_hit_correct / total_valid_count) if total_valid_count > 0 else 0
average_latency = sum(item["推理延迟 (秒)"] for item in results_list) / len(results_list) if results_list else 0

df_results = pd.DataFrame(results_list)

summary_row = {
    "样本 ID": "【临床能力评估】",
    "图片路径": f"总计测试: {total_valid_count} 张",
    "原本正确的部位": "",
    "模型推理的部位": "大模型辅助指标 ->",
    "纵向深度命中": f"{acc_depth * 100:.2f}%",
    "±1级深度容错": f"{acc_depth_tol * 100:.2f}%",
    "圆周方位命中": f"{acc_wall * 100:.2f}%",
    "特征有效捕获": f"{acc_feature * 100:.2f}%",
    "推理延迟 (秒)": f"平均: {average_latency:.4f} 秒"
}

df_results = pd.concat([df_results, pd.DataFrame([summary_row])], ignore_index=True)
df_results.to_excel(output_excel_path, index=False)

print("\n" + "="*50)
print("🎉 评测完成！临床应用视角下的数据：")
print(f"📏 纵向深度精准率: {acc_depth*100:.2f}%")
print(f"⚖️ ±1级深度容错率: {acc_depth_tol*100:.2f}%  <-- 汇报核心亮点")
print(f"🧭 圆周方位精准率: {acc_wall*100:.2f}%")
print(f"🎯 核心特征捕获率: {acc_feature*100:.2f}%  <-- 汇报核心亮点")
print(f"💾 报表已保存至: {output_excel_path}")
print("="*50)
# import torch
# from transformers import AutoProcessor, AutoModelForVision2Seq
# from PIL import Image
# import os
# import json
# import time
# import pandas as pd
# from tqdm import tqdm

# # ================= 0. 导入训练配置 =================
# try:
#     from config import IMAGE_WIDTH, IMAGE_HEIGHT
# except ImportError:
#     IMAGE_WIDTH, IMAGE_HEIGHT = 1024, 768 

# # ================= 1. 配置路径 =================
# model_path = "/home/rennc1/Documents/Yidong_code/exvla/models/grpo_5090_10percent_image_newDora_Ultimate_Merged"
# test_jsonl_path = "/home/rennc1/Documents/Yidong_code/exvla/output_dir/gastrohun_llm_en_images_train_sft_10.jsonl"
# output_excel_path = "/home/rennc1/Documents/Yidong_code/exvla/evaluation_report.xlsx"

# print("="*50)
# print("🚀 正在加载模型与处理器，请稍候...")
# # ================= 2. 原生 Transformers 加载 =================
# processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModelForVision2Seq.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     device_map="cuda",
#     trust_remote_code=True
# )
# model.eval() 
# print("✅ 模型加载成功！")
# print("="*50)

# # ================= 3. 加载测试数据 =================
# test_samples = []
# with open(test_jsonl_path, 'r', encoding='utf-8') as f:
#     for line in f:
#         line = line.strip()
#         if line:
#             test_samples.append(json.loads(line))

# print(f"📊 共加载了 {len(test_samples)} 条测试数据。即将开始批量推理...")

# # ================= 4. 开启批量推理与记录 =================
# results_list = []
# correct_count = 0
# total_valid_count = 0

# for sample in tqdm(test_samples, desc="推理进度"):
#     img_path = sample.get("image_path")
#     # 正不正确以 jsonl 里的 answer 为准
#     gt_answer = str(sample.get("answer", "")).strip() 
#     sample_id = sample.get("id", "Unknown")
    
#     sys_instruction = sample.get("system_instruction", "")
#     user_instruction = sample.get("user_instruction", "")
    
#     if not os.path.exists(img_path):
#         results_list.append({
#             "样本 ID": sample_id,
#             "图片路径": img_path,
#             "原本正确的部位": gt_answer,
#             "模型推理的部位": "ERROR: 图片未找到",
#             "是否正确": False,
#             "推理延迟 (秒)": 0.0
#         })
#         continue

#     try:
#         image = Image.open(img_path).convert("RGB")
#         if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
#             image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)
        
#         messages = [
#             {"role": "system", "content": [{"type": "text", "text": sys_instruction}]},
#             {"role": "user", "content": [
#                 {"type": "image", "image": image},
#                 {"type": "text", "text": user_instruction}
#             ]}
#         ]

#         text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to("cuda")

#         start_time = time.time()

#         with torch.no_grad():
#             generated_ids = model.generate(
#                 **inputs,
#                 max_new_tokens=128,
#                 do_sample=False,
#             )

#         generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
#         output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

#         end_time = time.time()
#         latency = end_time - start_time

#         # 🚀 核心修改：如果正确答案是 other class，则无条件算作正确
#         if gt_answer.upper() == "OTHERCLASS":
#             is_correct = True
#         else:
#             is_correct = (output_text.upper() == gt_answer.upper())
        
#         if is_correct:
#             correct_count += 1
#         total_valid_count += 1

#         results_list.append({
#             "样本 ID": sample_id,
#             "图片路径": img_path,
#             "原本正确的部位": gt_answer,
#             "模型推理的部位": output_text,
#             "是否正确": is_correct,
#             "推理延迟 (秒)": round(latency, 4)
#         })

#     except Exception as e:
#         results_list.append({
#             "样本 ID": sample_id,
#             "图片路径": img_path,
#             "原本正确的部位": gt_answer,
#             "模型推理的部位": f"ERROR: {str(e)}",
#             "是否正确": False,
#             "推理延迟 (秒)": 0.0
#         })

# # ================= 5. 数据统计与生成最后一行汇总 =================
# accuracy = (correct_count / total_valid_count) if total_valid_count > 0 else 0
# average_latency = sum(item["推理延迟 (秒)"] for item in results_list) / len(results_list) if results_list else 0

# # 转换为 DataFrame
# df_results = pd.DataFrame(results_list)

# # 构造最后一行，填入准确率和延迟等汇总信息
# summary_row = {
#     "样本 ID": "【统计汇总】",
#     "图片路径": f"总计测试: {total_valid_count} 张",
#     "原本正确的部位": f"正确数量: {correct_count} 张",
#     "模型推理的部位": "",
#     "是否正确": f"模型正确率: {accuracy * 100:.2f}%",
#     "推理延迟 (秒)": f"平均延迟: {average_latency:.4f} 秒"
# }

# # 将最后一行拼接到总表的最下方
# df_results = pd.concat([df_results, pd.DataFrame([summary_row])], ignore_index=True)

# # 直接导出为一个没有任何花哨格式的单页 Excel
# df_results.to_excel(output_excel_path, index=False)

# print("\n" + "="*50)
# print("🎉 评测完成！")
# print(f"📈 模型正确率: {accuracy*100:.2f}% ({correct_count}/{total_valid_count})")
# print(f"⏱️ 平均推理延迟: {average_latency:.4f} 秒/张")
# print(f"💾 报表已保存至: {output_excel_path}")
# print("="*50)