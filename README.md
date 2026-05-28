## Endo_VLM

本仓库包含：
- `exvla/`：EndoVLA 训练/推理代码
- `baseline_gasthun/`：GastroHUN baseline 相关代码
- `exendovla.yml`：conda 环境定义（从原 `exendovla` 导出）

---

## 1) 使用 `exendovla.yml` 创建一个新的 conda 环境（推荐）

### 1.1 在当前机器创建新环境
# 0) 进入仓库（有 exendovla.yml 的目录）
cd /home/ren9/yidong-code/exendovla/Endo_VLM

# 1) 在“源机器/源环境”导出（我原来的大环境叫 exendovla）
conda activate exendovla
conda env export --no-builds > exendovla.yml

# 2)  exendovla.yml （已在仓库里）
# rsync/scp 都行（示例略）

# 3) 在“目标机器”用 yml 创建一个全新的环境（起个新名字）
conda env create -f exendovla.yml -n exendovla_dongdongjun

# 4) 激活并验证
conda activate exendovla_dongdongjun
python -V
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"



3) 运行 exvla
进入 exvla/ 目录后运行脚本，例如：

cd exvla
python -c "import torch; print('cuda_available:', torch.cuda.is_available())"
3.1 SFT 训练（示例）
cd exvla
python sft_train.py \
  --data_path output_dir/gastrohun_llm_en_multimodal_train_sft_cot_10.jsonl \
  --image_dir /media/ren9/Elements/exvla_clinical \
  --runname train_sft_cot_multimodal_v1 \
  --mode 1 \
  --epochs 3
3.2 GRPO 视频训练（示例）
cd exvla
python rft_grpo_video_train.py \
  --data_path output_dir/gastrohun_llm_en_multimodal_train.jsonl \
  --image_dir /media/ren9/Elements/exvla_clinical \
  --checkpoint models/sft_cot1_image_v1 \
  --runname grpo_video_cot1_v1 \
  --mode 2 \
  --task gastrohun \
  --epochs 3 \
  --max_completion_length 256
3.3 GRPO 图像训练（示例）
cd exvla
python rft_grpo_image_train.py \
  --data_path output_dir/gastrohun_llm_en_images_train_cot1.jsonl \
  --image_dir /media/ren9/Elements/exvla_clinical \
  --checkpoint models/grpo_video_cot1_v1 \
  --runname grpo_image_cot1_v1 \
  --mode 2 \
  --task gastrohun \
  --epochs 3 \
  --max_completion_length 1024

  4) 常见问题
4.1 conda env create 很慢/卡住
建议配置更快的 solver（可选）：

conda install -n base -c conda-forge mamba -y
mamba env create -f exendovla.yml -n exendovla_dongdongjun

4.2 torch.cuda.is_available() 是 False
通常是驱动/CUDA/torch wheel 不匹配或没有 GPU 权限，请检查：
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
