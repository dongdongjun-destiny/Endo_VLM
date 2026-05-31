# Endo_VLM

EndoVLA 多模态视觉-语言模型，面向 **GastroHUN / SSS** 任务的胃镜 **图像与视频** 理解与定位。本仓库包含 SFT、GRPO 强化学习训练、推理与 GastroHUN baseline 代码。

## 仓库结构

| 路径 | 说明 |
|------|------|
| [`exvla/`](exvla/) | EndoVLA 训练 / 推理主代码 |
| [`baseline_gasthun/`](baseline_gasthun/) | GastroHUN baseline |
| [`exendovla.yml`](exendovla.yml) | Conda 环境定义 |

### `exvla/` 主要入口

| 脚本 | 用途 |
|------|------|
| `sft_train.py` | 监督微调（SFT） |
| `rft_grpo_video_train.py` | 视频 GRPO（`GRPO_MODALITY=video`） |
| `rft_grpo_image_train.py` | 图像 GRPO（`GRPO_MODALITY=image`） |
| `rft_grpo_core.py` | GRPO 核心逻辑（patch、奖励、训练循环） |
| `inference_core.py` | 推理与评估 |
| `config.py` | 模型、数据、奖励与 prompt 配置 |

训练数据 jsonl 位于 `exvla/output_dir/`（已随仓库提供索引文件；原始影像需自行准备，见下文）。

---

## 环境配置

### 1. 创建 Conda 环境

```bash
git clone https://github.com/dongdongjun-destiny/Endo_VLM.git
cd Endo_VLM

# 推荐：用 mamba 更快
conda install -n base -c conda-forge mamba -y
mamba env create -f exendovla.yml -n exendovla

# 或使用 conda
conda env create -f exendovla.yml -n exendovla

conda activate exendovla
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 2. 临床数据路径

jsonl 中的 `image_path` / `video_path` 需指向本地影像目录。示例（按你的磁盘修改）：

```bash
export EXVLA_DATA_ROOT=/media/<user>/Elements/exvla_clinical
```

训练命令中的 `--image_dir` 即该根目录。

> **说明**：模型权重（`*.safetensors`、`*.bin`、`*.pt` 等）默认不上传 GitHub，需在本地 SFT / GRPO 训练后生成，或从团队共享存储获取。

---

## 输出格式（GastroHUN / SSS）

模型输出与评估均要求 **首行** 为严格格式：

```text
Final SSS: <LABEL>
<简短英文解释>
```

`<LABEL>` 为 23 类 SSS 标签之一（`A1`–`A6`、`G1`–`G4`、`L1`–`L6`、`P1`–`P6`、`NA`）。

GRPO 奖励机制（见 `config.py` → `REWARD_CONFIG`）：

- **格式分**：首行 `Final SSS`、解释结构等
- **准确分**：首行严格匹配 1.0；fallback 解析正确 0.5；同字母 / 相邻区域有部分奖励
- **reward_style**：视频默认 `short`，图像默认 `cot`（可用 `--reward_style` 覆盖）

---

## 训练示例

以下命令均在 `exvla/` 目录下执行：

```bash
cd exvla
conda activate exendovla
```

### SFT（多模态 CoT）

```bash
python sft_train.py \
  --data_path output_dir/gastrohun_llm_en_multimodal_train_sft_cot_10.jsonl \
  --image_dir /media/<user>/Elements/exvla_clinical \
  --runname train_sft_cot_multimodal_v1 \
  --mode 1 \
  --epochs 3
```

### GRPO — 视频

```bash
python rft_grpo_video_train.py \
  --data_path output_dir/gastrohun_llm_en_multimodal_train.jsonl \
  --image_dir /media/<user>/Elements/exvla_clinical \
  --checkpoint models/sft_cot1_image_v1 \
  --runname grpo_video_cot1_v1 \
  --mode 2 \
  --task gastrohun \
  --epochs 3 \
  --max_completion_length 1024 \
  --reward_style short
```

### GRPO — 图像

```bash
python rft_grpo_image_train.py \
  --data_path output_dir/gastrohun_llm_en_images_train_cot1.jsonl \
  --image_dir /media/<user>/Elements/exvla_clinical \
  --checkpoint models/grpo_video_cot1_v1 \
  --runname grpo_image_cot1_v1 \
  --mode 2 \
  --task gastrohun \
  --epochs 3 \
  --max_completion_length 1024 \
  --reward_style cot
```

### 常用 GRPO 参数

| 参数 | 说明 |
|------|------|
| `--mode 1` | 从基座训练；`--mode 2` 从 checkpoint 继续 |
| `--task gastrohun` | GastroHUN 奖励与解析（默认） |
| `--max_completion_length` | 生成长度；视频建议 ≥512，常用 1024 |
| `--reward_style short\|cot` | 短答 / 链式 CoT 格式奖励 |
| `--temperature` / `--top_p` / `--top_k` | GRPO rollout 采样 |
| `--no_gpu_guard` | 关闭 GPU 占用检测 |

训练日志默认写入 `wandb/`（已 gitignore，不会上传）。

---

## 本地与 GitHub 同步

若你在本机维护独立 `exvla/` 工作目录，可先镜像到本仓库再 push：

```bash
# 将本地 exvla 同步到 Endo_VLM/exvla（排除 wandb / __pycache__）
rsync -av --delete \
  --exclude 'wandb/' --exclude '__pycache__/' --exclude '*.pyc' \
  /path/to/your/exvla/ Endo_VLM/exvla/

cd Endo_VLM
git add -A exvla/ README.md
git commit -m "Sync latest exvla code"
git push origin main
```

使用 Personal Access Token 时：Username 填 GitHub 用户名，Password 填 `ghp_...` token。

---

## 常见问题

### `conda env create` 很慢

```bash
mamba env create -f exendovla.yml -n exendovla
```

### `torch.cuda.is_available()` 为 False

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

检查 NVIDIA 驱动、CUDA 与 PyTorch wheel 是否匹配。

### GRPO 显存不足

尝试：`--batch_size 1 --num_generations 2 --max_prompt_length 3072`，并关闭其他 GPU 进程。

---

## 引用与数据

- GastroHUN 数据集与 SSS 协议请参考原论文与 baseline 目录。
- 本仓库代码仅供研究与复现使用。
