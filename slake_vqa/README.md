# SLAKE 影像问答、CT 分割与局部审计

在同一批 SLAKE 问题上比较 Base 与 LoRA 的看图答复。CT 用 U-Net 提供区域，胸片可用 MAIRA-2 定位心影增大。局部审计比较原图与不同模糊图上的回答。 [复现报告](REPRODUCTION.md) · [结果索引](results/)

## 先打开网页

用 VS Code 或 PyCharm 打开整个 `MedVQA_Evidence` 仓库，把 Python 解释器选为仓库里的 `.venv\Scripts\python.exe`。打开一个新的 IDE 终端，确认终端在仓库根目录。已有环境时，直接运行：

```powershell
python -m slake_vqa demo
```

打开 [本机网页](http://127.0.0.1:8000/)。上面是知识检索模式。要做影像问答，先按 `Ctrl+C`，再运行 `python -m slake_vqa demo --model cpu`。

## 1. 安装依赖、检查数据

首次使用先看 [本机环境准备](../evidence/README.md)。然后在 IDE 终端执行：

```powershell
python -m pip install -r slake_vqa/requirements.txt
python -m slake_vqa data
```

缺少数据时运行 `python -m slake_vqa data --download`。数据在 `slake_vqa/artifacts/data/slake/`。**实测** 642 张 CT、MRI、X 光，英文训练、验证、测试题为 4,919／1,053／1,061 道。 [数据盘点](results/slake_inventory.json)

## 2. 训练 LoRA

有可用的 CUDA GPU 就能训练。已有适配器可以直接到测试。输出在 `slake_vqa/artifacts/checkpoints/slake-lora/`。

```powershell
python -m slake_vqa train
```

**实测** 1 轮、615 步，验证损失 0.1567。 [训练摘要](results/lora_training.json)

## 3. 测试四组问答

`--with-lora` 会同时评测 Base 和 LoRA。优先使用本机训练的适配器，其次使用归档权重。CUDA 环境可以把 `cpu` 改成 `cuda`。结果在 `slake_vqa/artifacts/results/benchmark/summary.json`。

```powershell
python -m slake_vqa evaluate --with-lora --split test --device cpu
```

| 已有验证结果 · 150 题 | 看图 EM | 仅文本 EM |
| --- | ---: | ---: |
| Base | 47.33% | 35.33% |
| LoRA | **74.00%** | 38.67% |

**待补**独立测试的四组汇总。要复现验证集，改用 `--split validation`，并加 `--output slake_vqa/artifacts/results/validation-150`。 [验证集汇总](results/vqa_validation.json)

## 4. 训练和测试 U-Net

三行分别训练加权左肺、不加权左肺和四器官 U-Net。逐行执行。脚本自动选 CPU 或 CUDA。

```powershell
python -m slake_vqa segment --size 96 --epochs 64 --test
python -m slake_vqa segment --size 96 --epochs 64 --test --unweighted
python -m slake_vqa segment --organs "left lung" "right lung" heart liver --size 96 --epochs 64 --test
```

**实测** 5 张测试 CT 的左肺 Dice 分别为 0.5904、0.6289，40 张测试 CT 的四器官宏 Dice 为 0.6583。 [分割结果](results/)

## 5. 做局部审计

先用项目自带的左肺题和 SLAKE 标注掩膜，输出 `slake_vqa/artifacts/results/audit_cases.json`。第 4 步训练了加权 U-Net 时，执行第二行测试预测掩膜。

```powershell
python -m slake_vqa audit run --device cpu
python -m slake_vqa audit run --device cpu --mask-source unet
```

**本机实测** 左肺单题的原图和扰动结果均答 `left lung`，目标和对照各 933 像素。 [本机审计记录](results/local_cpu_audit_smoke.json)

### 可选 · 用 MAIRA-2 做胸片审计

第一行直接运行 MAIRA-2 环境里的 `python.exe`。第二行回到 IDE 终端的 Qwen 环境。 [MAIRA-2 环境准备](../evidence/README.md#maira-2-环境)

```powershell
.\.venv-maira\Scripts\python.exe -m slake_vqa maira --split audit
python -m slake_vqa audit run --device cpu --mask-source maira
```

## 6. 在网页看 U-Net 证据

```powershell
python -m slake_vqa demo --model cpu --checkpoint slake_vqa/artifacts/checkpoints/unet.pt
```

网页截图**待补**。保存允许公开的页面截图到 `slake_vqa/media/`。

<!-- ![SLAKE 问答与局部证据](media/visual-audit.png) -->
<!-- ![知识检索与来源](media/knowledge-search.png) -->
