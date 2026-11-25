# SLAKE 复现报告与历史 GPU 操作记录

Windows 本机复现请按 [本机 README](README.md) 从环境准备开始。本文保留 GPU 实验详细设置、结果和原 AutoDL 环境的处理记录。

本文复现 SLAKE 短答案问答、CT 分割和局部审计。新版 [SLAKE 子项目入口](../slake_vqa/README.md) 汇总这些命令。若要复现胸片 Findings 生成和报告级审计，请进入 [IU X-Ray 手册](../cxr_report_evidence/README.md)。

历史 GPU 命令在 **AutoDL 的 JupyterLab 终端**运行，Windows 另有标注。已有源码、数据和 Qwen 权重时，从第 1 步检查。第一次准备依赖和权重时，使用 [本机环境说明](../evidence/README.md) 的包版本与模型来源。

这个项目分开回答三个问题。**LoRA** 测问答表现有没有变化，**U-Net** 测能不能找准待审计的区域，**CLEAR 近似审计**测能不能把缺少视觉证据的回答排到更前面。三项分别保留结果。


### 目前从哪里继续

已完成 GPU LoRA 训练及验证集评测、单类和四类 CT U-Net、MAIRA-2 单张和 57 张胸片批量定位，以及 CT 和胸片各一道反事实审计题。**现在从 4.4 的第四步开始，在网页验证这两个案例。** 要继续做第 5 节的风险评分，需要先按 4.3 整理批量题目和人工支持性标签。 [网页验证命令](#第四步在同一网页里验证两个分支) · [结果清单](#7-哪些结果要补到项目里)


## 1. 确认 GPU、数据和本地权重

新开终端时先切到项目、激活已有环境。

```bash
cd /root/autodl-tmp/MedVQA_Evidence
source /root/autodl-tmp/medvqa-runtime/bin/activate
nvidia-smi
python -c 'import torch, torchvision, peft, accelerate; from transformers import TrainingArguments; print("PyTorch", torch.__version__, "torchvision", torchvision.__version__, "accelerate", accelerate.__version__); args=TrainingArguments(output_dir="/tmp/medvqa-device-check", report_to="none"); print("Trainer device", args.device); assert args.device.type == "cuda"'
```

- `nvidia-smi` 显示 GPU 型号和显存。Python 命令另外检查当前环境能否调用 GPU，并验证 LoRA 使用的 Trainer 能否初始化。
- **通过标准**是看到 `Trainer device cuda:0` 或其他 CUDA 序号。依赖版本参照 [训练依赖](../evidence/requirements-research.txt)，PyTorch 与 torchvision 按同一套 CUDA 版本安装。 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)

继续确认之前上传的文件，模型路径用一个变量记住。

```bash
MODEL=models/Qwen3-VL-2B-Instruct
python -m slake_vqa data --root data/slake
ls -lh "$MODEL"/{config.json,tokenizer.json,model.safetensors}
```

- `MODEL=...` 只在当前终端有效。换终端后重新执行这行，其余命令用 `"$MODEL"` 加载该目录。
- SLAKE 使用 `data/slake/` 下的 `train.json`、`validation.json`、`test.json`、`imgs.zip` 和 `mask.txt`。程序直接打开 zip。`python -m slake_vqa data --root data/slake` 会把数据版本校验和、划分和样本量写到 `results/slake_inventory.json`。
- **通过标准**是数据检查显示 `train 4919 questions 450 images` 等统计，且模型配置、分词器和 `model.safetensors` 都存在。数据和权重可按 [SLAKE 数据准备](README.md#1-安装依赖检查数据) 和 [模型下载](../evidence/README.md#qwen-和网页环境) 获取。

### 1.1 SLAKE 的图像、问答和语言

当前用的是作者发布的 **SLAKE 1.0**，数据包固定在 `BoKelvin/SLAKE` 的 `a9083ce` 版本。下面的数字来自本机三份 JSON 和 `imgs.zip` 的实际盘点，文件校验和与复现实验一致。影像是二维医学图像，覆盖脑部、颈部、胸腹部和盆腔等部位。问答包括器官、位置、成像模态、异常、大小等问题，也有结合医学知识的题。原数据还提供分割掩膜和目标框。 [SLAKE 作者主页](https://www.med-vqa.com/slake/) · [作者发布的数据集](https://huggingface.co/datasets/BoKelvin/SLAKE)

| 模态 | 去重影像数 | 英文问答 | 中文问答 | 中英文问答合计 |
| --- | ---: | ---: | ---: | ---: |
| CT | 282 | 3,321 | 3,318 | 6,639 |
| MRI | 181 | 1,590 | 1,585 | 3,175 |
| X-ray | 179 | 2,122 | 2,092 | 4,214 |
| 合计 | 642 | 7,033 | 6,995 | 14,028 |

一张图对应多道题。英文问答覆盖全部 642 张影像，其中开放题 4,252 道、封闭题 2,781 道。当前 LoRA、问答评测和审计入口只读取 `q_lang=en`。U-Net 用英文问答划分中的影像 ID 来定位分割样本，再按模态与掩膜筛选。

| 数据划分 | 影像数（英文划分） | CT / MRI / X-ray 影像 | 英文问答 | 原文件中的中文问答 |
| --- | ---: | --- | ---: | ---: |
| train | 450 | 200 / 128 / 122 | 4,919 | 4,916 |
| validation | 96 | 42 / 27 / 27 | 1,053 | 1,046 |
| test | 96 | 40 / 26 / 30 | 1,061 | 1,033 |

**使用双语时要按影像重新划分。** 英文和中文各自的训练、验证、测试影像 ID 两两无交集，但是有些影像的英文题和中文题在不同文件。把两个语言直接合并后，训练文件与测试文件会共享 142 个影像 ID。当前英文实验的三个划分，影像 ID 和像素哈希都没有交集。 [英文数据盘点](results/slake_inventory.json) · [读取和筛选逻辑](dataset.py)


## 2. Qwen 基线与 LoRA

### 2.1 先训练两步并重新加载

```bash
python -m slake_vqa train --root data/slake --model "$MODEL" \
  --max-train 32 --max-val 8 --steps 2 \
  --output checkpoints/gpu/lora-smoke
ls -lh checkpoints/gpu/lora-smoke/adapter/adapter_model.safetensors
cat checkpoints/gpu/lora-smoke/training_summary.json
```

| 参数 | 在这一步发生了什么 |
| --- | --- |
| `--model "$MODEL"` | 从本地读入 Qwen3-VL-2B-Instruct 的参数和处理器 |
| `--max-train 32 --max-val 8` | 最多取 32 道训练题和 8 道验证题。训练题更新参数，验证题计算损失 |
| `--steps 2` | 更新参数两次。脚本每次处理一道题，默认每累积 8 个小批次的梯度再更新一次 |
| `--output` | 保存训练日志与 LoRA 适配器。重跑用一个新目录，并同步调整下方路径 |

LoRA 冻结原模型的大部分参数，把更新保存在一组小矩阵中。这组权重叫**适配器**，推理时要加载原 Qwen 权重，再叠加这组权重。你之前看到的 `trainable%: 0.1507` 表示约 0.15% 的参数参与训练。训练过程读取影像、问题和参考答案，代码会屏蔽提示部分的标签，让损失主要来自参考答案的生成。 [训练入口](../slake_vqa/lora.py)

用一题验证保存的适配器可以重新加载。这个命令会在同一题上依次跑 Base 和 LoRA。

```bash
python -m slake_vqa evaluate --root data/slake --model "$MODEL" --device cuda \
  --adapter checkpoints/gpu/lora-smoke/adapter \
  --split validation --limit 1 --output results/gpu/lora-smoke
cat results/gpu/lora-smoke/summary.json
```

**通过标准**是报告有有限的 `train_loss` 和 `eval_loss`、适配器文件存在、评测报告同时出现 `base` 和 `lora` 两组。两步和一题用来检查训练和推理链路，通过后再训练正式适配器。

### 2.2 正式训练一轮，再做配对评测

```bash
python -m slake_vqa train --root data/slake --model "$MODEL" --epochs 1 --max-val 120 \
  --output checkpoints/gpu/slake-lora-v1

python -m slake_vqa evaluate --root data/slake --model "$MODEL" --device cuda \
  --adapter checkpoints/gpu/slake-lora-v1/adapter \
  --split validation --limit 150 --output results/gpu/vqa-val-150
```

`--epochs 1` 让程序遍历一次训练集。省略 `--max-train` 时使用 4,919 道训练题，`--max-val 120` 限定计算验证损失的题数。下方 benchmark 的 `--limit 150` 则独立控制生成式评测题数。benchmark 按固定种子 42 选题，Base 与 LoRA 共用这些题、提示词和生成设置。如果评测中断，同一条命令和同一输出路径可从已有逐题 JSON 继续。LoRA 训练重跑则使用新输出目录。

验证集调通、选定适配器和生成设置以后，在同一配置下用独立测试集跑一次。这里先固定 150 道测试题。若决定报告整套英文测试集，使用 `--limit 1061` 并换一个输出目录，在结果表明确题数。

```bash
python -m slake_vqa evaluate --root data/slake --model "$MODEL" --device cuda \
  --adapter checkpoints/gpu/slake-lora-v1/adapter \
  --split test --limit 150 --output results/gpu/vqa-test-150
```

打开 `summary.json`，先比较同一输入条件下的 Base 与 LoRA，再比较同一个模型在 `image`（原图＋问题）与 `text_only`（仅问题）两种输入下的分数。

- **EM** 表示按代码归一化后的整条答案是否完全匹配。`lora_minus_base_em_pp` 已换算成百分点。例如 0.40 到 0.46 是提高 6 个百分点。
- **Token-F1** 衡量预测与参考答案的词元重合。答案只答对一部分时，EM 可能是 0，Token-F1 仍可大于 0。
- **图像增益**可以手算为 `EM(image) − EM(text_only)`。如果 LoRA 的两种输入都涨，仍需用后面的局部审计判断它有没有更依赖目标区域。
- `run.json` 记录题号、数据和适配器校验和。`base_cases.json` 和 `lora_cases.json` 能用来找具体例子。可选错图实验用 `--wrong-images 路径`，CSV 要有 `qid,wrong_image_id,reviewed` 三列，人工确认替换图无法支持原答案后将 `reviewed` 填为 `yes`。错图命令会只评清单里的题，和主结果分别记录。

### 2.3 已跑出的 GPU 结果

**运行环境与训练。** 本次实例是 NVIDIA GeForce RTX 4090 D（`nvidia-smi` 显示总显存 24,564 MiB）。本地保存的两份适配器都包含 3,211,264 个 LoRA 参数。正式配置为 `r=8`、`alpha=16`、`dropout=0.05`、学习率 `1e-4`，适配器作用在语言部分自注意力的 `q_proj`、`k_proj`、`v_proj`、`o_proj`。 GPU 型号见本机原终端记录。

| 训练 | 训练 / 验证题数 | 更新步数 | 训练用时 | 平均训练损失 | 验证损失 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 两步冒烟 | 32 / 8 | 2 | 4.85 秒 | 1.5200 | 0.5066 |
| 正式 LoRA `slake-lora-v1` | 4,919 / 120 | 615 | 21 分 13 秒 | 0.3198 | 0.1567 |

训练用时是报告中的 `train_runtime`，另有正式验证用时 9.01 秒。正式训练日志中，更新第 10 步的区间损失为 1.0441，第 610 步为 0.2683，说明模型在这轮训练中逐渐学会了生成参考答案。`eval_loss` 衡量已知参考答案时模型对答案词元的预测，下面的 EM 衡量模型自由生成能否答对。两次训练用的题量不同，判断微调收益看同一批题上的配对评测。

**两步链路已通过。** 适配器已保存，GPU benchmark 成功加载它。验证题 `9840` 问影像是否包含肝脏，参考答案为 `Yes`，Base 与 LoRA 在原图和仅文本下都答 `Yes`。这题确认两条推理路径都能运行，单题结果用于验收链路。

**正式配对评测。** 官方 SLAKE 英文 `validation` 固定抽取 150 题、73 张影像，种子 42，生成上限 48 token。重新按本项目的计算规则核对了逐题预测、题号、影像像素哈希、提示词和适配器校验和。下面的完全匹配计数按题统计。`pp` 表示百分点。

| 输入条件 | Base EM（题数） | LoRA EM（题数） | EM 变化 | Base → LoRA Token-F1 | Token-F1 变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原图＋问题 | 47.33%（71/150） | 74.00%（111/150） | +26.67 pp | 50.72% → 78.77% | +28.05 pp |
| 仅问题 | 35.33%（53/150） | 38.67%（58/150） | +3.33 pp | 38.07% → 43.10% | +5.03 pp |

- **原图表现提升主要来自改对的题更多。** 原图下，Base→LoRA 有 44 题从错变对、4 题从对变错，净增 40 题。仅文本下有 35 题从错变对、30 题从对变错，净增 5 题。
- **同一个模型看图能多答对多少题？** Base 的 26 题因加图从错变对，8 题从对变错，EM 图像增益为 `+12.00 pp`。LoRA 对应为 61 题和 8 题，增益为 `+35.33 pp`。两种模型的图像增益相差 `23.33 pp`。这表明这一批题上 LoRA 从整张影像输入获得了更大的收益。区域层面的证据需要用之后的局部扰动审计检验。
- **同一张图有多题。** 按 73 张影像成组抽样，每次抽到一张图就带上它的所有题，做 10,000 次配对 bootstrap（种子 42）。EM 差值的 95% 区间为原图 `[+18.49, +34.90] pp`、仅文本 `[−7.04, +13.61] pp`。图像增益差的区间为 `[+13.19, +33.12] pp`。这些区间描述本批验证样本的波动，独立 test 结果仍待评测。
- **模态拆分用于检查提升来自哪里。** 原图下，CT 为 35/70 → 54/70，MRI 为 15/36 → 25/36，X-RAY 为 21/44 → 32/44。这些分组样本数较小，按描述统计报告。

| 题号 | 参考答案 | Base 原图 / 仅文本 | LoRA 原图 / 仅文本 | 怎么解读 |
| --- | --- | --- | --- | --- |
| 10260（胸片肺部外观） | `No` | `Yes` / `Yes` | `No` / `Yes` | LoRA 在原图条件下答对，可作为后续检查区域证据的候选案例 |
| 9984（CT 扫描平面） | `Transverse Plane` | `axial` / `Axial` | `Transverse Plane` / `Transverse Plane` | `axial` 和 `transverse` 可指同一扫描平面，EM 会把 Base 判错。只看 EM 会把部分答题术语对齐也算作提升 |
| 10449（CT 下颌骨计数） | `1` | `1` / `2` | `2` / `2` | 原图下 Base 答对、LoRA 答错，保留作微调退化案例 |

本次归档中的冒烟和正式训练汇总、验证配对汇总均与此前保存的文件一致。训练损失读 `train_metrics.train_loss` 和 `validation_metrics.eval_loss`，配对分数读 `metrics.base` 和 `metrics.lora` 下的 `image`、`text_only`。**待补**固定现有适配器后运行第 2.2 节的独立 `test` 评测，填入题数、影像数、四组 EM / Token-F1、图像增益和对应的案例。 冒烟训练汇总可在本机查看 `artifacts/gpu-run/checkpoints/gpu/lora-smoke/training_summary.json` · [正式训练汇总](results/lora_training.json) · [验证配对汇总](results/vqa_validation.json)

## 3. U-Net 分割与加权消融

U-Net 输入影像，输出每个像素属于目标器官的概率。当前实验只使用 SLAKE 中有有效标注的 **CT 左肺**切片。它输出的掩膜会用于第 4 步。选择一次固定配置，先训练加权版，再将完全相同的配置跑一遍，仅移除 BCE 的正类权重。

```bash
python -m slake_vqa segment --root data/slake --organ "left lung" --modality CT --size 96 --epochs 64 \
  --checkpoint checkpoints/gpu/unet_weighted.pt \
  --result results/gpu/unet_weighted.json --test

python -m slake_vqa segment --root data/slake --organ "left lung" --modality CT --size 96 --epochs 64 \
  --checkpoint checkpoints/gpu/unet_unweighted.pt \
  --result results/gpu/unet_unweighted.json --test --unweighted
```

`--size 96` 把影像和掩膜调整为 96×96，`--epochs 64` 是 64 轮。分割脚本自动选用 CUDA，查结果中的 `device` 是否为 `cuda`。两条命令都用验证集 Dice 选择最佳轮次，`--test` 让程序随后用该权重测一次留出的测试集。如果正在试超参数，先去掉 `--test`，配置冻结后再运行上面两条。

**`--unweighted` 为什么叫消融？** 消融就是去掉一个组成部分，其他条件对齐，然后看结果改变多少。分割训练的损失是 `BCE + (1 − Soft Dice)`。BCE 逐像素惩罚分错，Soft Dice 鼓励整块预测区域与标注重叠。左肺像素少、背景像素多，默认版会给左肺像素更高的 BCE 权重。项目根据训练集前景比例 `p` 计算 `min(20, sqrt((1−p)/p))`。例如 `p=2.56%` 时权重约为 6.17。加 `--unweighted` 会把这项权重设为 1，两次都保留 Soft Dice，同样使用 U-Net 和相同的数据划分。这个实验回答「给稀少的器官像素加权是否提高分割重叠」。 [分割实现](segmentation/unet.py)

| 从 JSON 找什么 | 怎么读 |
| --- | --- |
| `bce_positive_weight` | 核对加权版的实际权重、消融版是否为 1 |
| `best_epoch` | 验证集 Dice 最高对应的训练轮次 |
| `validation.dice`、`test.dice` | 预测与标注的重叠分数，1 表示完全重合 |
| `validation.iou`、`test.iou` | 预测与真实区域的交集除以并集，1 表示完全重合 |
| `train_images`、`validation.images`、`test.images` | 记录样本量，多个问题可以共享同一影像 |

### 3.1 这次为什么跑得快

完整数据集有 282 张 CT。沿用项目的英文影像划分，再只保留含**有效且非空左肺标签**的 CT 影像，训练从 200 张 CT 缩到 25 张，验证从 42 张缩到 5 张，测试从 40 张缩到 5 张。同一张 CT 对应的多个问答只计为一张分割训练影像。没有左肺阳性标注的切片未参与这一轮。

- 批大小为 8。25 张图拆成 8、8、8、1，每轮只做 4 次参数更新。64 轮共 256 次更新，训练样本累计送入模型 25×64＝1,600 次。每轮验证也只有 5 张图。
- 输入在数据准备时读入内存并缩为 `96×96` 的二维灰度图，网络只有 117,985 个参数。这批小图用 CUDA 运行，工作量很小。
- 两份报告都记录了 64 轮并保存了权重。终端日志和 JSON 没有秒数，准确耗时需要从以后带计时的运行取得。

### 3.2 已跑出的 GPU 分割结果

两组都是 `cuda`、96×96、64 轮，训练／验证／测试为 **25／5／5 张**。训练像素中，左肺前景平均仅占 **2.5564%**。我用本机数据重新加载两份权重，验证与测试数值均与 GPU 汇总一致。

| BCE 设置 | 前景权重 | 验证 Dice 最佳轮次 | 验证 Dice | 验证 IoU | 测试 Dice | 测试 IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 默认加权 | 6.1739 | 64 | 0.5472 | 0.5031 | 0.5904 | 0.4994 |
| `--unweighted` | 1.0000 | 63 | 0.5580 | 0.5215 | 0.6289 | 0.5428 |
| 不加权减加权 | — | — | +0.0107 | +0.0184 | +0.0385 | +0.0435 |

加权组从第 31 轮起首次测到非零验证 Dice，不加权组在第 36 轮。前者更早出现与标注重叠的预测，后者最终选中的权重在本次验证和测试的两项重叠指标都较高。两组在同一张测试图 `xmlab543` 的 Dice 均为 0，且每个划分只有 5 张，结果适合判断这次小样本实验的表现。选择权重时看验证 Dice，所以后续左肺审计示例使用 `checkpoints/gpu/unet_unweighted.pt`，离线打分和网页用同一份权重。

本次归档中的两份报告与此前的 GPU 归档一致。前景权重读 `bce_positive_weight`，各划分的 Dice / IoU 读 `validation` 和 `test`。 [加权报告](results/unet_weighted.json) · [不加权报告](results/unet_unweighted.json)

### 3.3 左肺分割预实验与后续定位对照

本节的加权消融定位为**左肺小样本预实验**。重新加载两份权重复核 5 张测试 CT，不加权组有 4 张掩膜 Dice 更高、1 张持平。平均 Dice 从 0.5904 到 0.6289，验证 Dice 也由 0.5472 到 0.5580。这支持本批左肺审计选用不加权版。若要检验该选择在其他数据上是否稳定，需要增加影像和器官，并在相同划分上重复多个训练种子。

**分割和框定位分别评估。** 将预测掩膜和人工掩膜都取最小外接矩形，在 `96×96` 网格上计算五张测试 CT 的平均框 IoU，加权版为 0.4353，不加权版为 0.4874。以 `xmlab548` 为例，不加权版掩膜 Dice 由 0.4138 到 0.5067，框 IoU 却由 0.3180 到 0.2614。平均掩膜质量提高时，框的边界仍可能在某些图上偏得更多。这组框指标是同一批 CT 上两份现有 U-Net 权重的复核。

U-Net 提供**指定器官的像素区域**，适合 CT 左肺的区域扰动对照。要扩大覆盖，可以分别训练有标签的「模态＋器官」组合，或为同一模态训练多类别分割网络；每类都需要匹配的标注与评测。MAIRA-2 根据胸片和英文陈述给出**候选矩形框**，适合按陈述选择要模糊的区域。两者可以作为同一审计流程的不同区域来源。在同一模态且标注匹配时，还可以把器官轮廓和陈述框一起查看。 [MAIRA-2 的输入、输出与适用范围](https://huggingface.co/microsoft/maira-2#uses)

| 想验证什么 | 可比的实验设计 | 指标 |
| --- | --- | --- |
| 现有 CT 掩膜有没有帮助审计 | 同一 CT、同一题、同一个 Qwen 及生成参数，对比人工掩膜、U-Net 预测掩膜和同面积异位置扰动。人工掩膜作为已知目标轮廓的参照 | 目标区域相对对照的回答变化；取得视觉支持标签后补 AP、AUROC 和评分覆盖率 |
| MAIRA-2 找胸片陈述的框准不准 | 固定同一批正位胸片、同一目标病灶和标准框。要纳入 U-Net，就训练覆盖**同一胸片病灶**的 U-Net，再把预测掩膜转成外接框；只有框标注时，可以用框监督的检测基线，或把框栅格化为弱监督区域训练 U-Net 并明确训练标签来源 | 平均框 IoU、IoU≥0.5 的命中率、无框比例。对于有标准框的样本，无预测框的 IoU 记 0，按原图坐标比较 |
| 换一个定位器是否让风险分数更好 | 固定胸片、问答模型、原始回答、样本标签和干预设置，仅替换区域来源。对每个目标区域配同面积对照，记录框面积并在相同可评题上计算指标 | AP、AUROC、评分覆盖率，并记录定位错误或区域缺失的案例 |

本地 SLAKE 胸片按英文划分为 122／27／30 张，`detection.json` 分别有 132／29／31 个异常框。它们可供筛选胸片正位及目标一致的定位样本。179 张胸片的 `mask.png` 都未通过项目现有的灰度标签校验，因此现成 U-Net 训练入口没有可直接用于这批胸片的像素标签。MAIRA-2 已在验证和测试胸片上完成心影增大定位，结果见 4.4；区域来源对风险分数的影响仍需人工标注和后续评测。该权重需在 Hugging Face 接受访问条件，官方示例的 Transformers 版本范围与当前 Qwen 环境不同，推理可放在独立环境。 [SLAKE 标注说明](https://www.med-vqa.com/slake/) · [MAIRA-2 下载与环境](https://huggingface.co/microsoft/maira-2#how-to-get-started-with-the-model)


## 4. 局部反事实审计

### 4.1 当前项目与论文 CLEAR 的对应关系

论文写的是 **CLEAR**（Counterfactual Localized Evaluation of Alignment and Response）。它是一套在视觉语言模型生成之后执行的审计方法。论文用 **MAIRA-2** 提议每条临床陈述对应的区域，再用高斯模糊干预；结合回答变化、图文相似度变化和分位数特征，输出短语级风险。 [论文 Fig. 3 与方法 III-B 到 III-D](https://arxiv.org/html/2610.38389v1#S3) · [论文实现细节 IV-D](https://arxiv.org/html/2610.38389v1#S4.SS4)

| 环节 | 论文 CLEAR | MedVQA-Evidence 当前实现 |
| --- | --- | --- |
| 审计单位 | 从自由生成答案拆出的单条临床陈述 | 一整条 SLAKE 短答案 |
| 定位 | MAIRA-2 根据陈述给出候选框 | CT 可用标注掩膜或 U-Net 器官掩膜；胸片心影增大可读取 MAIRA-2 报告中的候选框 |
| 图像干预 | 在候选区域做模糊，主实验半径 20 | 全图、外接框、器官掩膜、同面积位置对照，默认半径 12 |
| 审计特征 | 临床语义变化与医学图文对齐等 | 词元 F1、生成 token 置信度、同面积差值；可选 BiomedCLIP 对齐变化 |
| 风险估计 | 原始与分位数特征送入轻量风险估计器 | 同样可拼接分位数，验证集人工标签训练标准化＋逻辑回归 |

因此目前可以表述为 **在 SLAKE 上实现 CLEAR 思路的回答级审计与风险评分**。胸片分支接入了心影增大的 MAIRA-2 候选框，原论文还对自由文本的每条陈述独立定位和评分。论文评测覆盖 VQA-RAD、MIMIC-Diff-VQA 等；本手册记录 SLAKE 的独立实验结果。 [论文数据与评测 IV-A–IV-D](https://arxiv.org/html/2610.38389v1#S4)

### 4.2 先审计一题，看懂每个变化

先用同一张图、同一个问题、同一个 Base Qwen，分别用标注掩膜与 U-Net 掩膜。

```bash
python -m slake_vqa audit run --root data/slake --cases slake_vqa/examples/audit.csv \
  --model "$MODEL" --device cuda --mask-source annotation \
  --output results/gpu/audit-example-annotation.json

python -m slake_vqa audit run --root data/slake --cases slake_vqa/examples/audit.csv \
  --model "$MODEL" --device cuda --mask-source unet \
  --checkpoint checkpoints/gpu/unet_unweighted.pt \
  --output results/gpu/audit-example-unet.json
```

`--cases` 清单指定验证题 10733 和目标器官。`run` 对原图提问一次，再对每种模糊后的图提同一个问题，输出 JSON 和供人工标注的同名 CSV。已有结果时，用新 `--output` 名字。

- `global` 模糊整张图，检查整体影像信息变化后回答怎么变。
- `box` 模糊器官外接矩形，矩形通常也包含邻近组织。`mask` 只在掩膜对应的像素位置使用模糊图。
- `control` 将同一掩膜形状平移到一个不与目标重叠的位置；面积、形状和模糊半径相同。对照位置仍可能包含相关解剖结构，所以仍需看具体样例。
- `persistence` 是**扰动后的答复与原答复**的 Token-F1。例子中假设 `mask=0.2`、`control=0.9`，则 `target_specific_change=0.7`，说明在这次测试中，模糊目标引起的变化更大。若两边都为 1，结合问题、掩膜和扰动强度再解释。
- 顶层 `confidence` 是生成答案 token 概率的几何平均。`target_specific_change` 是扰动差值。第 5 步的 `risk_score` 则来自这些特征与人工标注拟合出的模型。

**通过标准**是能读出原答案、掩膜扰动的答案和同面积对照的答案。若掩膜为空或找不到对照，缺失信号保留为空，照实记录。

### 4.3 导候选题、看图、标注原答案

```bash
python -m slake_vqa audit candidates --root data/slake \
  --organ "left lung" --modality CT --query-term lung \
  --output results/gpu/ct_lung_candidates.csv
```

1. 打开候选表，看 `qid`、`image_id`、`question`、`reference_answer`。筛选目标区域与问题确实相关、能在该影像上核查的题。关键词 `lung` 会找出候选，可能也包含右肺或其他需要更广范围的问题。
2. 把保留的题写入 `data/audit_cases.csv`，表头为 `split,qid,organ`。保留候选题的官方 `validation` / `test` 划分，同一影像可以有多题，只能出现在一个划分。下方是一行可用的格式示例，正式评分需要补充样本。

   ```csv
   split,qid,organ
   validation,10733,left lung
   ```

3. 统一用同一个问答模型和分割模型跑全批。下面选择 Base；为了比较 LoRA，另跑一批并加入 `--adapter checkpoints/gpu/slake-lora-v1/adapter`，换新输出和标注表。模型原答可能改变，需要分别核对支持性。

   ```bash
   python -m slake_vqa audit run --root data/slake --cases data/audit_cases.csv \
     --model "$MODEL" --device cuda --mask-source unet \
     --checkpoint checkpoints/gpu/unet_unweighted.pt \
     --output results/gpu/audit-base-raw.json
   cp results/gpu/audit-base-raw.csv results/gpu/audit-base-reviewed.csv
   ```

4. 对照**原图、问题、原图上的模型回答**填写副本的 `unsupported` 和 `notes`。可在 JupyterLab 用文本编辑器打开 CSV，或下载到本机编辑，保存为 UTF-8 逗号分隔的 CSV 后再上传。`0` 表示回答有明确图像支持，`1` 表示图上支持不足或与所见矛盾，无法判断则留空，写下原因。标注时先隐藏扰动特征列；参考答案可帮助理解题意，按影像里的具体证据判断。保留原始 CSV。
5. 检查两组是否都包含特征齐全的 `0` 和 `1`。如果某个划分不满足条件，继续标注覆盖同样范围的样本。左肺样本不足时先保留定性展示；扩大到其他器官时，每个器官要有匹配的分割模型，按影像隔离划分并分别记录。

**待填**验证和测试的影像数、每组 `0` / `1` / 留空的数量、缺失掩膜或对照数，以及至少一个具体案例。

### 左肺样本不够时，换 CT 肝脏做较大的审计批次

我用同一份 SLAKE 数据实际检查了筛选规模。`lung` 查到验证 3 题 / 3 张图、测试 5 题 / 3 张图。`liver` 查到验证 45 题 / 14 张图、测试 61 题 / 18 张图。这些是待核查的候选数；人审、有效掩膜和对照筛选后再统计最终样本量。左肺适合学会流程，出现两类标签不足时，可按下面的步骤换到肝脏，沿用同一审计方法。

```bash
python -m slake_vqa segment --root data/slake --organ liver --modality CT --size 96 --epochs 64 \
  --checkpoint checkpoints/gpu/unet_liver.pt \
  --result results/gpu/unet_liver.json

python -m slake_vqa audit candidates --root data/slake --organ liver --modality CT --query-term liver \
  --output results/gpu/ct_liver_candidates.csv
```

按 4.3 节挑题，另建 `data/audit_cases_liver.csv`，第三列统一填 `liver`。批量运行和网页用 `checkpoints/gpu/unet_liver.pt`；批量 JSON、审核表和第 5 步的三个评分产物另取带 `liver` 的文件名。网页「器官」填 `liver`，再用肝脏批次导出的估计器。若仍缺某类，保留定性结果，等待更多经过人工审核的样本。


### 4.4 多类 CT 与 MAIRA-2 胸片审计

两个区域来源共用 `python -m slake_vqa audit run` 和网页。CT 使用多类 U-Net 按指定器官返回像素掩膜。胸片使用 MAIRA-2 自己先生成带框报告，由规则提取明确的心影增大陈述和对应候选框。两条路线都让同一个 Qwen 在原图和扰动图上回答同一问题。心影增大是否在报告中出现、是否成功定位、以及最终的视觉证据风险是三个字段。

#### 第一步：同步代码，再在 AutoDL 训练四类 CT U-Net

把当前版本的 `slake_vqa/` 和 `evidence/` 一同同步到 AutoDL 项目根目录。打包排除本机的模型、原数据和实验产物。 [同步代码命令](../cxr_report_evidence/REPRODUCTION.md#1-把代码从-windows-同步到-autodl)

接下来换到 **AutoDL 的 JupyterLab 终端**。激活原来的 Qwen 环境，再确认能看到 GPU 和新版训练参数：

```bash
cd /root/autodl-tmp/MedVQA_Evidence
source /root/autodl-tmp/medvqa-runtime/bin/activate
python -c 'import torch; print("CUDA available:", torch.cuda.is_available()); assert torch.cuda.is_available()'
python -m slake_vqa segment --root data/slake --help
```

看到 `CUDA available: True`，且帮助里有 `--organs` 后，就用下面的命令开始训练。如果帮助里只有单数的 `--organ`，回到前面的上传步骤，检查是否同步到了正在使用的项目目录。

```bash
python -m slake_vqa segment --root data/slake --modality CT \
  --organs "left lung" "right lung" heart liver \
  --size 96 --epochs 64 --test \
  --checkpoint checkpoints/gpu/unet_ct_multi.pt \
  --result results/gpu/unet_ct_multi_summary.json
```

模型输出背景加四个器官，共五个通道。训练用按像素频率加权的交叉熵和前景宏平均 Soft Dice。加 `--unweighted` 会把交叉熵的类别权重设为 1。所有有效 CT 切片都参与本次任务，未选标注编码按背景处理。结果表的器官 Dice / IoU 只统计该器官有标注的影像。`absent_label_images_with_prediction` 单独记录无该标签但模型画出区域的图数，便于排查误分或标注覆盖问题。最佳轮次由验证宏平均 Dice 决定。

**通过标准**是两份产物都存在，`results/gpu/unet_ct_multi_summary.json` 的 `device` 为 `cuda`，并且 `validation.per_organ` 和 `test.per_organ` 都列出了四个器官。先记下最佳轮次、四类 Dice／IoU、各类有标注的测试图数和宏平均，再做第二步。

```bash
ls -lh checkpoints/gpu/unet_ct_multi.pt results/gpu/unet_ct_multi_summary.json
python -c 'import json; from pathlib import Path; r=json.loads(Path("results/gpu/unet_ct_multi_summary.json").read_text()); print("device:", r["device"], "best_epoch:", r["best_epoch"]); print(json.dumps({k:r[k] for k in ("validation","test")}, indent=2))'
```

**AutoDL GPU 实测。** 输入分辨率 `96×96`，训练 64 轮，按验证宏 Dice 选中第 **63** 轮。按训练入口的掩膜筛选规则从本地 SLAKE 重数，总图数为训练 **200**、验证 **42**、测试 **40**。各器官的有标注图数读 `positive_images`，同一图可包含多个器官。验证和测试总数也可直接读 `validation.images`、`test.images`。 [GPU 分割报告](results/unet_ct_multi_summary.json) · [筛选与指标实现](segmentation/multiclass.py)

| 器官 | 有标注图数（训练／验证／测试） | 验证 Dice | 验证 IoU | 测试 Dice | 测试 IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 左肺 | 25／5／5 | 0.6889 | 0.6013 | 0.5670 | 0.4788 |
| 右肺 | 27／5／5 | 0.7932 | 0.6989 | 0.7611 | 0.6453 |
| 心脏 | 17／3／3 | 0.8485 | 0.7378 | 0.5095 | 0.4139 |
| 肝脏 | 85／15／18 | 0.6608 | 0.5556 | 0.7958 | 0.6854 |
| 器官宏平均 | — | 0.7478 | 0.6484 | 0.6583 | 0.5558 |

逐器官分数读 `validation.per_organ` 和 `test.per_organ`，宏平均对四个器官等权。`history` 显示第 63 轮验证宏 Dice 为 0.7478，第 64 轮为 0.6216，因此审计加载第 63 轮保存的权重。肝脏的测试 Dice 在四类中最高，含标注的测试图有 18 张。心脏从验证 0.8485 到测试 0.5095，两组各只有 3 张有标注图。肝脏在缺该标签的 22 张测试图上仍有 13 张出现预测，字段为 `test.per_organ.liver.absent_label_images_with_prediction`，这些图需要结合影像与标注覆盖情况具体核查。 [GPU 分割报告](results/unet_ct_multi_summary.json)

此前的本机 CPU 开发验证选中第 62 轮，其测试宏 Dice／IoU 为 0.7246／0.6125。另一次 48 轮不加权试跑的测试宏 Dice 为 0.4442，左肺 Dice 为 0；训练轮数不同，单独评估加权需要同预算对照。 CPU 开发验证报告可在本机查看 `artifacts/results/unet_ct_multi_summary.json`

#### 第二步：用新 GPU 权重验证一道 CT 肝脏审计题

选验证题 `9962`（`xmlab207`）。在同一个 AutoDL 主环境运行：

```bash
mkdir -p results/gpu
printf 'split,qid,organ\nvalidation,9962,liver\n' > results/gpu/ct_multi_example_cases.csv

python -m slake_vqa audit run --root data/slake --cases results/gpu/ct_multi_example_cases.csv \
  --model models/Qwen3-VL-2B-Instruct --device cuda --mask-source unet \
  --checkpoint checkpoints/gpu/unet_ct_multi.pt --output results/gpu/ct_multi_liver_audit.json
```

审计和网页都用第一步选出的 GPU 权重。运行后打开 JSON，查看 `region`、原答案、`variants` 中的全图/外接框/掩膜/对照答复。掩膜为空时，记录这次缺少局部信号并核对掩膜。这一步通过后，再做 MAIRA-2。

**实测题 9962。** CT `xmlab207` 问图像左侧的器官是什么，SLAKE 参考答案为 `Liver`，Base Qwen 原答 `kidney`，生成置信度为 **0.8040**。从 `cases[0].variants` 依次读到下表。 CT 审计 JSON可在本机查看 `artifacts/gpu-run/results/gpu/ct_multi_liver_audit.json`

| 输入变化 | 回答 | 与原答的 Token-F1（`persistence`） |
| --- | --- | ---: |
| 全图模糊 `global` | `liver` | 0.0000 |
| 外接框模糊 `box` | `kidney` | 1.0000 |
| 肝脏掩膜模糊 `mask` | `left kidney` | 0.6667 |
| 同形状位置对照 `control` | `kidney` | 1.0000 |

`region` 给出外接框 `[70, 162, 304, 348]`，目标和对照各 10,215 像素，`control_match=same_shape`。`target_specific_change = 1.0000 − 0.6667 = 0.3333`，来自加上 `left` 一词，局部扰动后仍答肾脏。全图模糊后恰好答到参考答案，这是一题的观测。`region_file.sha256` 与归档中的 `unet_ct_multi.pt` 一致；同名 CSV 的 `unsupported` 仍需人工填写。 CT 审计 JSON可在本机查看 `artifacts/gpu-run/results/gpu/ct_multi_liver_audit.json` · CT 审计 CSV可在本机查看 `artifacts/gpu-run/results/gpu/ct_multi_liver_audit.csv`

#### 第三步：单独运行 MAIRA-2 的心影增大识别

##### 3.1 确认 MAIRA 环境

在 AutoDL 终端中检查已有的独立环境。确认 CUDA 可用、Transformers 版本为 `4.51.3`，并且 SentencePiece 的 Protobuf 绑定能导入，再做 3.2。主项目的 Qwen 仍使用 `medvqa-runtime`。

```bash
cd /root/autodl-tmp/MedVQA_Evidence
/root/autodl-tmp/maira-runtime/bin/python -c 'import torch, transformers; print(torch.cuda.is_available(), transformers.__version__)'
/root/autodl-tmp/maira-runtime/bin/python -c 'from google import protobuf; import sentencepiece; from sentencepiece import sentencepiece_model_pb2; print("SentencePiece OK:", sentencepiece.__version__, "Protobuf:", protobuf.__version__)'
```

仅第一次创建环境，或还没装依赖时，执行以下命令。之后再做上面的检查。

```bash
cd /root/autodl-tmp/MedVQA_Evidence
/root/miniconda3/bin/python -m venv --system-site-packages /root/autodl-tmp/maira-runtime
/root/autodl-tmp/maira-runtime/bin/python -m pip install -r evidence/requirements-maira.txt
```

##### 3.2 在 AutoDL 把完整模型下载到 `models/maira-2`

**先给 AutoDL 终端保存 Hugging Face 下载令牌。** 在浏览器用你的 Hugging Face 账号接受 MAIRA-2 的访问条件，然后在同一账号下创建一个 **Read** 令牌。浏览器登录只让你在网页操作，AutoDL 上的下载程序通过令牌访问模型。 [MAIRA-2 模型页](https://huggingface.co/microsoft/maira-2) · [创建访问令牌](https://huggingface.co/settings/tokens)

在 **AutoDL 的 JupyterLab 终端**运行：

```bash
/root/autodl-tmp/maira-runtime/bin/python -c 'from huggingface_hub import login; login()'
```

看到 `Enter your token` 后粘贴以 `hf_` 开头的令牌并按回车。输入内容不会显示。如果问是否保存为 Git 凭据，输入 `n`。登录成功后，下面的下载命令会读取 AutoDL 上保存的令牌。 [Hugging Face 令牌说明](https://huggingface.co/docs/hub/security-tokens)

接着复制下面的整块命令。它在 AutoDL 上关闭 `hf-xet` 客户端、改用 HTTP，一次下载一个文件，适用于小配置已下载、大权重一直停在 `0.00 B` 的情况。模型保存到小写路径 `/root/autodl-tmp/MedVQA_Evidence/models/maira-2`。六个权重分片合计约 27.5 GB，可先用 `df -h /root/autodl-tmp` 查看数据盘空间。 [Hugging Face 下载环境变量](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhubdisablexet) · [官方权重文件列表](https://huggingface.co/microsoft/maira-2/tree/main) · [AutoDL 学术加速](https://api.autodl.com/docs/network_turbo/)

```bash
(
  cd /root/autodl-tmp/MedVQA_Evidence || exit 1
  if [ -r /etc/network_turbo ]; then source /etc/network_turbo; fi
  unset HF_ENDPOINT HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  HF_HUB_DISABLE_XET=1 HF_HUB_ETAG_TIMEOUT=20 HF_HUB_DOWNLOAD_TIMEOUT=40 \
  /root/autodl-tmp/maira-runtime/bin/python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

folder = Path(snapshot_download("microsoft/maira-2", local_dir="models/maira-2", token=True, max_workers=1))
required = ("config.json processor_config.json preprocessor_config.json chat_template.json "
            "tokenizer.json tokenizer_config.json tokenizer.model added_tokens.json special_tokens_map.json "
            "configuration_maira2.py processing_maira2.py modeling_maira2.py generation_config.json "
            "model.safetensors.index.json").split()
required += [f"model-{i:05d}-of-00006.safetensors" for i in range(1, 7)]
missing = [name for name in required if not (folder / name).is_file() or (folder / name).stat().st_size == 0]
if missing:
    raise SystemExit("Download incomplete: " + ", ".join(missing))
print("MAIRA-2 downloaded:", folder.resolve())
PY
)
```

看到 `MAIRA-2 downloaded:` 后继续 3.3。下载中断时可以重跑同一条命令，已在这个目标目录成功下载且版本匹配的文件会被复用。 [Hugging Face 本地目录下载](https://huggingface.co/docs/huggingface_hub/guides/download#download-files-to-a-local-folder)

- 如果出现 `LocalTokenNotFoundError`、`401`、`403` 或 `GatedRepoError`，核对模型访问权限，然后用上面的终端登录命令保存同一账号的 Read 令牌，再重跑下载命令。 [MAIRA-2 模型页](https://huggingface.co/microsoft/maira-2)
- 若切换后大文件的已下载字节数两分钟仍不增加，或反复出现 `ConnectTimeoutError`，按 `Ctrl+C` 停止，切换到下面的 Windows 下载流程。

###### AutoDL 下载卡住时，从 Windows 下载再上传

按实际权重下载速度估算，27.5 GB 在 5 MB/s 时约 1 小时 32 分钟，在 10 MB/s 时约 46 分钟。Windows 下载和 SFTP 上传各需要一段时间。微软的发布页指向 Hugging Face 的受限仓库，以下步骤用已有访问权限的账号从该仓库下载。 [微软发布入口](https://www.microsoft.com/en-us/research/group/biomedical-imaging/tools/) · [官方权重文件列表](https://huggingface.co/microsoft/maira-2/tree/main)

**1. 在 Windows 下载。** 打开你电脑上的 PowerShell，进入项目目录。先查看 D 盘空间，建议预留至少 30 GB。使用你已在网页取得 MAIRA-2 访问权限的同一个账号，粘贴 Read 令牌完成登录。Windows 只负责下载文件。

```powershell
# 在仓库根目录运行
Get-PSDrive D
$env:HF_ENDPOINT = 'https://huggingface.co'
$env:HF_HUB_OFFLINE = '0'
.\.venv\Scripts\hf.exe auth login
```

登录成功后再执行。第一个命令只下载小配置，用来确认 Windows 能连通并具有模型权限。它成功后运行第二条下载完整模型。模型会保存在 `evidence/models/maira-2`。下载中断可以重跑完整下载命令，沿用同一目录。 [Hugging Face 下载说明](https://huggingface.co/docs/huggingface_hub/guides/download#download-files-to-a-local-folder)

```powershell
.\.venv\Scripts\hf.exe download microsoft/maira-2 processor_config.json --local-dir .\models\maira-2
.\.venv\Scripts\hf.exe download microsoft/maira-2 --local-dir .\models\maira-2 --max-workers 1
```

下载结束后列出分片，确认有 6 个以 `.safetensors` 结尾的文件，每个都有几 GB：

```powershell
Get-ChildItem .\models\maira-2 -Filter 'model-*-of-00006.safetensors' -File |
    Select-Object Name, Length
```

**2. 准备 AutoDL 目标目录。** 在服务器的 JupyterLab 终端执行。`mkdir -p` 可以沿用已创建的小写目录。

```bash
mkdir -p /root/autodl-tmp/MedVQA_Evidence/models/maira-2
df -h /root/autodl-tmp
```

**3. 从 Windows 传文件。** 回到 Windows PowerShell，进入已经下载好的模型目录，再启动现有的 `medvqa` SFTP 连接。

```powershell
Set-Location 'evidence/models/maira-2'
sftp medvqa
```

出现 `sftp>` 后先单独执行 `cd`，用 `pwd` 确认服务器目标路径确实以 `models/maira-2` 结尾，再执行 `put`。第一次传用 `put`。某个大文件是**由这份本地文件上传到一半才断开**，重连后可以用 `reput 文件名` 续传。已传完的分片可以保留。 [AutoDL 上传说明](https://www.autodl.com/docs/scp/) · [OpenSSH SFTP 续传说明](https://man.openbsd.org/sftp)

```text
cd /root/autodl-tmp/MedVQA_Evidence/models/maira-2
pwd
put *.json
put *.py
put tokenizer.model
put *.safetensors
bye
```

上传完成后，回到 AutoDL 的 JupyterLab 终端，继续第 3.3 节。看到 `Processor OK` 和 `Weight shards OK: 6` 再做第 3.4 节的离线推理。

###### Windows 和 AutoDL 分头下载

下面按当前进度分工。AutoDL 续传第 **2、3、4** 片和配置文件，Windows 下载第 **1、5、6** 片。Windows 显示有稳定下载速度后，再切换 AutoDL 的任务。如果 Windows 连不上或同样很慢，可以沿用已有的 AutoDL 下载。两台机器使用同一个 Hugging Face 仓库版本。 [Hugging Face 按版本和文件下载](https://huggingface.co/docs/huggingface_hub/guides/download)

**1. 用 Windows 登录 Hugging Face。** 在浏览器确认同一账号有 MAIRA-2 访问权限，再复制一个 **Read** 令牌。打开自己电脑的 **Windows PowerShell** 并运行：

```powershell
# 在仓库根目录运行
Get-PSDrive D
$env:HF_ENDPOINT = 'https://huggingface.co'
$env:HF_HUB_OFFLINE = '0'
.\.venv\Scripts\hf.exe auth login
.\.venv\Scripts\hf.exe auth whoami
```

D 盘建议留至少 16 GB。提示输入 token 时粘贴 `hf_` 开头的令牌并回车。输入时不会显示字符。如果问要不要保存 Git 凭据，输入 `n`。`auth whoami` 会显示当前登录的账号。 [MAIRA-2 访问页](https://huggingface.co/microsoft/maira-2) · [令牌设置](https://huggingface.co/settings/tokens)

**2. 取版本号，开始 Windows 下载。** 在 AutoDL 的另一个 **JupyterLab 终端**执行下面这条。下载任务可以同时继续。复制输出的 40 位模型版本号。

```bash
head -n 1 /root/autodl-tmp/MedVQA_Evidence/models/maira-2/.cache/huggingface/download/config.json.metadata
```

回 Windows PowerShell 执行。出现版本提示时粘贴刚复制的 40 位值。

```powershell
$rev = (Read-Host '粘贴 AutoDL 输出的 40 位模型版本号').Trim()
if ($rev -notmatch '^[0-9a-fA-F]{40}$') { throw '模型版本号应为 40 位十六进制字符' }
.\.venv\Scripts\hf.exe download microsoft/maira-2 model-00005-of-00006.safetensors model-00006-of-00006.safetensors model-00001-of-00006.safetensors --revision $rev --local-dir .\models\maira-2 --max-workers 1
```

**3. 将 AutoDL 切换成只续传第 2、3、4 片。** Windows 已经开始稳定下载后，在 AutoDL 原下载终端按 `Ctrl+C`，等提示符回来，再把下面整块粘贴进去。

```bash
(
  cd /root/autodl-tmp/MedVQA_Evidence || exit 1
  if [ -r /etc/network_turbo ]; then source /etc/network_turbo; fi
  unset HF_ENDPOINT HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  REV="$(head -n 1 models/maira-2/.cache/huggingface/download/config.json.metadata)" || exit 1
  MAIRA_REV="$REV" HF_HUB_DISABLE_XET=1 HF_HUB_ETAG_TIMEOUT=20 HF_HUB_DOWNLOAD_TIMEOUT=40 \
    /root/autodl-tmp/maira-runtime/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    "microsoft/maira-2",
    revision=os.environ["MAIRA_REV"],
    local_dir="models/maira-2",
    token=True,
    allow_patterns=["*.json", "*.py", "tokenizer.model", "model-0000[2-4]-of-00006.safetensors"],
    max_workers=2,
)
PY
)
```

**4. 只上传 Windows 完成的三个分片。** 在 Windows PowerShell 执行：

```powershell
Set-Location 'evidence/models/maira-2'
sftp medvqa
```

出现 `sftp>` 后，先切到服务器的模型目录，再上传已下载完成的分片：

```text
cd /root/autodl-tmp/MedVQA_Evidence/models/maira-2
pwd
put model-00001-of-00006.safetensors
put model-00005-of-00006.safetensors
put model-00006-of-00006.safetensors
bye
```

首次上传用 `put`。只有同一份 Windows 文件已经上传了一部分才用 `reput 文件名` 续传。两边都完成后，在 AutoDL 做第 3.3 节的验证。确认处理器和六个分片都能读取，再继续第 3.4 节。

##### 3.3 验证处理器和六个权重分片

在 AutoDL 项目根目录执行下面的检查。它先单独加载处理器，再检查索引里的分片，检查过程不需要把权重加载到 GPU。 [MAIRA-2 官方处理器加载示例](https://huggingface.co/microsoft/maira-2#how-to-get-started-with-the-model)

```bash
cd /root/autodl-tmp/MedVQA_Evidence
HF_HUB_OFFLINE=1 /root/autodl-tmp/maira-runtime/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("models/maira-2").resolve()
required = (
    "config.json", "processor_config.json", "preprocessor_config.json",
    "tokenizer_config.json", "tokenizer.json", "tokenizer.model",
    "processing_maira2.py", "configuration_maira2.py", "modeling_maira2.py",
    "model.safetensors.index.json",
)
missing = [name for name in required if not (root / name).is_file() or (root / name).stat().st_size == 0]
if missing:
    raise SystemExit(f"Missing or empty in {root}: {', '.join(missing)}")

for name in ("processor_config.json", "preprocessor_config.json"):
    config = json.loads((root / name).read_text(encoding="utf-8"))
    print(name, "AutoProcessor =", config.get("auto_map", {}).get("AutoProcessor"), flush=True)

from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(str(root), local_files_only=True, trust_remote_code=True)
if not callable(getattr(processor, "format_and_preprocess_phrase_grounding_input", None)):
    raise SystemExit(f"Unexpected processor: {type(processor).__name__}")
print("Processor OK:", type(processor).__name__, flush=True)

from safetensors import safe_open
index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
shards = sorted(set(index["weight_map"].values()))
if not shards:
    raise SystemExit("No weight shards listed in the index.")
for name in shards:
    with safe_open(str(root / name), framework="pt", device="cpu") as weights:
        if not weights.keys():
            raise SystemExit(f"No tensors found in shard: {name}")
print("Weight shards OK:", len(shards))
PY
```

看到 `Processor OK: Maira2Processor` 和 `Weight shards OK: 6` 后，继续 3.4。如果出现 `Unrecognized processing class` 或提示文件缺失，检查文件是否直接位于小写 `models/maira-2/` 这一层，再用上一步下载命令补齐。

如果分词器加载时报 `gencode 7.36.1 runtime 5.28.3`，为 `maira-runtime` 安装匹配的 Protobuf。报错中的 `5.28.3` 来自该虚拟环境继承的 Conda 包。模型文件可以继续使用。 [Protobuf 运行时兼容规则](https://protobuf.dev/support/cross-version-runtime-guarantee/)

```bash
/root/autodl-tmp/maira-runtime/bin/python -m pip install --upgrade --only-binary=:all: 'protobuf==7.36.1'
/root/autodl-tmp/maira-runtime/bin/python -c 'from google import protobuf; from sentencepiece import sentencepiece_model_pb2; print("Protobuf OK:", protobuf.__version__, protobuf.__file__)'
```

如果当前 pip 镜像找不到 `7.36.1`，在安装命令末尾加上 `--index-url https://pypi.org/simple`。看到 `Protobuf OK: 7.36.1`，并确认路径位于 `/root/autodl-tmp/maira-runtime/`，重新运行 3.3 的离线检查。 [PyPI 上的 Protobuf 7.36.1](https://pypi.org/project/protobuf/7.36.1/)

##### 3.4 用同一份本地模型跑单张和批量

先在 **AutoDL 的项目根目录**跑验证图 `xmlab164`。程序生成报告时只读取影像，人工框随后用于评估预测。

```bash
HF_HUB_OFFLINE=1 /root/autodl-tmp/maira-runtime/bin/python -m slake_vqa maira --root data/slake \
  --model models/maira-2 --split validation --image-id xmlab164 \
  --output results/gpu/maira_heart_smoke.json
```

打开烟测 JSON，查看 `finding_status`、`report`、`box_source` 和 `boxes_xyxy`。报告明确写到心影增大时记为 `present`；该句没有框时，会再用这句做一次短语定位。`uncertain` 或 `no_positive_mention` 表示规则没有得到明确的阳性表述，生成这样的结果文件也能验证推理链路。

单张跑通后，再用同一份本地模型跑批量：

```bash
HF_HUB_OFFLINE=1 /root/autodl-tmp/maira-runtime/bin/python -m slake_vqa maira --root data/slake \
  --model models/maira-2 --split audit --output results/gpu/maira_heart_audit.json
```

`--split audit` 一次处理 SLAKE 的验证集和测试集胸片，并在 `summary_by_split` 分别输出指标。同配置、同输出路径可以续跑。用 `summary_by_split.test` 读取 30 张测试胸片的结果。SLAKE 有 6 张带心影增大人工框的测试胸片，汇总区分报告识别与框定位。没有框的图只记为缺少该类框标注，识别指标衡量与数据集标注的一致性。

**单张定位实测。** `maira_heart_smoke.json` 的 `cases[0]` 对应验证图 `xmlab164`。`matched_findings` 包含 `Heart is enlarged.`，`finding_status=present`，`box_source=grounded_report` 表示框来自这句报告。预测框 `boxes_xyxy` 是 `[465, 517, 947, 885]`，人工框 `reference_boxes_xyxy` 是 `[399, 477, 940, 924]`，交并比 `best_box_iou` 为 **0.7152**。该图也在批量验证集中，两个文件中该图记录一致。 MAIRA-2 单张结果可在本机查看 `artifacts/gpu-run/results/gpu/maira_heart_smoke.json` · [MAIRA-2 批量结果](results/maira_cardiomegaly.json)

**57 张胸片批量实测。** 验证和测试分别读 `summary_by_split.validation` 和 `summary_by_split.test`，合计读 `summary`。下表的「报告命中」表示模型写出明确的心影增大陈述，且同一张图有 SLAKE 心影增大人工框。 [MAIRA-2 批量结果](results/maira_cardiomegaly.json)

| 划分 | 胸片数 | 有人工框 | 报告阳性 | 报告命中 | 报告精确率 | 报告召回率 | 平均框 IoU | IoU ≥ 0.5 命中率 | 有人工框却无预测框 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 验证 | 27 | 6 | 3 | 3 | 100.00% | 50.00% | 0.3620 | 50.00%（3/6） | 3 |
| 测试 | 30 | 6 | 4 | 2 | 50.00% | 33.33% | 0.2566 | 33.33%（2/6） | 4 |
| 合计 | 57 | 12 | 7 | 5 | 71.43% | 41.67% | 0.3093 | 41.67%（5/12） | 7 |

- **报告识别。** 测试集 `reported_cardiomegaly=4`、`images_with_cardiomegaly_box=6`、`matched_box_annotated_images=2`。所以报告精确率是 `2/4`，召回率是 `2/6`。另两张报告阳性图 `xmlab312`、`xmlab357` 缺该类人工框，结论限于与数据集框标注的一致性。
- **框定位。** `mean_box_iou_on_annotated_images` 在 6 张有人工框的测试图上平均，每张取 `cases[].best_box_iou`，无预测框记 0。`xmlab135`、`xmlab142` 的 IoU 分别为 0.7211、0.8184，其余 4 张为 0，且均为 `no_positive_mention`。因此测试集 `box_hit_rate_at_0_5=2/6`；在这 6 张图里，覆盖不足主要出现在报告未明确提到目标的样本。

两项数字都是这批胸片的描述统计。测试集只有 6 张有心影增大框，单张冒烟的 IoU 用于核对定位链路。 [逐图报告、状态与框](results/maira_cardiomegaly.json) · [指标计算代码](maira.py)

回到 Qwen 主环境，运行验证题 `9923`。`organ` 这一列在胸片分支表示审计目标 `cardiomegaly`。

```bash
source /root/autodl-tmp/medvqa-runtime/bin/activate
printf 'split,qid,organ\nvalidation,9923,cardiomegaly\n' > results/gpu/maira_example_cases.csv

python -m slake_vqa audit run --root data/slake --cases results/gpu/maira_example_cases.csv \
  --model models/Qwen3-VL-2B-Instruct --device cuda --mask-source maira \
  --maira-results results/gpu/maira_heart_smoke.json \
  --output results/gpu/cardiomegaly_audit_example.json
```

`localization` 保留报告识别状态、预测框和来源。框为空时只做全图扰动。框较大、原形平移放不下时，胸片分支可用不重叠的同面积替代区域，记录为 `control_match=same_area`。这时比较控制了干预面积，区域形状仍可能影响答复。批量审核可以用 `python -m slake_vqa audit candidates --root data/slake --organ cardiomegaly --modality X-RAY --output results/gpu/heart_candidates.csv` 导候选，另跑一次并添加 `--query-term heart` 可扩大候选范围。人工检查题意后编辑正式清单，再用完整的 `maira_heart_audit.json` 跑。CT 和胸片分别标注、分别拟合风险估计器。

**胸片审计实测题 9923。** `cases[0].reference_answer` 为 `Cardiomegaly`，Base Qwen 原答 `pneumonia`，`confidence=0.9511`。`localization` 确认 MAIRA-2 已提到心影增大并返回预测框，该框与单张定位结果一致，`region_file.sha256` 也与归档中的 `maira_heart_smoke.json` 一致。框覆盖 177,376 像素，对照是同形状的异位置区域。单个矩形框生成的 `box` 和 `mask` 扰动区域相同。`variants` 中全图、框、掩膜和对照后均答 `pneumonia`，四个 `persistence` 都为 1，所以 `target_specific_change=1−1=0`。这次目标区已经定位，当前模糊条件下 Qwen 的原答依然保持。`confidence` 是生成置信度，风险评分须等人工标签齐备后拟合。 胸片单题审计 JSON可在本机查看 `artifacts/gpu-run/results/gpu/cardiomegaly_audit_example.json` · 胸片单题审计 CSV可在本机查看 `artifacts/gpu-run/results/gpu/cardiomegaly_audit_example.csv`

这两份单题 CSV 各有一条验证样本，`unsupported` 均为空，两题均有对照。原答和参考有分歧可作为人审核查线索，目前没有可用于拟合风险估计器的人工标签。 CT 审计 CSV可在本机查看 `artifacts/gpu-run/results/gpu/ct_multi_liver_audit.csv` · 胸片单题审计 CSV可在本机查看 `artifacts/gpu-run/results/gpu/cardiomegaly_audit_example.csv`

#### 第四步：在同一网页里验证两个分支

在主环境启动服务：

```bash
python -m slake_vqa demo --data data/slake --model cuda --model-id models/Qwen3-VL-2B-Instruct \
  --checkpoint checkpoints/gpu/unet_ct_multi.pt \
  --maira-results results/gpu/maira_heart_audit.json
```

影像问答中，CT 示例填 `xmlab207`、模态 `CT`、审计目标 `liver`，胸片示例填 `xmlab164`、模态 `X-RAY`、审计目标 `cardiomegaly`。点「预览候选区域」，有掩膜或框时会叠上绿色，再发送问题查看审计详情。胸片页面也会单独显示 MAIRA-2 报告识别状态。两种区域来源可同时启用；`--risk-model` 每次加载一个与该来源和目标匹配的估计器。

## 5. 拟合风险评分并导出测试结果

```bash
python -m slake_vqa audit score --review results/gpu/audit-base-reviewed.csv \
  --normalization percentile \
  --predictions results/gpu/audit-risk-test.csv \
  --export-estimator checkpoints/gpu/audit_estimator.json \
  --output results/gpu/audit-risk-summary.json
```

**风险是怎么计算的？**

1. 每题先得到原置信度，以及全图、框、掩膜、对照下的回答保持度和局部差值。加对齐后，还会用掩膜或对照干预后的图文相似度变化。
2. `--normalization percentile` 给原特征增加一列验证集相对位置。例如验证集中某个特征有 100 个有效值，其中 80 个小于或等于这道题的值，该题的分位数特征就是 0.80。脚本保留原值，拼上分位数。参考分布来自特征完整的验证题，包含未标注的验证题，测试题只接受评分。
3. 脚本标准化这些特征，用有 `0/1` 标签的验证题拟合逻辑回归。每一题将标准化特征加权求和、加截距，再经 Sigmoid 得到 0–1 的分数。模型根据标注学习每一项该往哪个方向影响分数。分数越高，模型越倾向于把这道题排在「回答缺少视觉支持」的一端；比如分位数 0.80 和最终风险 0.80 是两个不同量。 [论文公式 (19)–(21)](https://arxiv.org/html/2610.38389v1#S3.SS4)

| 看哪里 | 产物与读法 |
| --- | --- |
| `audit-risk-test.csv` | 每道测试题的题号、原答、人工标签和 `risk_score`。未标注但特征完整仍可评分；缺特征的行保留，显示 `missing_features` |
| `audit-risk-summary.json` | 汇总验证拟合数、测试评估数、测试评分数、正类比例及各方法 AP / AUROC。评估数只含标签和特征齐全的测试题 |
| `audit_estimator.json` | 网页加载的参数，包含特征、验证参考分布、均值、尺度、系数和审计来源。每份标注表使用一种来源，可导出 U-Net 或 MAIRA-2 估计器 |

`low_confidence` 用 `1 − 原答置信度` 排序。`global_only`、`box`、`mask`、`random_region` 分别比较全图基线和加入单一局部信号的效果。这里的 `random_region` 指 `control` 那个随机平移出的同面积区域。`matched_control` 是本项目的完整特征组。脚本在相同一批测试题上比较这些分支，所以分数差可以解释为当前样本下加入某类信号的作用。

- **AUROC** 看一个有风险样本的分数能否排在一个有支持样本前面。**AP**（JSON 的 `auprc_ap`）看高风险区间能否优先找出正例，报告时附上测试正例比例作为对照。两个都用真实人工标签计算。
- `test_scored` 可能大于 `test_evaluated`，因为缺少标签的一题也可以有风险分数。`excluded_missing_features` 统计有标签但缺特征的行。
- 若要做「分位数有无」的消融，用同一个 `--review` 再执行一次 `score`，把 `--normalization` 改为 `standard`，三处输出都改名。`standard` 仍做标准化，只用原始特征。两次对齐样本，比较两份汇总的 AP 和 AUROC。
- 首轮只展示连续分数。需要高风险标记时，先在独立于测试集的标注数据上确定阈值，再把阈值固定到网页 `--risk-threshold`。论文还报告了分类 F1；本节的自动汇总目前输出的是无需阈值的排序指标。

### 可选升级：把图文对齐真正纳入审计

先完成上面的基线。AutoDL 已有 CUDA 依赖约束文件时，以下命令保留已配对的 PyTorch 版本。BiomedCLIP 首次加载需要下载模型和文本编码器资源，请先用一题确认能加载。 [BiomedCLIP 模型与用法](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)

```bash
python -m pip install -c /root/autodl-tmp/medvqa-torch-constraints.txt 'open_clip_torch==2.23.0'
python -m slake_vqa audit run --root data/slake --cases slake_vqa/examples/audit.csv \
  --model "$MODEL" --device cuda --mask-source unet --alignment \
  --checkpoint checkpoints/gpu/unet_unweighted.pt \
  --output results/gpu/audit-example-aligned.json
```

看 CSV 是否出现 `alignment_mask` 和 `alignment_control` 的数值。这里用 BiomedCLIP 比较原图和扰动图各自与「问题＋原答」的余弦相似度，输出原始相似度减去扰动后相似度。确认单题后，把 `--cases` 换成 `data/audit_cases.csv`，换新 `--output` 重新审计全批；核对同一 `qid` 和原答案后可沿用人工标签。评分时在同一条 `score` 命令加 `--alignment`，更换 CSV、JSON 和估计器输出路径。此时用同一份汇总内的 `response_only` 和 `matched_control` 比较回答信号与加入对齐后的表现。

## 6. 在网页查看预测掩膜和风险评分

```bash
python -m slake_vqa demo --data data/slake --model cuda --model-id "$MODEL" \
  --checkpoint checkpoints/gpu/unet_unweighted.pt \
  --risk-model checkpoints/gpu/audit_estimator.json
```

网页的问答模型、适配器、分割权重、扰动条件要与打分模型的训练数据一致。上述命令加载 Base 和无 BiomedCLIP 的估计器，使用默认模糊半径 12 和输出上限 48 token。用 LoRA 批次训练估计器时，在网页命令也加同一个 `--adapter`。如用了对齐版估计器，服务会按保存的特征自动启用 BiomedCLIP。

在连接 AutoDL 的 VS Code 打开端口转发面板，将服务器的 `8000` 转发到本机，然后打开面板给出的本机地址。默认应用只监听回环地址；可在启动命令末尾用 `--port 8001` 改服务器端口，再映射对应端口。

1. 选择「影像问答」。从 `audit-risk-test.csv` 选一个 `scored` 样例，照填它的 `image_id` 和 `question`。
2. 展开「分割审计设置」，填 `CT` 和 `left lung`。点「预览预测掩膜」看绿色区域，再发送问题。
3. 在「影像回答」下查看 0–1 的风险分数，展开「区域审计」看各扰动回答和原始特征。若特征不足，网页显示无法评分。
4. 已跑通 GPU 推理、尚未完成标注时，启动命令写到 `--checkpoint checkpoints/gpu/unet_unweighted.pt` 为止，末尾去掉反斜杠，先查看影像回答和审计特征。体验知识检索时切到检索模式，问 `How does a CT scanner reconstruct a slice?`，检查返回的来源。

**通过标准**是同一条合格测试题能看到预测掩膜、扰动前后回答和风险分数。对于同一组模型与推理设置，页面可见值保留三位小数，接口 `audit.learned_risk_score` 可和 CSV 核对。

## 7. 哪些结果要补到项目里

本次归档已核对到 4.4 的 3.4，包括 GPU 训练、单题审计和 MAIRA-2 批量定位。 [LoRA 结果](#23-已跑出的-gpu-结果) · [U-Net 结果](#32-已跑出的-gpu-分割结果) · [CT 与胸片审计结果](#44-多类-ct-与-maira-2-胸片审计)

| 结果 | 验收材料 | 当前记录 |
| --- | --- | --- |
| LoRA 冒烟 | `lora-smoke/training_summary.json`、适配器与一题评测 | **已完成。** RTX 4090 D，2 步，训练/验证损失 1.5200 / 0.5066，GPU 重载通过 |
| 正式 LoRA 训练 | `slake-lora-v1/training_summary.json`、`checkpoint-615`、适配器 | **已完成。** 4,919 训练题、615 步，训练/验证损失 0.3198 / 0.1567 |
| 验证集 Base / LoRA 对比 | `vqa-val-150` 的 `run.json`、`summary.json`、逐题预测 | **已完成。** 150 题、73 张图。原图 EM 47.33% → 74.00%，仅文本 35.33% → 38.67%，详见第 2.3 节 |
| 独立测试集 Base / LoRA 对比 | `vqa-test-150` 的 `run.json` 和 `summary.json` | **待跑。** 题数、影像数、四组 EM / Token-F1、图像增益 |
| U-Net 加权消融 | `unet_weighted.json`、`unet_unweighted.json` 和两个 checkpoint | **已完成。** GPU 25／5／5 张，前景占 2.5564%。加权／不加权验证 Dice 0.5472／0.5580，测试 Dice 0.5904／0.6289，详见第 3.2 节 |
| 多类 CT U-Net | `checkpoints/gpu/unet_ct_multi.pt`、`results/gpu/unet_ct_multi_summary.json` | **已完成。** 200／42／40 张，第 63 轮最佳，测试宏 Dice 0.6583、IoU 0.5558，肝脏测试 Dice 0.7958 |
| 多类 CT 肝脏单题审计 | `results/gpu/ct_multi_liver_audit.json` | **已完成。** 题 9962，原答 `kidney`、掩膜后 `left kidney`、对照 `kidney`，目标特异变化 0.3333 |
| MAIRA-2 心影增大定位 | `maira_heart_smoke.json`、`maira_heart_audit.json` | **已完成。** 单张 IoU 0.7152。批量验证 27 张、测试 30 张，测试报告精确率 50.00%、召回率 33.33%、平均框 IoU 0.2566、框命中 2/6 |
| 胸片单题审计 | `cardiomegaly_audit_example.json` | **已完成。** 题 9923，MAIRA-2 给出心影增大框，Qwen 在四种扰动下仍答 `pneumonia`，目标特异变化为 0 |
| 局部审计批量 | 验证/测试审核副本、原始审计 JSON | **待补。** 两个单题示例已有。需整理批量题单并人工核对支持性，再统计验证/测试题数、标签数和缺失对照数 |
| 评分与消融 | `audit-risk-summary.json`、`audit-risk-test.csv`、`audit_estimator.json` | **待填。** AP / AUROC、测试正类比例、评分覆盖率、各方法对比 |
| GPU 网页 | 一条可重跑请求或截图 | **待填。** 一条实际的原答、掩膜、对照和评分，附模型及 checkpoint |

收工前从 JupyterLab 下载汇总和训练产物。若要一次打包 GPU 结果和权重，可在项目根目录运行 `tar -czf /root/autodl-tmp/MedVQA-Evidence-gpu-results.tar.gz results/gpu checkpoints/gpu`。复习和复跑可从小样本开始。正式表格填写独立测试结果和对应的样本量。

## 附录：本地知识检索演示

Windows PowerShell 从项目根目录运行。这个入口适合离线验证检索、API 和跨会话讲解偏好。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r evidence/requirements-app.txt
.\.venv\Scripts\python.exe -m evidence.rag evaluate
.\.venv\Scripts\python.exe -m slake_vqa verify
.\.venv\Scripts\python.exe -m slake_vqa demo
```

浏览器打开 [应用](http://127.0.0.1:8000/) 或 [API 文档](http://127.0.0.1:8000/docs)。修改讲解偏好后点击「新会话」可检查同一档案的偏好恢复。默认 Docker 镜像也运行此知识检索入口。

```bash
docker build -f slake_vqa/Dockerfile -t medvqa-evidence .
docker run --rm -p 127.0.0.1:8000:8000 \
  --mount source=medvqa-data,target=/app/slake_vqa/artifacts/data \
  --mount source=medvqa-qdrant,target=/app/evidence/runtime medvqa-evidence
```

项目入口和已有结果索引见 [README](README.md)。
