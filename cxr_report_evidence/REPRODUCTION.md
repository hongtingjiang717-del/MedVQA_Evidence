# IU X-Ray 复现报告与历史 GPU 操作记录

在本机复现时从 [本机 README](README.md) 开始。本文保留实验来历、原始结果位置和此前 GPU 运行的具体配置。下面的 AutoDL 示例沿用此前服务器的模型和数据盘路径。

这套流程从正位胸片生成英文 `Findings`，在同一病例上比较 Base Qwen3-VL 和胸片报告 LoRA。每个模型分别生成原图＋文本和仅文本报告，参考报告在网页中并排展示。MAIRA-2 定位模型生成的发现，视觉核查器比较原图、目标模糊和同面积异位置对照。医学影像知识问答通过 LangGraph 调用本地 Qdrant 检索。

SLAKE 问答、CT U-Net 和心影增大实验归入 [SLAKE 子项目](../slake_vqa/README.md)。IU X-Ray 复用 Qwen 基座、MAIRA-2 模型和知识库，为英文 Findings 单独训练报告适配器。

**当前先跑 IU X-Ray（NLM Open-i）**。它有胸片 PNG 和配套的英文报告 XML，下载走 NLM 的入口。先用 24 个病例验证整条链路，后续可按附录接入 MIMIC-CXR。 [NLM 数据说明](https://openi.nlm.nih.gov/faq#collection) · [数据集论文](https://pubmed.ncbi.nlm.nih.gov/26133894/)

| 环节 | 文件 |
| --- | --- |
| IU X-Ray 报告配图、正位筛选、固定划分 | `openi.py` |
| 通用病例清单、MIMIC-CXR 入口 | `data.py` |
| 报告 LoRA、四组生成、报告指标 | `reports.py` |
| 短语定位、反事实特征、覆盖率与差值汇总 | `audit.py` |
| 命令入口 | `__main__.py` |
| FastAPI、LangGraph 和浏览器界面 | `web.py`、`demo.html` |

## 0. 当前结果与本机查看

本节使用 `artifacts/` 中的原件。包含早期 55 例不完整归档、本机 CPU 单例、100 例完整 GPU 评测和同批次批量审计。本机复现请从 [README](README.md) 开始。 [完整测试结果](results/test100_gpu_metrics.json) · [批量审计结果](results/test100_gpu_audit.json)

### 0.1 数据和正式训练

IU X-Ray 是胸部 X 光数据集。下载包中有 **7,470 张 PNG**，含正位和侧位。项目用正位名单筛图，关联 XML 中有英文 `Findings` 的报告，匹配 **3,201 例**。用报告 UID 按种子 42 划分，池大小为训练 **2,561**、验证 **320**、测试 **320**，实际选 **1,600／100／100**，每例选一张正位图。按病例关联正位图和参考段落，使用报告 UID 隔离划分。正式报告提示包含可用的检查指征 `indication`。 [正式清单](results/data_inventory.json)

| 项目 | 已跑出的结果 | 从哪里核对 |
| --- | --- | --- |
| 正式 LoRA 更新 | 1,600 训练例、100 验证例，梯度累积 8，10 轮共 2,000 步 | `train_studies`、`validation_studies`、`gradient_accumulation_steps`、`epochs_completed`、`global_step` |
| 训练耗时和损失 | 4,284.96 秒，约 71 分 25 秒，训练平均损失 0.9302 | `train_metrics.train_runtime`、`train_metrics.train_loss` |
| 选出的权重 | 第 7 轮、第 1,400 步，验证损失 **1.0231**。跑完第 10 轮，早停标记为 false | `best_global_step`、`epoch_validation`、`stopped_early` |
| 后三轮验证损失 | 第 8／9／10 轮分别为 1.0291／1.0288／1.0359 | `epoch_validation`。汇总的 `validation_metrics.eval_loss` 是回载最佳权重后再次测得的 1.0231 |

验证损失逐轮为 1.2903 → 1.1595 → 1.0817 → 1.0617 → 1.0492 → 1.0335 → **1.0231** → 1.0291 → 1.0288 → 1.0359。下载包的 `adapter/adapter_model.safetensors` 与 `checkpoint-1400/` 中的适配器 SHA256 一致，以 `f8c8b3a6cfdcdff8` 开头。 [训练摘要](results/lora_training.json)

### 0.2 下载包里的报告进度

`reports.json` 的 `config.split` 是 `audit`，表示在验证和测试集生成四组。计划 200 例各生成四份。实际文件的 `rows[].reports` 只有 **55 个验证病例的 Base 两组**，共 110 份报告。完整四组为 **0/200**。文件登记的 `config.adapter_sha256` 以 `9ff7c1cb078db0637` 开头，与正式权重的 `f8c8b3a6cfdcdff8` 不同。用正式权重时请写入新结果目录。下载包里未找到 `ground.json`、`probes.json`、`probe-summary.json`。

| 划分与范围 | Base 看图＋文本 | Base 仅文本 | LoRA 看图＋文本 | LoRA 仅文本 |
| --- | ---: | ---: | ---: | ---: |
| 下载包验证集，55 个配对病例 | 0.1073 | 0.1101 | 0 份输出 | 0 份输出 |
| 下载包测试集 | 0 份输出 | 0 份输出 | 0 份输出 | 0 份输出 |

数值是 ROUGE-L F1 的逐例平均。用同一病例的参考 `Findings`，以词序为基础比较英文重合度。看图减仅文本为 **−0.0028**，在 55 例中 25 例看图分较高、28 例仅文本分较高、2 例持平。参考段落中位长度为 30 词，Base 看图和仅文本分别为 175 和 166 词。样本是这次作业已写出的验证集前 55 例。这组结果只覆盖 Base 的小范围比较，报告与参考的文字重合也无法直接度量临床准确性。 服务器报告原件可在本机查看 `artifacts/results/cxr-openi-formal-eval.8VY46V/reports.json` · [本机复算与逐例索引](results/base_validation_partial.json) · [ROUGE-L 实现说明](https://pypi.org/project/rouge-score/0.1.2/)

### 0.3 用正式 LoRA 补跑的四组对照

本机从正式清单选第一张独立测试片 `iu-CXR3023`，沿用 768 像素长边上限、256 生成 token 上限和检查指征。同一 Qwen3-VL-2B 基座，分别有无第 1,400 步的报告适配器、输入有无该张原图。CPU 推理约 4 分钟。仅文本条件只有报告指令和检查指征，两组输入都不把参考报告发给模型。英文原文用于评分。

| 条件 | ROUGE-L F1 | 对照参考 Findings 的可观察差异 |
| --- | ---: | --- |
| Base 看图＋文本 | 0.1117 | 输出心脏及纵隔正常，也提到未在参考报告中记录的左下肺密度增高和病史 |
| Base 仅文本 | 0.1158 | 写了右上肺密度增加及较长的通用描述，参考报告记录的是左上肺小钙化肉芽肿 |
| LoRA 看图＋文本 | **0.7778** | 三句与参考报告中的正常心脏、纵隔和清晰肺野相符，漏掉左上肺钙化肉芽肿 |
| LoRA 仅文本 | 0.2162 | 写了肺野清晰、心脏及纵隔正常，提到无积液或气胸，漏掉左上肺钙化肉芽肿 |

LoRA 看图输出在这一例里较短，保留了参考段落句序中的三句，因此词序匹配高。单例用于确认适配器加载和对照流程。批量平均效果见第 0.5 节。四组英文原文与适配器哈希可在本机查看 `artifacts/results/local-review/cpu-example-reports.json`。 [单例指标](results/cpu_example_metrics.json)

<details>
<summary>在本机复现这一个病例</summary>

从项目根目录运行。命令会另存一份单例结果，本机需要有下载包的正式适配器和 Qwen 基座。

```powershell
$archive = 'cxr_report_evidence/artifacts'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.\.venv\Scripts\python.exe -m cxr_report_evidence generate `
  --manifest "$archive/data/cxr-openi/manifest-formal.json" `
  --images "$archive/data/cxr-openi/images" --model evidence/models/Qwen3-VL-2B-Instruct `
  --adapter "$archive/checkpoints/cxr-openi-formal.xlu8LK/adapter" `
  --split test --limit 1 --use-indication --device cpu `
  --output cxr_report_evidence/artifacts/results/local-review/cpu-example-rerun.json
```

</details>

### 0.4 在本机查看报告

用 IDE 打开仓库，在仓库根目录的终端运行：

```powershell
python -m cxr_report_evidence.web --local-results
```

打开 [报告对照网页](http://127.0.0.1:8010/?study=iu-CXR3023)，查看原图、四份英文报告和参考 Findings。同目录有审计产物时会一同加载。知识问答页可查本地语料和引用。

### 0.5 独立测试 100 例的正式四组结果

2026 年 11 月 25 日下载的 GPU 结果位于本机 `artifacts/results/cxr-openi-test100-gpu/`。其中 `reports.json` 保存 100 个正式测试病例的 400 份输出，`report-metrics.json` 的 `by_split.test` 保存服务器汇总，`generate.log` 保存生成日志。已核对 100 个病例 ID、原图像素哈希、正式清单和第 1,400 步适配器的 SHA256。对原始 Findings 与生成报告重算 ROUGE-L，四组均与服务器汇总一致。 [可公开汇总](results/test100_gpu_metrics.json)

| 模型 | 原图＋文本 | 仅文本 | 加图的平均变化 |
| --- | ---: | ---: | ---: |
| Base | 0.1159 | 0.1194 | −0.0035 |
| LoRA | **0.3024** | **0.2638** | +0.0386 |
| LoRA 减 Base | +0.1865 | +0.1444 | |

看图时，LoRA 在 95 例上分数更高、5 例更低。仅文本时，LoRA 在 85 例更高、14 例更低、1 例持平。固定 LoRA 后，原图令 53 例更高、42 例更低、5 例持平。由此可以报告英文重合度的配对变化。报告里的临床细节与图像对应程度需要后续逐条核查。

本机可用 `python -m cxr_report_evidence.web --local-results` 浏览 100 例。首例的两份 LoRA 输出与此前 CPU 单例逐字一致。

### 0.6 正式批量反事实审计

`artifacts/results/cxr-openi-test100-gpu/` 中的 `ground.json`、`probes.json` 和 `probe-summary.json` 已与 `reports.json` 核对 SHA256。逐条核对病例、报告组、发现编号和文本，共 **100 个病例、400 份报告、3,711 条发现**。重新运行汇总与下载的结果完全一致。审计使用 MAIRA-2、原图短边的 2.5% 作为模糊半径，以及 Base Qwen 核查器。此次未启用额外图文对齐模型。 [审计汇总和源文件哈希](results/test100_gpu_audit.json)

每份报告最多检查前 4 条发现。2,224 条超出条数上限，403 条按规则跳过定位，剩余 **1,084** 条参与定位。其中 **738** 条得到候选框，344 条未得到框，2 条输出格式不完整。得到框的条目里，**585** 条有有效匹配对照，153 条无可用对照。有效对照中，**562** 条同形且同面积，**23** 条为同面积异形状。有效覆盖率按全部发现计算为 **15.76%**，按进入定位的条目计算为 **53.97%**。

| 报告组 | 进入定位 | 有效对照／全部发现 | 差值均值 | 差值中位数 |
| --- | ---: | ---: | ---: | ---: |
| Base 看图 | 352 | 183／1,496 | +0.02490 | −0.00000077 |
| Base 仅文本 | 320 | 299／1,499 | +0.00859 | −0.00000200 |
| LoRA 看图 | 191 | 41／349 | +0.04650 | +0.00027962 |
| LoRA 仅文本 | 221 | 62／367 | −0.01384 | −0.00002424 |

差值定义为 `对照图支持分 − 目标模糊图支持分`，只统计有效对照。各组的中位数都接近 0。LoRA 看图组 41 条有效对照中，24 条差值为正，17 条为负。四组的文本长度、进入定位的条目和有效对照数均不同。均值用于描述各组已核查条目的反应。逐条正确性结论与风险概率需要独立标注。

**网页例子 `iu-CXR227`。** LoRA 看图的发现 `The heart is enlarged.` 得到 MAIRA-2 候选框 `[186, 195, 451, 371]`。核查器在原图上的支持分为 **0.7847**，目标区域模糊后为 **0.0005**，同形状同面积的对照区模糊后为 **0.7096**，差值约 **+0.7091**。这是用于展示定位与核查链路的样例。选择「LoRA」「原图＋文本」，点击该条发现即可切换三种图像。 [打开审计案例](http://127.0.0.1:8010/?study=iu-CXR227)

## 1. 把代码从 Windows 同步到 AutoDL

你之前的 AutoDL 项目路径是 `/root/autodl-tmp/MedVQA_Evidence`。本次在该目录下新增 `cxr_report_evidence/`，继续使用其中已有的 `evidence/` 和 `models/`。Windows 打开 PowerShell：

```powershell
# 在仓库根目录运行
tar.exe -czf cxr-report-code.tar.gz --exclude=__pycache__ --exclude=artifacts --exclude=models --exclude=runtime cxr_report_evidence evidence slake_vqa
sftp medvqa
```

看到 `sftp>` 后逐条执行。密码使用 AutoDL SSH 密码。

```text
cd /root/autodl-tmp
pwd
put cxr-report-code.tar.gz
bye
```

打开 **AutoDL JupyterLab 终端**。用 `tar -tzf` 查看内容，再解包：

```bash
cd /root/autodl-tmp/MedVQA_Evidence
tar -tzf /root/autodl-tmp/cxr-report-code.tar.gz
tar -xzf /root/autodl-tmp/cxr-report-code.tar.gz
test -f evidence/audit/maira.py && test -f evidence/vqa/model.py && test -f evidence/rag/corpus.json
```

这个代码包包含两条流程和复用模块。AutoDL 上的 SLAKE 实验入口见 [SLAKE 手册](../slake_vqa/README.md)。

下方命令按 **同一个 AutoDL 终端**编写。换终端后先恢复第 2 节开头的变量，再恢复当前实验的 `CXR_MANIFEST`、`CXR_RUN` 和 `CXR_ADAPTER`。

## 2. 确认 GPU 环境和已有模型

```bash
cd /root/autodl-tmp/MedVQA_Evidence
umask 077
export CXR_PY=/root/autodl-tmp/medvqa-runtime/bin/python
export CXR_MAIRA_PY=/root/autodl-tmp/maira-runtime/bin/python
export CXR_QWEN="$PWD/models/Qwen3-VL-2B-Instruct"
export CXR_MAIRA="$PWD/models/maira-2"
export CXR_DATA="$PWD/data/cxr-openi"
export CXR_IMAGES="$CXR_DATA/images"
export CXR_RUN="$PWD/results/cxr-openi-smoke"
export CXR_ADAPTER="$PWD/checkpoints/cxr-openi-smoke"
mkdir -p "$CXR_IMAGES" "$CXR_RUN" "$CXR_DATA/tmp"
export TMPDIR="$CXR_DATA/tmp"
```

```bash
nvidia-smi
test -f "$CXR_QWEN/config.json" && ls -lh "$CXR_QWEN"/*.safetensors
test -f "$CXR_MAIRA/processor_config.json" && ls -lh "$CXR_MAIRA"/model-*-of-00006.safetensors
"$CXR_PY" -c 'import torch, torchvision, accelerate, transformers, peft; assert torch.cuda.is_available(); print(torch.__version__, torchvision.__version__, accelerate.__version__, transformers.__version__, peft.__version__)'
"$CXR_MAIRA_PY" -c 'import torch, transformers; from sentencepiece import sentencepiece_model_pb2; assert torch.cuda.is_available(); print("MAIRA environment OK", transformers.__version__)'
```

这里直接使用**你在 AutoDL 上下载好的 MAIRA-2**。如果 MAIRA 环境再次报 `gencode 7.36.1 runtime 5.28.3`，先按原手册修复该环境。 [Protobuf 修复](../slake_vqa/REPRODUCTION.md#33-验证处理器和六个权重分片)

检查报告项目附加依赖。已有环境通过检查时跳到第 3 节：

```bash
"$CXR_PY" -c 'import fastapi, langgraph, qdrant_client, sklearn, rouge_score; print("Report dependencies OK")'
"$CXR_PY" -m cxr_report_evidence --help
```

有模块缺失时，先记录当前可用的 PyTorch 和 torchvision 配对，再安装：

```bash
"$CXR_PY" -c 'from importlib.metadata import version; print("torch=="+version("torch")); print("torchvision=="+version("torchvision"))' > "$CXR_RUN/torch-constraints.txt"
"$CXR_PY" -m pip install -c "$CXR_RUN/torch-constraints.txt" -r cxr_report_evidence/requirements.txt
"$CXR_PY" -c 'import torch, torchvision, peft, langgraph, rouge_score; assert torch.cuda.is_available(); print("Report dependencies OK")'
```

## 3. 准备 IU X-Ray 数据

### 3.1 下载报告、图像和正位名单

在 AutoDL 终端先下载小的 XML 报告包。命令成功后，再下载 PNG 图像包。正位名单来自 RATCHET 作者发布的 Open-i 视图筛选文件。 [NLM 官方报告包](https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz) · [NLM 官方图像包](https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz) · [RATCHET 视图说明](https://github.com/farrell236/RATCHET/blob/master/preprocessing/openi/README.md#dataset-statistics)

```bash
wget -c --timeout=30 --tries=3 -P "$CXR_DATA" \
  https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz
wget -c --timeout=30 --tries=3 -P "$CXR_DATA" \
  https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz
wget --timeout=30 --tries=3 -O "$CXR_DATA/view_frontal.txt" \
  https://raw.githubusercontent.com/farrell236/RATCHET/master/preprocessing/openi/view_frontal.txt
ls -lh "$CXR_DATA"/*.tgz "$CXR_DATA/view_frontal.txt"
```

如果 AutoDL 无法连接，点击上面的官方包链接，在 Windows 下载。正位名单也可以从 [原始名单](https://raw.githubusercontent.com/farrell236/RATCHET/master/preprocessing/openi/view_frontal.txt) 保存为 `view_frontal.txt`。若三个文件在 Windows 的「下载」目录，运行：

```powershell
Set-Location (Join-Path $env:USERPROFILE 'Downloads')
sftp medvqa
```

在 `sftp>` 下执行：

```text
cd /root/autodl-tmp/MedVQA_Evidence/data/cxr-openi
put NLMCXR_reports.tgz
put NLMCXR_png.tgz
put view_frontal.txt
bye
```

回 AutoDL 解压 PNG。报告包直接交给脚本读取。重复下载图像可继续用 `wget -c`。出现 `Unexpected EOF` 时，先把图像包续传完整再解压。

```bash
mkdir -p "$CXR_IMAGES" && tar -xzf "$CXR_DATA/NLMCXR_png.tgz" -C "$CXR_IMAGES"
find "$CXR_IMAGES" -type f -iname '*.png' -print -quit
```

最后一行应显示一张已解压的 PNG 的路径。目录里只有 `.tgz` 时，先解压。脚本会搜索 `--images` 下的子目录。如果 PNG 已经解压到其他路径，把 `CXR_IMAGES` 指到那个目录。

### 3.2 生成冒烟清单，选 24 个正位病例

`prepare-openi` 从每份 XML 读取 `Findings`、`Indication` 和 `parentImage`。每份报告只选一张有正位名单标记的 PNG。它用固定种子按报告 UID 建立约 80%／10%／10% 的训练、验证和测试池，再从各池取指定数量。IU X-Ray 这里按报告 UID 做隔离，并检查选中影像是否重复。

```bash
export CXR_MANIFEST="$CXR_DATA/manifest-smoke.json"
"$CXR_PY" -m cxr_report_evidence prepare-openi \
  --reports "$CXR_DATA/NLMCXR_reports.tgz" --images "$CXR_IMAGES" \
  --frontal-list "$CXR_DATA/view_frontal.txt" \
  --train 16 --validation 4 --test 4 --seed 42 --output "$CXR_MANIFEST" &&
"$CXR_PY" -m cxr_report_evidence check --manifest "$CXR_MANIFEST" --images "$CXR_IMAGES"
```

通过时显示 `train: 16`、`validation: 4`、`test: 4` 和样例图尺寸。`manifest-smoke.json` 的 `matched_reports` 记录配对且有 `Findings` 的报告数，`split_pool_sizes` 是抽样前各池大小。正式记录使用 IU X-Ray 的报告级划分和样本数。升级实验时沿用同一份图像、报告、名单和种子，保持划分池固定。

如果报 `No frontal PNG filenames match` 或提示找到 PNG 但无法配对，对照下面两组文件名。脚本按影像 ID 配对，大小写和末尾 `.dcm.png` 的格式差异可以兼容。如果 ID 完全不同，请核对图像包和名单是否对应同一套数据。只有 `prepare-openi` 成功后才会生成供 `check` 读取的清单。

```bash
printf 'CXR_IMAGES=%s\n' "$CXR_IMAGES"
find "$CXR_DATA" -type f -iname '*.png' | head -n 5
head -n 5 "$CXR_DATA/view_frontal.txt"
```

## 4. 两步 LoRA 冒烟测试

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$CXR_PY" -m cxr_report_evidence train \
  --manifest "$CXR_MANIFEST" --images "$CXR_IMAGES" --model "$CXR_QWEN" \
  --max-train 16 --max-val 4 --steps 2 --accumulate 2 --use-indication \
  --output "$CXR_ADAPTER"
ls -lh "$CXR_ADAPTER/adapter"
cat "$CXR_ADAPTER/training_summary.json"
```

`--steps 2` 让优化器更新两次，用来检查数据加载、梯度更新、验证和适配器保存。当前单卡每批 1 张图，`--accumulate 2` 表示每两批更新一次，所以冒烟阶段训练只用到约 4 个样本。`--use-indication` 将报告里的检查指征放在输入里，原图列和仅文本列都用同一份指征。模型只学习输出 `Findings`。

通过时可看到 `global_step: 2`，训练损失和验证损失是有限数，`adapter/` 里有 `adapter_model.safetensors`。第 5 节用这份权重验证重新加载和四组生成。正式指标使用[第 9 节按验证集选出的适配器](#9-正式训练与评估)。训练中断时可将 `CXR_ADAPTER` 改成新目录重新开始。

## 5. 生成四组报告，先看两张测试片

```bash
"$CXR_PY" -m cxr_report_evidence generate \
  --manifest "$CXR_MANIFEST" --images "$CXR_IMAGES" --model "$CXR_QWEN" \
  --adapter "$CXR_ADAPTER/adapter" --split test --limit 2 --use-indication \
  --output "$CXR_RUN/reports.json" &&
"$CXR_PY" -m cxr_report_evidence metrics \
  --manifest "$CXR_MANIFEST" --reports "$CXR_RUN/reports.json" \
  --output "$CXR_RUN/report-metrics.json"
```

`reports.json` 每张片应有 `base_image`、`base_text`、`lora_image`、`lora_text`。参考报告留在 manifest 的 `reference`。模型会先跑 Base，再释放并加载 LoRA。同配置断线后可以重跑命令继续。如果修改输入、模型、样本数或解码条件，另取输出文件名。

如果两张胸片已输出四组报告，而 `metrics` 报 `No module named 'rouge_score'`，只给 Qwen 运行环境补装评测包，然后直接从已有的 `reports.json` 计算指标。 [rouge-score 安装说明](https://pypi.org/project/rouge-score/0.1.2/)

```bash
"$CXR_PY" -m pip install 'rouge-score==0.1.2' &&
"$CXR_PY" -m cxr_report_evidence metrics \
  --manifest "$CXR_MANIFEST" --reports "$CXR_RUN/reports.json" \
  --output "$CXR_RUN/report-metrics.json"
```

`report-metrics.json` 的 `by_split.test` 分四组记录 `rouge_l_f1`，这是文本一致性的检查。正式对照建议再算 `radgraph_f1_partial`。在装好官方 `radgraph` 的本地评测环境中执行同一条 `metrics`，添加 `--radgraph`，使用新的输出文件。首次评测需按官方说明准备其权重，依赖建议装在独立评测环境。 [RadGraph 官方包](https://github.com/Stanford-AIMI/radgraph)

## 6. 用 MAIRA-2 定位，再做两种模糊对照

### 6.1 在 AutoDL 上直接接着审计 100 例

在 AutoDL 的 Linux 终端逐段执行。以下命令直接写出解释器和结果路径，兼容仍要求完整参数的服务器版本。先确认两个解释器、正式清单和已生成的报告都在。

```bash
cd /root/autodl-tmp/MedVQA_Evidence
ls -l /root/autodl-tmp/maira-runtime/bin/python /root/autodl-tmp/medvqa-runtime/bin/python data/cxr-openi/manifest-formal.json results/cxr-openi-test100-gpu/reports.json
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

用 MAIRA-2 生成候选框。等待这条命令完整结束，再执行核查。

```bash
/root/autodl-tmp/maira-runtime/bin/python -u -m cxr_report_evidence ground --manifest data/cxr-openi/manifest-formal.json --images data/cxr-openi/images --reports results/cxr-openi-test100-gpu/reports.json --model models/maira-2 --device cuda --max-findings 4 --output results/cxr-openi-test100-gpu/ground.json
```

分别核查原图、目标模糊和同面积对照。核查完成后再算汇总。

```bash
/root/autodl-tmp/medvqa-runtime/bin/python -u -m cxr_report_evidence probe --manifest data/cxr-openi/manifest-formal.json --images data/cxr-openi/images --reports results/cxr-openi-test100-gpu/reports.json --grounding results/cxr-openi-test100-gpu/ground.json --model models/Qwen3-VL-2B-Instruct --device cuda --output results/cxr-openi-test100-gpu/probes.json
```

```bash
/root/autodl-tmp/medvqa-runtime/bin/python -m cxr_report_evidence summarize-probes --probes results/cxr-openi-test100-gpu/probes.json --output results/cxr-openi-test100-gpu/probe-summary.json
```

定位或核查中断时，用同一条命令和输出路径续跑。结束后下载整个 `results/cxr-openi-test100-gpu/`，将审计文件放到 Windows 的 `cxr_report_evidence/artifacts/results/cxr-openi-test100-gpu/`。重启本机网页即可查看候选框、模糊对照和分数。

如果定位停在 MAIRA-2 的 `assert end_index != -1`，说明该次生成的定位标记缺少结尾。新版会保留已完整输出的框。没有可恢复框的异常条目会记录为 `malformed_output`，其余条目继续处理。

服务器仍是旧代码时，在 Windows 的仓库根目录打这个小包：

```powershell
tar.exe -czf cxr-maira-grounding-fix.tar.gz cxr_report_evidence/audit.py evidence/maira.py evidence/perturbation.py
```

上传到 AutoDL 的 `/root/autodl-tmp/`，再在 AutoDL 提取：

```bash
cd /root/autodl-tmp/MedVQA_Evidence
tar -xzf /root/autodl-tmp/cxr-maira-grounding-fix.tar.gz
```

然后重跑上方原来的 `ground` 命令，保留同一个 `ground.json` 路径。从已经保存的条目之后继续，完成后再运行 `probe`。

### 6.2 审计其他批次

先在 MAIRA 专用环境定位。每份报告的前 4 条发现参与定位筛选，能做局部干预的发现才会送给 MAIRA，超出范围的条目仍会在页面上显示。这里的短语来自 Qwen 报告。调用 MAIRA 后进程自行退出，释放显存。

```bash
"$CXR_MAIRA_PY" -m cxr_report_evidence ground \
  --manifest "$CXR_MANIFEST" --images "$CXR_IMAGES" --reports "$CXR_RUN/reports.json" \
  --model "$CXR_MAIRA" --max-findings 4 --output "$CXR_RUN/ground.json"
```

接着用 Base Qwen 作共同核查器。同一个短语分别在原图、目标模糊图和同面积异位置对照图上查询，四种生成模式都用相同的复核协议。

```bash
"$CXR_PY" -m cxr_report_evidence probe \
  --manifest "$CXR_MANIFEST" --images "$CXR_IMAGES" --reports "$CXR_RUN/reports.json" \
  --grounding "$CXR_RUN/ground.json" --model "$CXR_QWEN" --output "$CXR_RUN/probes.json"
```

在 `probes.json` 查 `claim`、原图坐标 `boxes` 和 `checks`。`checks` 记录核查器在「支持／矛盾／证据不足」三个固定选项上的相对分值。`relative_support_drop` 是对照图支持分减去目标模糊图支持分。正值表示目标模糊使核查器给出更低的支持分。第 8 节按划分和报告方式汇总这个差值及覆盖率。扩大运行很慢时可减 `--max-findings`，并更换相关输出文件名。

例如，对同一句发现，原图支持分 0.85、目标模糊后 0.30、同面积对照后 0.80，局部差值就是 `0.80 - 0.30 = 0.50`。这表示核查器对目标区域的模糊更敏感。支持分来自模型对三个选项的相对打分，临床正确性的概率需要另外验证。

这一版实现报告陈述的事后共同复核，两种生成输入都可审计。CLEAR 原文还比较生成响应与医学图文对齐。本项目把响应改为固定选项复核，用于统一比较看图报告和仅文本报告。可选 `probe --alignment` 接入现有 BiomedCLIP 对齐实现。该分支需另外准备 `open_clip_torch`、BiomedCLIP 本地缓存，并为 `probes.json` 另取文件名。正常描述、依赖历史对照、无框和找不到匹配对照的陈述保留各自的审计状态。 [CLEAR 方法](https://arxiv.org/html/2610.38389v1#S3)

## 7. 查看网页和知识检索

先启动归档模式。它读取已经跑完的文件，网页请求只加载所选病例及预览。

```bash
"$CXR_PY" -m cxr_report_evidence.web \
  --manifest "$CXR_MANIFEST" --images "$CXR_IMAGES" --reports "$CXR_RUN/reports.json" \
  --probes "$CXR_RUN/probes.json" --qwen "$CXR_QWEN" --port 8010
```

再开一个 **Windows PowerShell** 设置 SSH 隧道，保持这个窗口打开：

```powershell
ssh -N -L 8010:127.0.0.1:8010 medvqa
```

浏览器打开 `http://127.0.0.1:8010`。选病例后点「查看报告」，同时看到四组报告和参考报告。「审计模型」和「审计输入」下拉框用于切换证据列表。选择一条发现，切换原图、目标模糊图和对照图。若暂时只有报告文件，启动时去掉 `--probes`，网页会显示已归档的报告数。知识问答可先输入 `How does a CT scanner reconstruct a slice?`，核对资料节选和来源。服务监听服务器的 `127.0.0.1`，通过 SSH 转发访问。


点开「怎么看支持分和对照差值？」查看解释。网页从 `probes.json` 读取三种输入各自的支持、矛盾、证据不足分数，并在有匹配对照时展示差值。

要同时开启本地 JPG／PNG 上传和 Qwen 检索增强回答，在 AutoDL 停掉前面的网页命令，改成：

```bash
"$CXR_PY" -m cxr_report_evidence.web \
  --manifest "$CXR_MANIFEST" --images "$CXR_IMAGES" --reports "$CXR_RUN/reports.json" \
  --probes "$CXR_RUN/probes.json" --qwen "$CXR_QWEN" --adapter "$CXR_ADAPTER/adapter" \
  --maira "$CXR_MAIRA" --maira-python "$CXR_MAIRA_PY" --rag-generate --port 8010
```

上传模式依次启动 Qwen、MAIRA-2 和核查器，避免两套模型同时常驻显存。每个上传请求生成 LoRA 的两份报告及候选框，临时数据在请求结束后清理。参考报告和 Base 对照可以在归档模式查看。上传模式使用统一指令，无检查指征。知识问答用独立语料。打开 `--rag-generate` 后，引用编号可核对时显示 Qwen 生成的带引用回答，其他情况显示检索原文。默认语料是项目自带的成像教学资料。

## 8. 汇总覆盖率与对照差值

当前实验以报告质量和反事实响应为结果。`report-metrics.json` 基于参考报告，`probe-summary.json` 基于模型在原图、目标模糊和同面积对照上的响应。两份汇总都可直接从已有产物计算。

```bash
"$CXR_PY" -m cxr_report_evidence summarize-probes \
  --probes "$CXR_RUN/probes.json" --output "$CXR_RUN/probe-summary.json"
cat "$CXR_RUN/probe-summary.json"
```

在 `by_split.test` 中分别读四组结果。`with_matched_control / findings` 是成功构建对照且得到核查分数的比例。分母包含超出定位上限、无框、无对照等条目，原因可查 `probe_status_counts`。平均和中位差值只统计有完整对照的条目。比较四组时一并报告覆盖率和样本量。现有数据支持描述核查器对扰动的反应。评估幻觉识别的 AP、AUROC 或给出风险概率需要独立的逐条视觉支持标注。

## 9. 正式训练与评估

下面复用第 3 节的图像、报告、正位名单和种子，从固定划分池抽 **1,600 / 100 / 100** 个训练、验证、测试病例。单卡每批 1 张图，`--accumulate 8` 每累计 8 张更新一次，所以每轮约 **200 步**。本次先给 **最多 10 轮**的训练预算，对应最多约 **2,000 次更新**。实际轮数和步数看 `training_summary.json` 的 `epochs_completed` 和 `global_step`。 [Transformers Trainer 的梯度累积说明](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)

`--early-stopping-patience 3` 每轮验证一次。**连续 3 轮的验证损失没有创下新低时提前结束**，最终 `adapter/` 保存验证损失最低的 checkpoint。例如第 5 轮最低，第 6–8 轮都没改善，就在第 8 轮停下并导出第 5 轮权重。`epoch_validation` 记录每轮曲线，`best_global_step` 和 `best_model_checkpoint` 记录选中的版本。如果跑到第 10 轮验证损失仍持续下降，可以在同一划分上用新的输出目录扩大预算。选完适配器后再用独立测试集做四组报告比较。 [Transformers 的早停说明](https://huggingface.co/docs/transformers/main_classes/callback#transformers.EarlyStoppingCallback)

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CXR_MANIFEST="$CXR_DATA/manifest-formal.json"

"$CXR_PY" -m cxr_report_evidence prepare-openi \
  --reports "$CXR_DATA/NLMCXR_reports.tgz" --images "$CXR_IMAGES" --frontal-list "$CXR_DATA/view_frontal.txt" \
  --train 1600 --validation 100 --test 100 --seed 42 --output "$CXR_MANIFEST" &&
"$CXR_PY" -m cxr_report_evidence check --manifest "$CXR_MANIFEST" --images "$CXR_IMAGES"
```

清单检查已显示训练 1,600、验证 100、测试 100 时，直接从这里开始训练。`mktemp -d` 为本轮创建一个空目录，并保留已有 checkpoint。若此前提示 `Choose a fresh --output directory for a training run`，训练在检查目录时已退出。紧接着手动 `cat` 看到的 `global_step: 2`、`train_studies: 16`、`validation_studies: 4` 属于冒烟产物。下面用 `&&` 确保训练成功后才查看新摘要。

```bash
mkdir -p "$PWD/checkpoints" &&
CXR_ADAPTER="$(mktemp -d "$PWD/checkpoints/cxr-openi-formal.XXXXXX")" &&
export CXR_ADAPTER &&
printf '本轮训练目录 %s\n' "$CXR_ADAPTER" &&
"$CXR_PY" -m cxr_report_evidence train \
  --manifest "$CXR_MANIFEST" --images "$CXR_IMAGES" --model "$CXR_QWEN" \
  --epochs 10 --accumulate 8 --early-stopping-patience 3 --use-indication --output "$CXR_ADAPTER" &&
cat "$CXR_ADAPTER/training_summary.json"
```

记下训练目录。确认新摘要的训练／验证数量是 **1,600／100**、`global_step` 记录本次更新步数、`best_model_checkpoint` 有值，再做评测。下方命令还会核对摘要的清单哈希，避免拿冒烟适配器做正式评测。`--split audit` 处理验证和测试两组。每轮评测用单独的结果目录，已有的 `reports.json` 会保留。

```bash
"$CXR_PY" -c 'import hashlib, json, sys; from pathlib import Path; summary = json.loads(Path(sys.argv[1]).read_text()); manifest_hash = hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest(); valid = (summary.get("train_studies"), summary.get("validation_studies"), summary.get("manifest_sha256")) == (1600, 100, manifest_hash) and summary.get("best_model_checkpoint"); sys.exit(0 if valid else "Training summary does not match the completed formal run.")' \
  "$CXR_ADAPTER/training_summary.json" "$CXR_MANIFEST" &&
mkdir -p "$PWD/results" &&
CXR_RUN="$(mktemp -d "$PWD/results/cxr-openi-formal-eval.XXXXXX")" &&
export CXR_RUN &&
printf '本轮结果目录 %s\n' "$CXR_RUN" &&
"$CXR_PY" -m cxr_report_evidence generate \
  --manifest "$CXR_MANIFEST" --images "$CXR_IMAGES" --model "$CXR_QWEN" \
  --adapter "$CXR_ADAPTER/adapter" --split audit --use-indication --output "$CXR_RUN/reports.json" &&
"$CXR_PY" -m cxr_report_evidence metrics \
  --manifest "$CXR_MANIFEST" --reports "$CXR_RUN/reports.json" --output "$CXR_RUN/report-metrics.json"
```

记下结果目录，继续第 6 至第 8 节的定位、核查、网页展示和实验汇总。生成中断后，用同一目录和相同参数重跑 `generate` 就能续跑。如果遇到 `reports.json has a different configuration`，先确认使用了本次正式训练的适配器。配置确实改变时，用新目录开始本轮评测。

| 要记录的结果 | 读取位置 | 通过标准 |
| --- | --- | --- |
| 报告 LoRA 训练 | `training_summary.json`、`adapter/` | 损失有限、适配器可重载、记录病例数、总更新步数、最佳 checkpoint 和每轮验证损失 |
| 四组报告质量 | `reports.json`、`report-metrics.json` | 同病例四组都有输出，给出独立测试四组 ROUGE-L／RadGraph F1 |
| 逐条定位与覆盖 | `ground.json`、`probes.json` | 有候选框、有匹配对照以及未覆盖情况分别计数，展示一个实际案例 |
| 扰动对照统计 | `probe-summary.json` | 四组各自报告有效覆盖率、缺失状态、对照减目标支持分的均值和中位数 |
| 可追溯网页 | 归档病例及所用代码版本 | 原图、四组报告和参考报告同屏，点击条目能看到审计框与扰动，RAG 有来源 |

正式训练、验证集部分报告和本机独立测试单例的结果见第 0 节。保留依赖版本、脚本、命令和允许分享的汇总。NLM 的 Open-i 说明要求原始数据留在研究组或机构内部，公开展示请遵循数据使用要求。 [NLM Open-i 使用说明](https://openi.nlm.nih.gov/faq#collection)

### 9.1 沿用正式权重，补齐四组评测和审计

本机已有 100 例独立测试的完整四组。若只继续测试集审计，从第 6 节开始，令 `CXR_RUN` 指向已生成的 GPU 测试结果目录。下方 `--split audit` 命令同时生成验证和测试两组，适合另做两划分的完整实验。

先按第 1 节同步代码。下面针对你的 AutoDL 上已存在的训练目录 `cxr-openi-formal.xlu8LK`。先在原 JupyterLab 终端执行第 2 节变量定义，然后执行本段。SHA256 检查必须显示 `OK` 才创建新的结果目录。续跑时沿用第一次打印的 `CXR_RUN`，从 `generate` 开始执行。

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CXR_MANIFEST="$CXR_DATA/manifest-formal.json"
export CXR_ADAPTER="$PWD/checkpoints/cxr-openi-formal.xlu8LK"
mkdir -p "$PWD/results" &&
printf '%s  %s\n' f8c8b3a6cfdcdff8c76495c433ac3cb8250672f167fd0aa38e3da0219f3ad0da \
  "$CXR_ADAPTER/adapter/adapter_model.safetensors" | sha256sum -c - &&
  CXR_RUN="$(mktemp -d "$PWD/results/cxr-openi-formal-eval.XXXXXX")" &&
  export CXR_RUN && printf '本轮结果目录 %s\n' "$CXR_RUN"
```

生成时，`--split audit` 包含验证 100 例和测试 100 例，共 800 份报告。每例四份共用指令，只切换图像和适配器。

```bash
"$CXR_PY" -m cxr_report_evidence generate \
  --manifest "$CXR_MANIFEST" --images "$CXR_IMAGES" --model "$CXR_QWEN" \
  --adapter "$CXR_ADAPTER/adapter" --split audit --use-indication --output "$CXR_RUN/reports.json" &&
"$CXR_PY" -m cxr_report_evidence metrics \
  --manifest "$CXR_MANIFEST" --reports "$CXR_RUN/reports.json" --output "$CXR_RUN/report-metrics.json"
```

看 `report-metrics.json` 的 `coverage` 和 `by_split.test`。覆盖量应为 200 例，测试四格各 100 份。两个常用差值是看图时的 LoRA 增益 `lora_image − base_image`，以及 LoRA 内的图像增益 `lora_image − lora_text`。分数和差值都用同一指标、同一病例比较。RadGraph 可另跑，见第 5 节。

报告生成结束后，运行 [第 6 节定位与扰动](#6-用-maira-2-定位再做两种模糊对照) 和 [第 8 节汇总](#8-汇总覆盖率与对照差值)。依次保存 `ground.json`、`probes.json` 和 `probe-summary.json`。MAIRA-2 给短语候选框，CLEAR 思路用于构造目标与同面积异位置的模糊图，Qwen 核查器输出相对支持分和扰动差值。未取得人工视觉支持标签的情况下，报告覆盖率、支持分和差值即可。最后按第 7 节用同一个 `CXR_RUN` 启动网页。

如需用标准 `rouge-score` 在原始下载包重算第 0.2 节，只选两组有输出的条件，并明确接受部分配对：

```powershell
# 在仓库根目录运行
$archive = 'cxr_report_evidence/artifacts'
.\.venv\Scripts\python.exe -m cxr_report_evidence metrics `
  --manifest "$archive/data/cxr-openi/manifest-formal.json" `
  --reports "$archive/results/cxr-openi-formal-eval.8VY46V/reports.json" `
  --variants base_image base_text --allow-partial `
  --output cxr_report_evidence/artifacts/results/local-review/partial-base-rouge-score.json
```

## 10. 下载正式产物并关机

整个打包本次正式训练目录、正式结果目录和 IU X-Ray 数据目录，保留它们在项目中的相对路径。

| 打包目录 | 包含内容 |
| --- | --- |
| `$CXR_ADAPTER`，一般为 `checkpoints/cxr-openi-formal.XXXXXX/` | 整个训练目录，包括最终 `adapter/`、训练摘要和服务器仍保留的 `checkpoint-*` |
| `$CXR_RUN`，一般为 `results/cxr-openi-formal-eval.XXXXXX/` | 整个正式评测目录，包括报告、评价指标、定位结果、核查与汇总 |
| `$CXR_DATA`，一般为 `data/cxr-openi/` | 整个 IU X-Ray 数据目录，包括已解压 PNG、原始下载包、正式清单、视图名单 |

在 AutoDL 原终端执行。`CXR_ADAPTER` 和 `CXR_RUN` 指向本次正式实验实际打印的目录。`du -sh` 显示三个目录的大小。数据目录包含原始图像压缩包和解压后的图像，因此这份完整备份会更大。

```bash
cd /root/autodl-tmp/MedVQA_Evidence &&
du -sh -- "$CXR_ADAPTER" "$CXR_RUN" "$CXR_DATA" &&
CXR_PACKAGE="/root/autodl-tmp/cxr-openi-formal-$(date +%Y%m%d-%H%M%S).tar.gz" &&
tar -czf "$CXR_PACKAGE" -C "$PWD" \
  "${CXR_ADAPTER#"$PWD"/}" "${CXR_RUN#"$PWD"/}" "${CXR_DATA#"$PWD"/}" &&
tar -tzf "$CXR_PACKAGE" > "$CXR_PACKAGE.files.txt" &&
printf '压缩包 %s\n文件清单 %s\n' "$CXR_PACKAGE" "$CXR_PACKAGE.files.txt" &&
sha256sum "$CXR_PACKAGE"
```

从 JupyterLab 的 `autodl-tmp` 下载压缩包和 `.files.txt` 清单。Windows PowerShell 可用 `Get-FileHash -Algorithm SHA256` 核对下载包。关机前从清单确认正式训练摘要、适配器、`reports.json` 和 `probes.json` 已在包里。训练摘要应记录 1,600／100 张训练和验证图。`report-metrics.json` 的 `by_split.test` 中，各组报告数应为 100。

在本机选择一个单独的目录解压。启动归档网页时，`--manifest`、`--reports`、`--probes` 指向解压后的文件，`--images` 指向解压后的 `data/cxr-openi/images/`。本机查看预计算结果和对照图只需要网页依赖与图像文件。

## 附录 · 获得权限后接入 MIMIC-CXR

MIMIC-CXR 报告库与 MIMIC-CXR-JPG 图像库使用同一个已审核的 PhysioNet 账号。完成各项目的数据使用协议后，在符合其数据处理要求的环境继续。下方命令创建独立数据、结果和训练目录。 [报告库访问要求](https://physionet.org/content/mimic-cxr/2.1.0/#access) · [图像库访问要求](https://physionet.org/content/mimic-cxr-jpg/2.1.0/#access) · [PhysioNet 数据使用说明](https://physionet.org/news/post/llm-responsible-use/)

```bash
export CXR_DATA="$PWD/data/cxr-mimic"
export CXR_IMAGES="$CXR_DATA/jpg"
export CXR_MANIFEST="$CXR_DATA/manifest-smoke.json"
export CXR_RUN="$PWD/results/cxr-mimic-smoke"
export CXR_ADAPTER="$PWD/checkpoints/cxr-mimic-smoke"
mkdir -p "$CXR_IMAGES" "$CXR_RUN" "$CXR_DATA/tmp"
export TMPDIR="$CXR_DATA/tmp"
read -rp 'PhysioNet username: ' CXR_USER

wget -c --user="$CXR_USER" --ask-password -P "$CXR_DATA" \
  https://physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-reports.zip
wget -c --user="$CXR_USER" --ask-password -P "$CXR_DATA" \
  https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz \
  https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz

"$CXR_PY" -m cxr_report_evidence prepare \
  --metadata "$CXR_DATA/mimic-cxr-2.0.0-metadata.csv.gz" --splits "$CXR_DATA/mimic-cxr-2.0.0-split.csv.gz" \
  --reports "$CXR_DATA/mimic-cxr-reports.zip" --train 16 --validation 4 --test 4 \
  --output "$CXR_MANIFEST" --urls "$CXR_DATA/jpeg-smoke.txt"
wget -c -x -nH --cut-dirs=3 -P "$CXR_IMAGES" --user="$CXR_USER" --ask-password -i "$CXR_DATA/jpeg-smoke.txt"
"$CXR_PY" -m cxr_report_evidence check --manifest "$CXR_MANIFEST" --images "$CXR_IMAGES"
```

`prepare` 使用 MIMIC-CXR 的官方患者级划分，并为有 `Findings` 的检查选一张正位 JPG。下载完从第 4 节继续，训练并评测单独的 MIMIC-CXR 适配器。 [官方目录结构与部分下载](https://physionet.org/content/mimic-cxr-jpg/2.1.0/#usage-notes)
