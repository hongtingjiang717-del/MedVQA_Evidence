# 本机 IDE 环境准备

在 VS Code 或 PyCharm 打开整个 `MedVQA_Evidence` 仓库。下面的命令在仓库根目录执行。

## Qwen 和网页环境

有 `.venv` 就沿用。首次创建时运行：

```powershell
py -3.12 -m venv .venv
```

把 IDE 的解释器选为仓库里的 `.venv\Scripts\python.exe`，再打开一个新终端。第一行打印当前使用的 Python 路径。只用 CPU 时，第二行安装 CPU 版 PyTorch。使用本机 NVIDIA GPU 时，按官方页面选择与你的驱动匹配的 CUDA 版，将安装命令中的 `pip` 写成 `python -m pip`。 [PyTorch 本机安装](https://pytorch.org/get-started/locally/)

```powershell
python -c "import sys; print(sys.executable)"
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r slake_vqa/requirements.txt
```

仅浏览归档或试知识检索时，可只运行 `python -m pip install -r evidence/requirements-app.txt` 安装网页依赖。已有 Qwen 时把模型放在 `evidence/models/Qwen3-VL-2B-Instruct/`。首次下载可用：

```powershell
.\.venv\Scripts\hf.exe download Qwen/Qwen3-VL-2B-Instruct --revision 89644892e4d85e24eaac8bacfd4f463576704203 --local-dir evidence/models/Qwen3-VL-2B-Instruct
```

训练 LoRA 之前可用 `python -c "import torch; print(torch.cuda.is_available())"` 检查 CUDA 是否为 `True`。 [固定版本模型文件](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/tree/89644892e4d85e24eaac8bacfd4f463576704203)

## MAIRA-2 环境

只有运行胸片定位时才需要另建环境。

```powershell
py -3.12 -m venv .venv-maira
```

在这个环境安装 CUDA 版 PyTorch 与 torchvision。按 PyTorch 官方页面选择后，将命令里的 `pip` 写成 `.\.venv-maira\Scripts\python.exe -m pip`。然后安装项目验证过的 MAIRA-2 依赖：

```powershell
.\.venv-maira\Scripts\python.exe -m pip install -r evidence/requirements-maira.txt
```

已有权重时放进 `evidence/models/maira-2/`。首次下载先在模型页申请访问，再登录命令行下载。 [MAIRA-2 模型卡和许可](https://huggingface.co/microsoft/maira-2)

```powershell
.\.venv-maira\Scripts\hf.exe auth login
.\.venv-maira\Scripts\hf.exe download microsoft/maira-2 --local-dir evidence/models/maira-2
```
