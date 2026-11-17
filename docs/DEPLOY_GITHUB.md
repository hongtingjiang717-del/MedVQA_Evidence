# GitHub 更新与后续开发

目标仓库：[hongtingjiang717-del/MedVQA_Evidence](https://github.com/hongtingjiang717-del/MedVQA_Evidence)。本次是在已有 main 上追加整理提交，保留原历史，不重新创建仓库或强制覆盖历史。

## 在整理后的本机目录继续修改

```powershell
cd D:\desktop\sgg\MedVQA_Evidence
python -m unittest discover -s tests -v
python app.py --check
git status
git add -A
git commit -m "Update MedVQA project"
git push origin main
```

这些是以后修改后的常规命令。本次发布使用已授权 GitHub 工具创建文件树、创建以原 main 为父的新提交，再更新 main；不是通过 git push 完成的上传。

## 在其他机器获取项目

```bash
git clone https://github.com/hongtingjiang717-del/MedVQA_Evidence.git
cd MedVQA_Evidence
python -m unittest discover -s tests -v
python app.py --check
```

真正模型推理需要兼容的 CUDA/PyTorch 环境、requirements.txt 依赖、SLAKE 数据和 Qwen 模型；离线检查不会下载权重或数据。

## 提交范围

上传源码、测试、README、方法报告、演示截图、两份明确区分的聚合汇总。`.local_archive/`、`.resume_work/`、原始逐题预测和复核表、模型权重、原始数据与缓存均不上传。

如果之后确实要公开新的聚合统计，请先核对内容，再在 .gitignore 中单独放行；不要取消整个 results/ 或本机归档目录的忽略规则。
