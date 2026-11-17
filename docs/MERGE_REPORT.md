# 两份 MedVQA 项目整理说明

## 合并方式

原 medvqa_evidence_mvp 与 MedVQA-Evidence_GitHub_Release 已合并为 MedVQA_Evidence。

8 份公共核心文件经字节比较完全一致。Release 中的最终评测与 Demo 作为正式入口；MVP 全部25个文件在移动前后核对 SHA-256 一致，保留在本机 .local_archive/medvqa_evidence_mvp，该目录不加入 Git。

- app.py：最终 Gradio 页、SLAKE 示例与150条结果看板。
- evaluate.py：最终 validation 配对评测与图像簇 Bootstrap。
- day5_app.py / day4_evaluate.py：旧命令的兼容入口。
- archive/early_app.py / archive/early_evaluate.py：早期基础 Demo 和三组探索代码；不是推荐入口。
- docs/history/：两份原 README 历史文字，Release 本地链接已调整。
- results/day4_validation_summary.json：保留150条核心汇总原值。
- results/exploratory_50_summary.json：从 MVP paired_statistics.json 原样复制的50条探索汇总，独立解释。
- 原始演示图、逐题复核 CSV、个人编辑资料留在被忽略的本机归档中，不上传。

## 修改内容

统一主入口，更新目录、运行和 GitHub 文档。修复 percentile 辅助函数的单样本/最大分位点边界：原实现会因两个权重均为0而返回0。增加离线测试，覆盖分位数、图像聚类统计、重复记录拒绝、部分完成状态和提示模板不一致。

历史150条汇总值没有重算、没有替换；当前 Bootstrap 默认3000次、q=0.025/0.975 不受所修边界触发。本次不把函数边界修复说成模型性能提升。

## 验证记录

本次在本机执行离线检查，不下载 Hugging Face 模型或数据，不开展 GPU 推理。

- 代码语法与主 README 本地链接检查。
- 原有4个指标测试，加8个评测逻辑测试，共12项。
- 新主入口 app.py --check，以及旧 day5_app.py --check。
- 新旧评测入口 --help。
- 使用已有本机 Gradio 环境构建最终页面配置，不启动端口、不触发模型/数据下载。
- 150条汇总的样本数、EM、配对转移与增益内部一致性检查。
- 50条探索汇总与本机复核 CSV 的问题数、图像数、EM/F1和配对计数核对；逐题表不发布。

本次未重新运行150条实验、未重算其区间、未独立验证历史 AutoDL 环境或跨集合哈希审计。完整数据和原始150条预测不在两份文件夹中，公开汇总本身不能恢复逐题记录。

## 解释限制

50条和150条属于不同阶段；不能混用图像数、区间或将两者视为独立重复实验。早期 Wrong-image 缺少替换图像来源；旧记录若缺图像/提示哈希，复用检查只能依据已有索引、模型、划分、问题和标签，不能完全核实过去的输入版本。

历史 GPU 运行及哈希审计是已有报告记载，本次只验证当前可离线核实的内容。EM/Token-F1是字面答案指标，不是临床诊断准确率。
