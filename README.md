# MedVQA Evidence

两个医学影像实验，共用 Qwen3-VL 基座、MAIRA-2 定位和本地知识检索。分别训练问答与报告适配器，评测各自的数据和任务。

| 子项目 | 已有实验结果 | 本机操作 |
| --- | --- | --- |
| SLAKE 影像问答与 CT 分割 | 150 道英文验证题上，看图 EM 从 47.33% 提高到 74.00%。左肺单类 U-Net 的小样本测试 Dice 为加权 0.5904、不加权 0.6289 | [README](slake_vqa/README.md) · [复现报告](slake_vqa/REPRODUCTION.md) · [结果](slake_vqa/results/) |
| IU X-Ray 胸片报告与证据审计 | 100 例独立测试中，看图 ROUGE-L F1 从 0.1159 提高到 0.3024。反事实审计记录了 3,711 条发现，585 条取得有效匹配对照 | [README](cxr_report_evidence/README.md) · [复现报告](cxr_report_evidence/REPRODUCTION.md) · [结果](cxr_report_evidence/results/) |

```text
MedVQA_Evidence/
├── slake_vqa/              SLAKE 代码、README、复现报告和公开结果
├── cxr_report_evidence/    IU X-Ray 代码、README、复现报告和公开结果
├── evidence/              公用模型加载、图像扰动、检索和依赖
└── tests/                 指标和定位的单元测试
```

本机命令从仓库根目录执行。基础权重放在 `evidence/models/`，两条流程各自用 `artifacts/` 存放训练权重、原数据、逐例结果和运行日志。GitHub 展示使用各子项目的 `results/` 指标汇总。 [公共环境安装](evidence/README.md)

## 方法参考

反事实审计参考 Zou 等人的 CLEAR 短语级视觉证据核查框架。本项目在 IU X-Ray 中使用 MAIRA-2 候选框，比较目标区与同面积对照区模糊后的 Qwen 核查分数。结果用于研究模型对图像扰动的反应。 [Visually Irrelevant Hallucinations in Medical Vision-Language Models，arXiv:2610.38389](https://arxiv.org/abs/2610.38389)
