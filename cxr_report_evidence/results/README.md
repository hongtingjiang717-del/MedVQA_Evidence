# IU X-Ray 结果索引

| 阶段 | 已保存结果 | 待补 |
| --- | --- | --- |
| 数据与划分 | [正位/报告筛选及正式清单哈希](data_inventory.json) | |
| 正式 LoRA | [2,000 步训练曲线、最佳 checkpoint 和适配器哈希](lora_training.json) | |
| 历史验证归档 | [55 例仅 Base 的分数与覆盖率](base_validation_partial.json) | |
| 本机独立测试单例 | [同病例四组 ROUGE-L F1](cpu_example_metrics.json) | |
| 正式独立测试 | [100 例、400 份报告的四组 ROUGE-L 和配对差值](test100_gpu_metrics.json) | |
| MAIRA-2 与反事实审计 | [3,711 条发现、585 条有效对照、四组扰动差值](test100_gpu_audit.json) | 人工标签与风险校准 |
| 本机网页 | [报告、候选框、模糊对照和核查分数示例](http://127.0.0.1:8010/?study=iu-CXR227) | 允许公开的网页截图 |

JSON 中的 `source` 可用于在本机核对原始产物。原图、报告正文、清单和适配器位于 `cxr_report_evidence/artifacts/`。若发布病例截图，依据 Open-i 的使用条款选择允许公开的素材。 [Open-i 数据使用说明](https://openi.nlm.nih.gov/faq#collection)
