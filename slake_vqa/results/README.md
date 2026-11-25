# SLAKE 结果索引

| 阶段 | 已保存结果 | 待补 |
| --- | --- | --- |
| 数据准备 | [官方划分与数据指纹](slake_inventory.json) | |
| 正式 LoRA | [4,919 题、615 步训练摘要](lora_training.json) | |
| 问答对照 | [150 题验证集四组指标](vqa_validation.json) | 独立测试四组 |
| 左肺分割 | [加权](unet_weighted.json)、[不加权](unet_unweighted.json) | 预测掩膜截图 |
| 四器官分割 | [宏平均与逐器官结果](unet_ct_multi_summary.json) | 同预算不加权对照 |
| 局部审计 | [MAIRA-2 定位统计](maira_cardiomegaly.json)、[两例扰动摘要](audit_examples.json)、[本机 CPU 核查](local_cpu_audit_smoke.json) | 网页证据面板截图 |
| 本地网页 | [离线功能检查](smoke_summary.json)、[检索对照](rag_summary.json) | 最新页面截图 |

部分 JSON 的 `source` 字段指向已下载在本机的原始产物。完整的样本级文件、标注和权重放在 `slake_vqa/artifacts/`。
