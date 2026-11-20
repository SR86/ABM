# NBAB 单变量短实验结果

固定 NAD 兼容候选 ResNet20；仅使用 CIFAR 训练身份分割。不是原全量 ResNet18 实验，也不是论文数值复现。

|配置|状态|独立干净身份触发成功率|独立候选纯度|独立投毒召回率|
|---|---|---:|---:|---:|
|badnets_logits_alpha_0_sgd|complete|89.56%|23.27%|28.50%|
|badnets_logits_alpha_0.01_sgd|complete|89.17%|26.14%|34.50%|
|badnets_logits_alpha_0.1_sgd|complete|88.50%|31.68%|48.00%|
|badnets_logits_alpha_1_sgd|complete|72.56%|26.49%|89.00%|
|badnets_logits_alpha_10_sgd|complete|1.83%|10.17%|100.00%|
|badnets_logits_alpha_100_sgd|complete|0.28%|10.03%|100.00%|
|badnets_logits_alpha_1000_sgd|complete|0.06%|10.01%|100.00%|
|blend_logits_alpha_0_sgd|complete|94.33%|14.29%|8.50%|
|blend_logits_alpha_0.01_sgd|complete|86.28%|12.72%|18.00%|
|blend_logits_alpha_0.1_sgd|complete|87.22%|41.77%|82.50%|
|blend_logits_alpha_1_sgd|complete|79.44%|32.73%|90.00%|
|blend_logits_alpha_10_sgd|complete|2.67%|10.25%|100.00%|
|blend_logits_alpha_100_sgd|complete|2.11%|10.19%|100.00%|
|blend_logits_alpha_1000_sgd|complete|2.56%|10.24%|100.00%|

独立干净身份触发成功率、候选纯度与投毒召回率需联合判断；全部预测新增类可能导致零候选，不能算防御成功。
poison truth 仅事后评估；未用于训练损失或候选决策。短实验只用于定位问题，不能据此声称论文指标达成。
