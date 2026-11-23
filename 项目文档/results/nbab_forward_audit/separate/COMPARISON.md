# NBAB 单变量短实验结果

固定 NAD 兼容候选 ResNet20；仅使用 CIFAR 训练身份分割。不是原全量 ResNet18 实验，也不是论文数值复现。

|配置|状态|独立干净身份触发成功率|独立候选纯度|独立投毒召回率|
|---|---|---:|---:|---:|
|badnets_logits_alpha_0_sgd|complete|96.17%|10.39%|4.00%|
|badnets_logits_alpha_0.1_sgd|complete|98.67%|14.29%|2.00%|
|badnets_logits_alpha_1000_sgd|complete|4.72%|10.16%|97.00%|
|blend_logits_alpha_0_sgd|complete|93.72%|1.74%|1.00%|
|blend_logits_alpha_0.1_sgd|complete|95.89%|2.63%|1.00%|
|blend_logits_alpha_1000_sgd|complete|6.78%|10.36%|97.00%|

独立干净身份触发成功率、候选纯度与投毒召回率需联合判断；全部预测新增类可能导致零候选，不能算防御成功。
poison truth 仅事后评估；未用于训练损失或候选决策。短实验只用于定位问题，不能据此声称论文指标达成。
