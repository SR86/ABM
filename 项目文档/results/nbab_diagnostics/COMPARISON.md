# NBAB 单变量短实验结果

固定 NAD 兼容候选 ResNet20；仅使用 CIFAR 训练身份分割。不是原全量 ResNet18 实验，也不是论文数值复现。

|配置|状态|独立干净身份触发成功率|独立候选纯度|独立投毒召回率|
|---|---|---:|---:|---:|
|badnets_no_regularizer_sgd|complete|89.56%|23.27%|28.50%|
|badnets_logits_mse_sgd|complete|0.06%|10.01%|100.00%|
|badnets_pooled_mse_sgd|complete|0.56%|10.05%|100.00%|
|badnets_stage3_mse_sgd|complete|0.06%|10.01%|100.00%|
|badnets_stage3_sum2000_sgd|complete|0.00%|10.00%|100.00%|
|badnets_stage3_sum2000_adam|complete|0.61%|9.87%|98.00%|
|badnets_logits_mse_adam|complete|0.61%|9.96%|99.00%|
|blend_no_regularizer_sgd|complete|94.33%|14.29%|8.50%|
|blend_logits_mse_sgd|complete|2.56%|10.24%|100.00%|
|blend_pooled_mse_sgd|complete|0.61%|10.06%|100.00%|
|blend_stage3_mse_sgd|complete|0.00%|10.00%|100.00%|
|blend_stage3_sum2000_sgd|complete|0.44%|10.04%|100.00%|
|blend_stage3_sum2000_adam|complete|0.22%|10.02%|100.00%|
|blend_logits_mse_adam|complete|1.78%|10.16%|100.00%|

独立干净身份触发成功率、候选纯度与投毒召回率需联合判断；全部预测新增类可能导致零候选，不能算防御成功。
poison truth 仅事后评估；未用于训练损失或候选决策。短实验只用于定位问题，不能据此声称论文指标达成。
