# 作者缺失 ResNet 的兼容候选

`resnet.py` 是重建候选，**不是找回的作者原文件**，不接入当前正式训练。当前 `abm/models.py` 的 ResNet-18/34 保持原样。

已拉取 ABM 作者公开仓库完整 master 历史，六个提交没有提供 resnet.py。公开脚本要求 `from resnet import resnet`、`resnet(depth=20,num_classes=...)`，并解包四个返回值。

论文引用的 NAD 官方仓库提供相同工厂函数、depth 参数与四输出接口，作为可验证候选参考。固定提交与来源见 reference.json。不能仅凭接口一致断言 ABM 使用了 NAD 的同一文件。

候选结构为 CIFAR BasicBlock ResNet-20：3x3 stem，16/32/64 通道，三个阶段各三个残差块，降采样使用1x1投影；输出为三个阶段的特征图和类别logits。默认第三输出为 `[B,64,8,8]`，与 NAD 相同。另提供 near_mode='pooled' 和 'logits'，仅用于区分尚未确定的层选择假设。

作者 NBAB 工具对第三个返回值 `near` 进行平方差抑制。如果它实际对应 NAD 特征图，那么它与当前论文主实现的 logits 抑制不是同一目标，不能共用同一 α 和归约后假设数值等价。这是后续对照实验需要重点核实的变量，尚不能认定为当前复现失败的唯一原因。

```python
from author_compat.resnet import resnet
model = resnet(depth=20, num_classes=11)
activation1, activation2, near, logits = model(images)
```

四项测试验证10/11类输出、作者损失调用及梯度，见 tests/test_author_resnet.py。此外将固定版本 NAD 参考权重严格加载到候选模型后，同一输入的四个输出在 CPU 上逐位一致，见 verification.json。该验证说明候选与参考接口/结构兼容，不说明它与缺失 ABM 原文件或论文最终实验配置完全相同。

使用这个候选仍不能直接跑完整作者脚本：该脚本还有 CUDA 写死、额外数据/触发资源、与论文不同的优化器和推理公式等问题。不要把 resnet20 命名成论文的 ResNet18，也不要直接加载现有 ResNet18 检查点到此模型。
