# 第二轮：仅扫描 logits 抑制系数

依据第一轮发现：无抑制时控制触发可以泛化，但当前抑制设置下失败。本轮固定 NAD 兼容候选ResNet20、logits层、SGD及原数据/身份划分/初始化/批次顺序/训练轮数，只改变MSE项系数α。

预定网格：0、0.01、0.1、1、10、100、1000。BadNets、Blend各7组；每组8000训练身份、2000独立验证身份、5epoch，CPU8线程。其他设置不变。α=0与α=1000是两端重复对照，用来核查与第一轮结果的可比性。较小α属于诊断消融，不能冒充论文α=1000的正式配置。

输出目录为results/nbab_strength_scan。判读联合使用干净控制触发泛化、候选纯度和投毒召回，不以单个指标或官方测试集最优值选模型。若出现有希望的中间区间，需增加轮数、验证种子并保留原始失败配置，再考虑全量验证。

保留每轮检查点、源码/配置签名、分量损失和初始梯度量级。结束生成COMPARISON.md；Gotify进度及成功/失败报告使用此目录独立记录。旧诊断结果不覆盖。

```powershell
.\.venv-abm\Scripts\python.exe scripts/nbab_diagnostics.py --strength-scan --output results/nbab_strength_scan
```
