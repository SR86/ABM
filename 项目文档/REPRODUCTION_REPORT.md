# 本次实现与验证记录

本次工作完成论文 ABM 主方法的代码实现和流程验证，**尚未完成论文表格的全量数值复现**。内置攻击范围为现有项目涉及的 BadNets、Blend；其他六种原论文攻击需导入外部生成数据，防御基线和全部消融未实现。

## 代码改动

- `abm/models.py`：补回真实残差连接，提供 CIFAR/GTSRB ResNet-18、ImageNet 子集 ResNet-34。
- `abm/method.py`：实现算法 1、式 (2) 的 NBAB 控制后门与抑制损失、式 (3) 的 logits MSE 蒸馏、式 (4) 的 logits 惩罚；冻结教师参数和 BatchNorm 状态。
- `abm/attacks.py`、`abm/data.py`：在原始像素域加触发器，懒加载投毒，排除原目标类计算 ASR，支持外部污染数据。
- `abm/runner.py`、`configs/`：保存有效配置、训练历史、数据与代码签名、隔离索引、检查点、真实指标和失败原因；支持完整阶段恢复。
- 原 `abm_experiment.py` 草稿备份到 `tmp/original/abm_experiment.py`；独立标签翻转示例保持原样。

论文未明确的 L2 归约、NBAB 学习率、A 的统计方式等均在 README 逐项说明。推理采用论文式 (4)，并记录其与作者源码硬阈值公式的差异；隔离过程不读取已知投毒索引。

## 已实际执行的检查

| 检查 | 结果 |
|---|---|
| 算法/残差/数据/教师不变等单元测试 | 24 项通过 |
| Python 编译、pip 依赖一致性 | 通过 |
| ResNet-18 / ResNet-34 输出形状 | 10 / 43 / 12 类检查通过 |
| 合成数据端到端流程 | `results/smoke_final/` 完成 |
| 固定种子与恢复 | 投毒/隔离抽样可复现，完整阶段恢复检查通过 |
| 改变配置后恢复 | 正确拒绝复用旧阶段 |
| 真实 CIFAR-10 BadNets 流程 | `results/cifar10_badnets_validation/` 完成 |
| 真实 CIFAR-10 Blend 流程 | `results/cifar10_blend_validation/` 完成 |

测试在项目 `.venv-abm` 的 Python 3.12、CPU 版 PyTorch 上执行，版本记录见 `requirements-tested.txt` 和 `reproduction_provenance.json`。没有运行 CUDA 数值测试。

## 真实数据验证的实际含义

使用官方 `data_batch_4` 和 `test_batch` 的固定分层子集，各 512 个训练样本、256 个干净测试样本，51 个投毒样本，排除原目标类后 ASR 分母为 230。两个完整归档成员先与 torchvision 的已知 MD5 核对，再读取；来源、索引和导出文件哈希保存在 `data/cifar10_validation/`。

本次真实验证使用 NBAB 10 轮、教师/学生各 2 轮；教师/学生学习率均为 0.001，属于缩短训练的流程验证。结果中 `validation_only=true`、`synthetic_data=false`。

| 攻击 | NBAB 首轮 / 末轮损失 | 候选数 / 选取数 | 选取样本投毒纯度，仅诊断 |
|---|---:|---:|---:|
| BadNets | 2.607881 / 0.824471 | 456 / 5 | 0% |
| Blend | 2.633614 / 0.835145 | 477 / 5 | 0% |

**该缩小实验没有验证出有效的后门防御效果**：教师 CA 约 10%，两种攻击防御前后 ASR 均为 100%。这组欠训练模型只证明真实数据流程能够完成、数值有限、指标可保存和教师保持不变，不能证明隔离质量或论文准确率。不得将这些结果或 README 中的论文参考值当成全量复现结果。

## 运行入口

```powershell
& .\.venv-abm\Scripts\python.exe -m pytest tests -q
& .\.venv-abm\Scripts\python.exe abm_experiment.py --smoke-test --output-dir results/smoke_final --resume
& .\.venv-abm\Scripts\python.exe abm_experiment.py --config configs/validation_badnets.json --resume
& .\.venv-abm\Scripts\python.exe abm_experiment.py --config configs/validation_blend.json --resume
```

全量 CIFAR-10 已重新下载至 `data/cifar10/` 并完成解压，归档 MD5 为 `c58f30108f718f92721af3b95e74349a`，与 torchvision 官方记录一致；完整加载检查得到 50000 个训练样本、10000 个测试样本，记录见 `data/cifar10/integrity.json`。原先根目录 `data/` 下的截断压缩包保持原样。

全量实验配置为 `configs/cifar10_badnets.json` 和 `configs/cifar10_blend.json`；本次未执行全量 50 轮教师训练和论文指标对照验收。GTSRB 与 ImageNet 的实际论文划分也尚未取得与验证。
