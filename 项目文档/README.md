# Anti-Backdoor Model 论文复现

本项目依据 Chen 等人的 *Anti-Backdoor Model: A Novel Algorithm to Remove Backdoors in a Non-Invasive Way*（IEEE TIFS 2024，DOI：10.1109/TIFS.2024.3436508）实现 NBAB 隔离、教师训练、学生蒸馏和 ABM 推理。项目目录中的论文 PDF 是主要依据；[论文页面](https://ieeexplore.ieee.org/document/10620313)和[作者公开代码](https://gitee.com/dugu1076/ABM)用于核对。本次源码核对基于提交 `24908a4c1ce82c2df69b77cbcd21809a6e950654`，下载副本位于 `tmp/author_abm/`。

## 实现范围

- 支持 CIFAR-10、GTSRB 和使用者提供的 ImageNet 子集。
- 内置 BadNets 和 Blend 两种攻击，默认运行已有代码涉及的 CIFAR-10 实验。
- 其余原论文攻击（CBA、SIG、Refool、Bpp、Nar、ISSBA）可导入外部预先生成的污染数据，随后运行同一 ABM 防御流程。导入接口不等于实现这些攻击。
- 未将 NAD、ABL、BAB 基线及论文全部消融、混合攻击、all-to-all、压力测试、Grad-CAM 等实验冒充为已完成结果。
- `data_poisoning_experiment.py` 及已有 Iris/Digits 图片是独立的标签翻转演示，保持原样，不作为本论文实验结果。

“代码实现了主方法”“流程能够运行”与“数值复现论文表格”是不同结论。只有完整数据、足够训练和明确配置下实际产生的指标才可用于数值比较；烟雾测试或缩小数据集结果不代表论文精度。

## 安装与运行

在项目目录使用 Python 环境安装依赖；GPU 运行时按本机 CUDA 环境安装匹配的 PyTorch。以下命令中的 `python` 应指向安装了项目依赖的解释器。

```powershell
python -m pip install -r requirements.txt
python abm_experiment.py --help
python abm_experiment.py --smoke-test --output-dir results/smoke_final
python abm_experiment.py --config configs/cifar10_badnets.json
python abm_experiment.py --config configs/cifar10_blend.json
```

**本机运行**：已建立 `.venv-abm`，解释器为 Python 3.12。系统默认的 `python` 不一定是该环境，PowerShell 可直接运行：

```powershell
& .\.venv-abm\Scripts\python.exe abm_experiment.py --help
& .\.venv-abm\Scripts\python.exe abm_experiment.py --smoke-test --output-dir results/smoke_final
& .\.venv-abm\Scripts\python.exe -m pytest -q
& .\.venv-abm\Scripts\python.exe -m pip check
```

若 `results/smoke_final/` 已有本次实现产生的结果，检查已保存阶段时使用 `--smoke-test --output-dir results/smoke_final --resume`；重新完整运行可指定新的 `--output-dir`。实现源代码变化后，旧阶段不能直接恢复，需使用新目录。后续示例也可以将 `python` 替换为 `& .\.venv-abm\Scripts\python.exe`，无需激活环境或更改系统设置。当前安装为 CPU 版 PyTorch，`--device cuda` 仅适用于另行配置了 CUDA 版 PyTorch 且有可用显卡的环境；`--device auto` 默认自动选择。

配置文件使用 `snake_case` 参数名，CLI 使用连字符，例如 JSON 的 `nbab_lr` 对应 `--nbab-lr`。提供的配置已展开全部有效默认值，便于逐项审查；新配置中未填写的值由 `abm/runner.py` 的 `DEFAULTS` 补全。相对数据和输出路径以项目目录为基准。

烟雾测试使用合成随机图像，仅检查训练、隔离、蒸馏、推理及结果保存路径，不用于报告论文指标。它显式将三阶段各设为 1 epoch，将 NBAB/教师/学生学习率分别设为 `0.0001/0.001/0.001`，并调整数据量、batch size、控制/投毒/隔离比例；所有有效参数会写入运行配置。正式实验请使用完整数据及正式配置。

可先准备数据，再开始正式训练；`--download` 显式允许数据下载。

```powershell
python abm_experiment.py --config configs/cifar10_badnets.json --download --prepare-data
python abm_experiment.py --config configs/cifar10_badnets.json --device cuda
```

对真实数据执行小规模验证的示例：

```powershell
python abm_experiment.py --config configs/cifar10_badnets.json --max-train-samples 1000 --max-test-samples 200 --nbab-epochs 1 --teacher-epochs 1 --student-epochs 1 --nbab-lr 0.0001 --teacher-lr 0.001 --student-lr 0.001 --device cpu --output-dir results/cifar10_debug
```

这里将教师/学生学习率降为 `0.001` 属于调试调整，偏离论文的 `0.1/0.01`。NBAB 学习率在论文中未明确，默认工程选择为 `0.01`；此处为调试进一步降为 `0.0001`。配置及结果会保留所有有效值。小规模验证与正式结果使用不同输出目录。恢复已有正式运行时，应指定同一配置和输出目录：

```powershell
python abm_experiment.py --config configs/cifar10_badnets.json --output-dir results/cifar10_badnets --resume
```

`--resume` 跳过已完整保存的阶段；未完成阶段可从最后完整 epoch 检查点继续，当前未完成的轮次重跑。检查点保存优化器、调度器、随机状态和数据顺序状态，不提供 batch 级恢复。旧进程没有产生逐轮文件的阶段仍需从头重跑。不能用不同超参数或数据配置复用旧产物。此次兼容升级及对正在运行任务的生效范围见 [逐轮检查点说明](EPOCH_CHECKPOINTS.md)。

其他数据集和外部攻击入口：

```powershell
python abm_experiment.py --config configs/gtsrb_badnets.json
python abm_experiment.py --config configs/imagenet_subset_badnets.json
python abm_experiment.py --config configs/external.json
```

这些配置需要先提供对应数据，程序不会自动构造 GTSRB/ImageNet 划分或生成其他攻击。主入口为 `abm_experiment.py`，实际流程位于 `abm/runner.py`；原始 ABM 草稿备份在 `tmp/original/abm_experiment.py`。

## 方法与论文公式对应

### 1. NBAB 数据隔离：算法 1、式 (2)

令不可信训练集为 `D`，原类别数为 `C`。验证模型具有 `C+1` 个输出，新增类标记为 `C`。只对一部分训练样本添加防御方可控触发器并改为新增类，其余样本保留原标签。CIFAR-10/GTSRB 的控制触发器位于左上角、大小为 `2×2`；ImageNet 子集为 `16×16`。

```text
原始分支：L_clean = CE(v(x), y)
控制触发分支：L_control = CE(v(x_tri), C) + alpha * L2(theta, theta_tri)
```

`x` 是添加控制触发器之前的不可信样本，其中仍可能含攻击者后门；防御方没有真实干净集可用。`x_tri` 在同一样本上额外叠加控制触发器。本实现把论文“softmax 前一层”的 `theta` 解释为分类 logits，使用按类别取均值的平方误差实现 L2；两侧均参与求导。

默认配置 `nbab_l2_reduction="batch"` 按式 (2) 的分段逐样本损失在全 batch 上取均值。若全 batch 有 `B` 个样本，其中 `n_control` 个添加了控制触发器：

```text
L_NBAB = CE_all + alpha * (n_control / B) * MSE_control
```

`MSE_control` 在控制子集的样本和类别两个维度取均值。可选 `nbab_l2_reduction="triggered"` 则使用 `CE_all + alpha * MSE_control`，用于研究不同归约的影响。论文没有完整指定 L2 的内部归约，因此不能将相同 `alpha` 数值等同于任意其他 L2 实现；更改该配置须在结果中保留。

训练完成后，对全部 `D` 临时叠加控制触发器。预测不是新增类的样本成为隔离候选，从中选取低容量隔离集。保存供学生训练的是候选的原始不可信样本，不能残留防御方的控制触发器。隔离决策不能读取已知投毒索引或真实投毒标记；这些标记只用于事后评估筛选纯度。

### 2. 教师与激活阈值：IV.B

教师在原始不可信训练集上进行正常分类训练，不在 NBAB 重新标记的数据上训练。教师训练完成后冻结参数及 BatchNorm 状态。

论文仅将 `A` 描述为教师输出神经元的平均激活，没有给出完整统计式。默认参考作者 `Util.py` 的标签激活标量定义：

```text
A = (1 / len(D)) * sum(teacher_logits(x_i)[y_i])
```

这里 `y_i` 是不可信数据提供的标签，不是隐藏的真实干净标签。本实现使用最终教师在 `eval()` 状态下遍历完整训练集重新计算 `A`；作者代码则在最后一个训练 epoch 内在线累加，二者存在差异。逐类平均输出向量可作为可选实验设定，不能与默认标量实验混报。

### 3. 学生知识蒸馏：式 (3)

```text
L_student = MSE(student_logits(x), teacher_logits(x))
```

学生从随机初始化开始，仅在小规模隔离集上训练；教师使用 `eval()` 和无梯度前向。论文式 (3) 将损失简写为 `l(Mt(x), Ms(x))`，未明确具体形式。MSE 原始 logits 的选择有作者 `train_Model_distill_one` 代码支持，未额外引入 softmax 温度或标签交叉熵。

### 4. 非侵入式推理：式 (4)

```text
output = teacher_logits - delta * relu(student_logits - A * gamma)
```

这一步直接输出 `[batch_size, C]` 的类别 logits；不能把 512 维池化特征与类别 logits 相减，也不能对结果再次应用教师分类层。防御阶段不更新教师参数。

## 论文设置与实现约定

| 设置 | CIFAR-10 | GTSRB | ImageNet Subset |
|---|---:|---:|---:|
| 类别数 | 10 | 43 | 12 |
| 论文训练/测试样本 | 50000 / 10000 | 22770 / 3870 | 5760 / 1440 |
| 主干 | ResNet-18 | ResNet-18 | ResNet-34 |
| `alpha` | 1000 | 2000 | 1000 |
| `delta` | 2.5 | 2.5 | 2.5 |
| `gamma` | 0.8 | 0.8 | 1.0 |
| 控制触发器大小 | 2×2 | 2×2 | 16×16 |

论文 V.A.3 指定 NBAB 验证模型 SGD 训练 10 epoch；教师 SGD 训练 50 epoch，初始学习率 0.1、weight decay 为 `1e-4`，每 2 epoch 学习率除以 10；学生 SGD 训练 20 epoch，学习率 0.01。这里忠实保留“每 2 epoch 降 10 倍”的文字设置，即使该设置会非常快地降低学习率。正式配置不使用数据增强。

论文 Table II 的 BadNets/Blend 使用目标类 0、投毒比例 10%。BadNets 为白色方块；本实现参考作者代码在 CIFAR-10 的 `27:29, 27:29` 区域写入 `2×2` 白块，与左上角 NBAB 控制触发器分开。Blend 使用随机像素图案，混合系数 0.2，像素处理在归一化之前完成，避免白色触发器在归一化后取值错误。

默认 Blend 固定一张由种子生成的灰度 `uint8` 随机掩码并在三通道共享，符合 Table II 的 Fixed/Random Pixel 描述。作者公开代码在每个样本上重新生成随机掩码；可通过配置选择这种逐样本模式，结果应注明模式。

论文没有明确给出 NBAB 控制注入比例、NBAB 学习率、batch size 和全部 SGD 参数。本实现将这些作为显式配置；默认控制注入比例 0.1、NBAB 学习率 `0.01`、batch size 120、SGD momentum 0.9、weight decay `1e-4`。控制比例和 batch size 120 有作者配置支持；NBAB 学习率 0.01 参考作者验证模型学习率，但优化器按论文采用 SGD，区别于源码 Adam。ImageNet 子集配置使用 batch size 8，以控制 `224×224` 输入的资源占用。NBAB 学习率及其他未明确参数属于公开的工程选择，不能把它们误称为论文指定值。改变这些值可能影响隔离浓度和最终指标。

数值诊断中，`nbab_l2_reduction="triggered"` 配合 `nbab_lr=0.1` 曾出现损失爆炸和非有限值。当前默认采用全 batch 分段期望归约及较小 NBAB 学习率，保留论文的 `alpha=1000` 和 SGD 10 epoch。该诊断说明未明归约与学习率的组合会影响训练，不构成“论文无法复现”的结论。程序检测到非有限损失会报错，并写入失败原因；可通过 `--nbab-lr` 显式研究学习率，结果会记录选定值。

论文常用隔离集规模为训练集 1%。候选不足时不能用真实投毒标记补齐；应保留实际候选数量或明确报告无法继续，不能伪造高纯度隔离集。

## 数据准备与评估口径

正式实验使用真实图像数据。CIFAR-10 可从本地数据加载；现有压缩包需完整且通过数据集完整性检查，损坏或未下载完的文件不能直接视为完整数据。GTSRB 的完整官方划分与论文 Table I 所报样本数不同，使用官方全量划分时应记录实际数量，不能声称完全匹配论文。ImageNet 子集需由使用者提供固定的 12 类训练/测试划分；论文未列出具体类别和抽样索引，仅凑足相同数量不足以保证同一实验。

默认 CIFAR-10 根目录是 `data/cifar10/`，需在其中有完整 `cifar-10-batches-py/`，或使用 `--download --prepare-data` 下载并检查。GTSRB 和 ImageNet 子集通过 ImageFolder 加载：

```text
data/gtsrb/train/<class>/<image>
data/gtsrb/test/<class>/<image>
data/imagenet_subset/train/<class>/<image>
data/imagenet_subset/test/<class>/<image>
```

训练和测试的类别目录映射必须一致，分别要求 43 类和 12 类。GTSRB 统一 Resize 为 `32×32`，ImageNet 子集为 `224×224`；然后仅转为 `[0,1]` 张量，不做 Normalize、随机裁剪或翻转。小图像主干采用 `3×3` stem、无初始 max pooling；ImageNet 子集采用 `7×7` stride-2 stem 和 max pooling。这些预处理/网络适配是可审查的实现选择，论文未完整列出。

外部污染数据应由可信生成流程提供，并包含可对齐的训练图像、训练标签、干净测试数据和触发测试数据。导入后仍需验证图像形状、像素范围、标签范围及样本数量，并记录攻击名称和生成设置。外部数据导入只负责加载与防御，不会验证攻击算法是否等价于原论文。

外部触发测试集应提供 `attack_test_original_labels`，据此排除原本为目标类的样本。若无法提供该字段，导出方必须事先排除这些样本，并在 `.pt` 字典中明确写入 `attack_test_non_target_only=True`；不能把攻击后已被替换的目标标签当作原标签过滤。当前外部接口评估 all-to-one 攻击，要求 `attack_test_labels` 全部等于配置的 `target_label`；不直接支持 all-to-all ASR。

外部 `.pt` 文件为仅含张量等基础类型的字典，通过 `torch.load(..., weights_only=True)` 加载，不能保存自定义 Dataset/模型对象。必需字段如下：

| 字段 | 含义 |
|---|---|
| `train_images` / `train_labels` | 已污染训练集及其训练标签 |
| `clean_test_images` / `clean_test_labels` | 干净测试集及原始标签 |
| `attack_test_images` / `attack_test_labels` | 已添加攻击触发器的测试集及攻击目标标签 |

图像张量为 CPU 上的 `[N, 3, H, W]` RGB，类型为 `uint8`（0–255）或浮点（0–1）；标签为 `[N]` 整数，范围为 `[0, num_classes)`。不接受 NHWC、NaN、无穷值或已经 Normalize 的输入。可选 `poison_indices` 是训练样本索引，仅用于隔离纯度诊断；缺失它不会阻止防御训练。外部数据不会再次叠加内置攻击触发器。

`configs/external.json` 默认读取 `data/external_attack.pt`，使用 `dataset="external"`、`attack="external"`、`model_dataset="cifar10"`。导入其他数据集攻击时，应同时设置 `num_classes`、`image_size`、`model_dataset` 和该数据集的 ABM 参数；例如 ImageNet 子集设置 `model_dataset="imagenet_subset"`、`num_classes=12`、`image_size=224`、`control_trigger_size=16`、`gamma=1`，以选择 ResNet-34。攻击名称和生成参数应另随数据保留。

- **CA**：干净测试集上的原始类别准确率。
- **ASR**：攻击触发测试集上预测为攻击目标标签的比例。内置 all-to-one 攻击排除原本就是目标类的测试样本，避免正常预测抬高成功率。
- 论文只把 ASR 统称为后门测试集分类准确率，未完整说明是否排除目标类；本项目在比较时需注明上述分母口径。
- 同时记录教师和 ABM 的 CA/ASR、隔离候选/实际数量，以及在有模拟真值时事后计算的隔离纯度；不能根据测试集指标挑选最佳种子后仅报告该次结果。

## 论文参考值与数值验证

下表来自论文 Table III，是对照资料，不是本项目在本机运行得到的结果。指标单位均为百分比。

| CIFAR-10 攻击 | 无防御 ASR | 无防御 CA | ABM ASR | ABM CA |
|---|---:|---:|---:|---:|
| BadNets | 97.98 | 79.31 | 0.82 | 80.71 |
| Blend | 99.90 | 80.56 | 0.13 | 81.50 |

验收顺序为：公式/张量维度和教师冻结检查 → 小规模流程验证 → 完整数据正式训练 → 与论文参考值逐项比较。若没有运行完整正式训练，不能将上述参考值写入实际结果文件，也不能声称已达到论文指标。

## 输出与恢复

每次运行写入配置的 `output_dir`；已存在同一运行时需使用 `--resume` 或另选目录。

| 文件 | 内容 |
|---|---|
| `config.json` | 合并默认值与 CLI 后的有效配置 |
| `data_signature.json` | 实际训练/测试张量内容签名及实现源代码哈希，恢复时检查数据和实现变化 |
| `training.jsonl` | 逐 epoch 的阶段、损失及学习率 |
| `isolation.json` | 候选数、选取索引、实际隔离率及可用时的诊断纯度 |
| `nbab.pt` / `teacher.pt` / `student.pt` | 完整阶段权重及历史，教师文件另含 `A` |
| `nbab_epoch.pt` / `teacher_epoch.pt` / `student_epoch.pt` | 每轮完整训练状态及上一份 `.prev.pt` 备份 |
| `abm.pt` | 组合教师、学生及阈值的状态 |
| `results.json` | 状态、环境、有效样本量、CA/ASR、教师不变检查和错误信息 |
| `metrics.png` | 本次实际运行的教师/ABM 指标图 |

`results.json` 中 `synthetic_data=true` 表示合成数据烟测；`validation_only=true` 表示合成数据或使用了数据量限制。正式配置运行失败会保留 `status="failed"` 及原因，不会生成虚构成功指标。恢复操作要求有效配置、数据签名及实现哈希一致；若首次训练使用了 CLI 覆盖或 `--download`，恢复时也需传入相同覆盖值。结果文件中的参数和状态比目录名称更可靠。

## 论文与作者源码的差异

公开源码可以补充论文未写清的实现细节，但不是与论文完全相同的可执行规范。本项目优先采用论文公式和实验文字，并明确以下差异：

1. 作者 `Main_poision_Bab.py` 使用 `resnet(depth=20)`，公开目录缺失对应 `resnet.py`；论文主实验规定 ResNet-18/34。
2. 作者主脚本使用 5% 攻击投毒率，Table II 为 10%。
3. 作者验证/学生优化器为 Adam，轮数和学习率计划也与论文 V.A.3 不同；本项目正式配置按论文 SGD 设置。
4. 作者 `test_two_Model` 使用系数 2 的硬阈值门控 `ft - 2*fs*gate(fs>A*gamma)`，不同于论文式 (4) 的 `ft - delta*relu(fs-A*gamma)`；本项目按式 (4)。
5. 作者压制损失使用中间返回值 `near`，其 `resnet.py` 缺失，无法确定其对应层。本项目依据论文“softmax 前一层”使用 logits，并明确 L2 的均值归约和双侧梯度。
6. 作者的 `A` 统计方式、Blend 掩码生成方式与本项目默认设定的具体差异已在前文说明。
7. 作者公开代码依赖额外 trigger/图像等资源，不能把未取得的资源或缺失模型定义视为已验证可执行。

此外，论文未给出足以唯一确定所有数值的训练随机性、归一化方式、ImageNet 子集类别、L2 归约、`A` 统计式及隔离候选取样顺序。结果偏差应结合这些因素如实报告，不能通过读取模拟攻击真值、改写参考值或隐式切换训练计划消除差异。

## 本次实际验证记录

详见 [REPRODUCTION_REPORT.md](REPRODUCTION_REPORT.md)。已执行 24 项单元测试、合成烟测及 BadNets/Blend 的真实 CIFAR-10 小规模流程验证。小规模结果没有验证出有效防御效果，不能当作论文指标。

真实数据验证可用以下命令复查；来源和 MD5 见 `data/cifar10_validation/provenance.json`：

```powershell
python scripts/prepare_cifar_validation.py
python abm_experiment.py --config configs/validation_badnets.json --resume
python abm_experiment.py --config configs/validation_blend.json --resume
```

本次已重新下载并解压完整 CIFAR-10 到 `data/cifar10/`，归档 MD5 与 torchvision 记录一致，完整加载检查为 50000/10000 样本。完整性记录见 `data/cifar10/integrity.json`；原有截断压缩包保持原样。全量训练配置已提供，本次尚未运行全量论文指标验收。

最新全量实验已于 2026-11-19 完成，数值未达到论文指标，见 [全量实测结果](FULL_RESULTS.md)。此前“尚未全量训练”描述为历史记录。
