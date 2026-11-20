# 逐轮检查点

NBAB、教师、学生每完成一轮保存 `nbab_epoch.pt`、`teacher_epoch.pt`、`student_epoch.pt`。检查点包含模型（含 BatchNorm 缓冲）、SGD 优化器及动量、教师学习率调度器、已完成轮数和历史、Python/NumPy/PyTorch/CUDA 随机状态、DataLoader 随机生成器状态。

文件先写入临时文件，flush/fsync 后原子替换；同时保留 `.prev.pt` 上一份备份。主文件损坏时尝试恢复备份。尚未完成的一轮需要重跑，不提供 batch 级恢复。断电后仍需手动启动程序。

```powershell
.\.venv-abm\Scripts\python.exe scripts/run_full.py
```

批处理会使用 `--resume`。也可单独恢复某项：

```powershell
.\.venv-abm\Scripts\python.exe abm_experiment.py --config configs/cifar10_blend.json --resume
```

已完成阶段文件优先使用；未完成阶段从逐轮文件继续。配置、数据与实现签名仍需匹配。本次升级仅对 `abm/epochless_implementation.json` 记录的精确旧实现允许检查点兼容迁移，并写入 `epoch_checkpoint_migration.json` 留痕；其他实现变动仍拒绝恢复。

本次更新不重启当前 BadNets 进程，因此本次 BadNets 仍按旧方式保存完整阶段；后续 Blend 子进程会自动加载新代码并逐轮保存。更新前的轮次没有逐轮文件，不能补回尚在内存中的权重。

保存操作不改变损失、优化器参数、训练轮数或数据顺序，仅增加磁盘写入。三个阶段均通过有 Dropout、随机打乱的 CPU 对照测试：连续四轮与两轮后重启恢复至四轮的权重、损失和随机状态精确相同。不同硬件、软件版本或 CUDA 环境不保证位级一致。全测试共 27 项通过。

Gotify 监控是独立进程；电脑重启后需要重新启动通知脚本，凭据仍从 Windows 凭据管理器读取。
