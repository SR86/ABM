# ABM 复现归档

2026-11-23：已停止继续复现。完整流程已运行，但未达到论文报告的防御指标。

- [复现历史、实验结果与排查结论](项目文档/REPRODUCTION_HISTORY.md)
- [项目说明与运行方法](项目文档/README.md)
- [完整项目快照下载](https://github.com/SR86/ABM/releases/tag/reproduction-archive-2026-11-23)

源码、配置、测试、报告与实验日志保存在 `项目文档/`。Release 附件包含完整本地项目快照，包括数据、模型检查点、参考源码、临时文件和本地Python环境；仅排除嵌套Git管理目录。下载全部 `project-part-*.zip` 并解压到同一目录。`FILE_MANIFEST.json` 提供每个文件所属分卷、大小和SHA256，`ASSET_CHECKSUMS.json` 提供各分卷校验值。

Windows虚拟环境包含原机器绝对路径；保留它们用于完整归档，迁移运行时应按依赖文件重新创建环境。62组诊断实验并不代表62次全量论文复现，具体范围见复现历史。
