# 本次真实验证记录

软件测试：**51 passed，2 skipped，0 failed**。Python 3.12.14，PyTorch 2.15.0+cpu，Windows CPU。完整记录见 [summary.json](summary.json) 和 [pytest-results.xml](pytest-results.xml)。

已执行真实优化步骤、检查点保存/加载、原网格推理及原 CT 文件哈希保护；也执行了 SimpleITK 合成经典 DICOM 的真实写入/读取、HU、物理方向和规则层间距检查。合成病例不是临床训练队列，临时测试权重未作为可用模型打包。

- [doctor-cpu.json](doctor-cpu.json)：小网络前向/反向。
- [parity-cpu.json](parity-cpu.json)：CPU 自对照；`accelerator_validated=false`。
- [benchmark-cpu.json](benchmark-cpu.json)：记录网络、补丁、精度、线程数和同步耗时；仅测随机网络前向。
- [installed-wheel-doctor.json](installed-wheel-doctor.json)：wheel 构建后，安装到独立测试目录运行成功；测试没有从源码目录导入。

Skill 结构校验、本地文档链接检查和独立场景复核完成。源代码 wheel 已成功构建；SHA-256：`249eb9cadca6bacc105775798bce12a9c6280e4b872891017dc6a0af3db65856`。

未完成的验收：M5 Max / MPS 真机、CUDA 真机、旧 H5 数值等价迁移、真实 CT 临床准确率、真实 PyRadiomics 特征提取。两个跳过项均要求实际安装 PyRadiomics，不将模拟提取器的门禁测试算作真实提取成功。PyRadiomics 3.0.1 在本次 Python 3.12 下分别遇到缺少 NumPy 与旧 `SafeConfigParser` 构建错误，详见 [依赖说明](../docs/APPLE_SILICON.md#可选放射组学依赖)。
