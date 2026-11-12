# Apple Silicon / M5 Max 运行与验收

本项目用 PyTorch 的 **MPS 后端调用 Mac GPU**。它面向 Apple Silicon，M5 Max 也走这条路径；本项目没有自行编写 M5 专用 Metal 内核，也没有启用 Apple Neural Engine（ANE）。因此，“支持 MPS”“在某台 M5 Max 上测得加速”和“已针对 M5 专门优化”是三个不同结论。

## 环境

使用原生 arm64 Python 和独立虚拟环境，避免在 Rosetta x86_64 解释器中安装另一套包。本项目 `pyproject.toml` 限定 Python 3.11 或 3.12，即使 PyTorch 自身支持更新的 Python，也须满足项目约束。先安装与机器、Python 和操作系统兼容的 PyTorch，再按项目 README 安装本包。不要在这套环境里安装旧仓库的 TensorFlow 1.14 依赖。

截至 2026-11-12，PyTorch 2.15 官方发布说明要求 Python 3.11 及以上、MPS 使用 macOS 15 及以上。Apple 的 PyTorch 入门页仍标注较早的 2.11 版本及其要求，不能据此覆盖新版本要求。实际支持由安装的 PyTorch、macOS 和设备探针共同确定。[PyTorch 2.15 发布说明](https://pytorch.org/blog/pytorch-2-15-release-blog/)、[MPS 文档](https://docs.pytorch.org/docs/2.15/notes/mps.html)、[Apple 入门页](https://developer.apple.com/metal/pytorch/)

先运行：

```bash
python -c "import platform, torch; print(platform.machine()); print(torch.__version__); print(torch.backends.mps.is_built()); print(torch.backends.mps.is_available())"
```

`is_built=True` 只表示安装包包含 MPS，`is_available=True` 才表示该会话当前可使用它。还必须通过项目的实际 3D 算子/网络探针；创建一个 MPS 标量成功，不代表整套三维模型都能运行。

`doctor` 使用小型网络做 FP32 前向与反向探针；通过它不能证明默认四层网络或大补丁的内存、速度已经合格。后续基准应使用与计划训练相同的网络配置。

## 本项目采取的设计

- 默认 FP32。使用批量较小的三维滑窗，并把整幅结果的融合累积留在 CPU，减少 GPU 工作集；统一内存依然会被这些对象共同占用。
- 滑窗尺寸、重叠和批量是显式参数。调大前先测内存和耗时，不能仅因为总内存大就把整幅 CT 送入 GPU。
- 新网络使用插值与卷积的上采样路径，避免将旧转置卷积及其低精度问题直接带入。PyTorch 的历史问题记录表明 MPS 的部分三维低精度算子曾有限制，具体支持需以当前探针为准。[官方问题记录 #160739](https://github.com/pytorch/pytorch/issues/160739)
- 设备选择和实际设备写入运行报告。明确请求 `mps` 却不可用时应报错；若选择 `auto` 而最终使用 CPU，报告中必须看得出来。
- 测速包含预热与设备同步。异步提交耗时不能当成网络实际执行时间；模型推理耗时与读取/预处理/保存耗时分开理解。

MPS 的加速范围是神经网络计算。影像读写、部分几何/数组处理、CPU 滑窗融合以及可选的 PyRadiomics 提取仍在 CPU 执行；更换 GPU 不等于这些步骤都会同比提速。

PyTorch 2.14 官方说明包含 `conv3d` 等算子向原生 Metal 内核迁移，这支持采用现代 MPS 执行层的方向，但没有给出本项目或 M5 Max 的速度保证。该版本涉及 M5/MPP 的注意力加速也不能自动推导为三维 U-Net 的收益。[PyTorch 2.14 发布说明](https://pytorch.org/blog/pytorch-2-14-release-blog/)

## 可选放射组学依赖

先验收基础包或 Apple 额外依赖对应的分割路径，再处理可选 PyRadiomics。项目固定的 PyRadiomics 3.0.1 在 PyPI 上仅提供旧 Python/x86_64 预编译包，没有 Apple arm64 或本项目 Python 3.11/3.12 的 wheel；这些环境需要解决源码构建及 C 扩展兼容性。官方安装页也明确把缺少对应 wheel 的情况转为源码编译，不能把 `pip install` 指令本身当成安装成功的证明。[3.0.1 发行文件](https://pypi.org/project/pyradiomics/3.0.1/#files)、[官方安装说明](https://pyradiomics.readthedocs.io/en/latest/installation.html)

若这一依赖无法构建，可使用经过验证的独立 CPU 提取环境，通过原 CT、整数标签、审核 JSON 和参数 YAML 交接；不要因此降级或混装主 MPS 环境。必须先确认该提取环境能运行本项目及真实 PyRadiomics 合成集成测试，才称这一步已支持。模拟提取器的门禁测试、单独导入包或 MPS 网络成功都不能替代它。

## 本机验收顺序

1. 按项目 README 安装后运行设备诊断，保存报告；显式选择 `mps`，不要用静默 CPU 回退来取得“成功”。
2. 在 CPU 和 MPS 上运行同一检查点、同一合成体积和同一精度/滑窗设置；核对形状、数值有限性及输出差异。合成数据只能验证代码与设备路径。
3. 运行基准命令，保存设备、版本、网络配置、补丁尺寸、批量、精度、预热、重复次数和同步后的耗时。滑窗重叠另从推理运行记录核对；基准只测补丁前向，不执行整幅滑窗。
4. 用有使用权限的真实 CT 叠加检查概率图、掩膜和原图的物理对齐；再用独立标注评价分割表现。性能测试不能替代准确性测试。

安装完成后先在项目目录运行：

```bash
ctln doctor --device mps --output reports/doctor-mps.json
ctln validate-device --device mps --output reports/parity-mps.json
ctln benchmark --device mps --roi-size 64 64 64 --batch-size 1 \
  --base-channels 16 --levels 4 --warmup 3 --iterations 10 \
  --output reports/benchmark-mps.json
ctln benchmark --device cpu --roi-size 64 64 64 --batch-size 1 \
  --base-channels 16 --levels 4 --warmup 3 --iterations 10 \
  --output reports/benchmark-cpu.json
```

benchmark 中的随机网络只用于速度和算子检查，不能用于病例推理或临床表现评价。示例显式使用训练示例相同的 16 基础通道、4 层和注意力配置；其他模型应相应调整，关闭注意力用 `--no-attention`。CPU/MPS 的补丁、批量、网络宽度、精度、预热及重复设置要一致；比较报告所记录的实际设置。MPS 报告的 `current_tensor_bytes` 是当前张量分配量，不是整机统一内存占用或峰值内存；不得把它改称峰值。

具有本项目格式的已训练权重及可信 CT 时，再运行：

```bash
ctln infer --image ct.nii.gz --checkpoint model.pt --output results \
  --device mps --roi-size 64 64 64 --batch-size 1 --overlap 0.5 \
  --assume-hu --assume-mm
```

`--assume-hu` 需要原始 DICOM/转换记录支持。`--assume-mm` 仅在 NIfTI 头空间单位未知、但来源已证明为 mm 时使用；头中已有可信单位时可省略，不能用它覆盖已知单位。它们不会把截图或归一化数组转换成可信 CT。

参数默认值和可用选项以[项目 README](../README.md)及 `--help` 为准；本文件不把尚未运行的机器报告填成结果。

## 出错与优化

- 设备不可用：核对 arm64 解释器、macOS 与对应 PyTorch 轮子；保留诊断原文。
- 内存不足：减小滑窗或批量，关闭其他占用大的工作；记录原设置。不要通过关闭 MPS 内存水位保护来掩盖问题。
- 算子不支持/数值异常：先保留 FP32 和明确设备失败，记录报错；选择 CPU 是一个新的、可见的运行模式，不等价于 GPU 验收通过。
- 数据类型错误：MPS 不支持双精度，先核对是否意外构造了 `float64` 张量；本项目网络路径使用 FP32。不要以 CPU 回退代替修正输入类型。[MPS 数据类型说明](https://docs.pytorch.org/docs/2.15/notes/mps.html)
- 默认不启用 fast-math、半精度、`torch.compile` 或未验证的内存布局。只有当同一机器的数值与速度对照通过，才把这些实验设置纳入项目配置。

`PYTORCH_ENABLE_MPS_FALLBACK` 等变量会影响实际执行路径，必须记录在报告中。本项目在选择 MPS 时拒绝 `PYTORCH_ENABLE_MPS_FALLBACK=1`，需显式取消该设置后再验收；失败时也不会自动重跑 CPU。[MPS 环境变量文档](https://docs.pytorch.org/docs/2.15/mps_environment_variables.html)

## 数值对照探针

`validate-device` 默认用固定种子的 2 基础通道、2 层小型网络，逐项比较 FP32 输出、输入梯度及参数梯度并保存误差。它验证本架构所用算子的基本数值表现，不等于完整训练模型已在 M5 上验收。默认 `rtol=0.001, atol=0.0001` 为工程探针容差，可显式修改并记录，不能解释为临床标准。CPU 自检的报告会明确标记 `accelerator_validated=false`。

## 当前证据边界

开发环境是 Windows，不能实际运行 Apple MPS。本次交付会报告真实执行的 CPU/本机测试；M5 Max 的可用性、输出一致性、峰值工作集和速度需在该机器上运行诊断及基准后确认。没有给出 M5 Max 的实测加速倍数，也没有作 ANE 加速声明。
