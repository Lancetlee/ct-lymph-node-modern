# 胸部淋巴结 CT 现代化研究工具

将旧 `ct_mediastinal_structures_segmentation` 的研究用途改造成 Python 3.11/3.12、PyTorch 的训练与推理工程，并接入通用的 **thoracic-lymph-node-ct Skill**。支持 CPU、NVIDIA CUDA 和 Apple Silicon MPS 的显式设备选择。

**这是新的三维 Attention U-Net 架构，旧 TensorFlow/Keras `.h5` 权重不能直接使用。** 本项目不附带上游模型权重。本包也没有训练好的临床模型；需要用合规的 CT 与人工分割标签重新训练，或另行完成旧架构的忠实转换与数值验证。不要将安装成功、随机网络基准或合成测试当成淋巴结识别准确率。

## 本次改造

| 环节 | 实现 |
|---|---|
| 计算引擎 | PyTorch；`auto/cpu/cuda[:index]/mps`；明确请求的设备不可用时失败 |
| Mac GPU | FP32 三维卷积网络、插值上采样、滑窗、可调补丁/批量，整幅融合在 CPU；拒绝隐藏的 MPS→CPU 算子回退 |
| 训练 | 二值分割、CE + Dice 损失、患者隔离 train/val、按开发验证集选择检查点；不读取 test 图像 |
| 影像 | NIfTI / 可选经典 DICOM CT，保留原始 HU；模型网格预测还原到原 CT 物理坐标 |
| 复核 | 连通域只是候选，生成待审核 JSON；站别、真实结节 ID 和病理匹配需人工确认 |
| 放射组学 | 可选 PyRadiomics CPU 适配器；原 HU + 整数 ROI + 审核门禁 + 冻结参数/哈希 |
| 可追溯性 | 检查点架构/训练来源、原图与模型几何、实际设备、阈值、参数和输出记录 |
| Skill | IASLC 解剖、标注/证据要求与可执行流程整合，见 [SKILL.md](skill/thoracic-lymph-node-ct/SKILL.md) |

分割概率表示体素属于分割前景的得分，**不是淋巴结转移概率**。当前训练命令不训练良恶性/转移分类器，也不自动分配 IASLC 站别。连通域体积不能替代人工短轴测量。

## 安装

Mac 使用原生 arm64 Python 3.11 或 3.12。PyTorch 2.15 的 MPS 要求 macOS 15+；较早版本的要求可能不同，以安装版本的[官方说明](https://pytorch.org/blog/pytorch-2-15-release-blog/)和实际设备探针为准。

在本目录执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[apple,dicom,dev]'
ctln doctor --device mps --output reports/doctor-mps.json
ctln validate-device --device mps --output reports/parity-mps.json
ctln benchmark --device mps --roi-size 64 64 64 --output reports/benchmark-mps.json
```

Windows/Linux：先按 [PyTorch 安装选择器](https://pytorch.org/get-started/locally/)安装匹配本机的 CPU 或 CUDA 版本，再执行 `python -m pip install -e ".[dicom,dev]"`。PowerShell 虚拟环境激活为 `.\.venv\Scripts\Activate.ps1`。`ctln doctor --device cuda` 会实际执行小型三维网络的前向与反向。

`validate-device` 用相同权重和合成输入，对照 CPU 与目标设备的小型网络输出、输入梯度及各参数梯度；保存误差、容差、模型/软件配置与通过状态。失败返回非零退出码，不重跑 CPU。它是算子与数值验收，不是整模型或临床验证；默认容差也不是临床合格标准。

M5 Max 使用 MPS 调用 Mac GPU；本项目没有 M5 专用内核或 ANE 实现，也没有在 M5 Max 上测得加速倍数。具体安装、内存与测速说明见 [Apple Silicon 验收](docs/APPLE_SILICON.md)。本次真实开发验证环境为 Windows CPU。

## 训练新分割模型

1. 复制 [配置](configs/train.example.yaml) 与 [病例清单](configs/cases.example.csv)。示例病例路径是占位符，`dataset_description: null` 必须填写后才能训练。
2. 每行包含 `patient_id,scan_id,image,mask,split`；路径相对 CSV 所在目录解析。一个患者的所有检查放在同一 split；必须有 train 与 val，test 只登记、训练时不打开。不同伪 ID 仍可能属于同人，需在去标识化和划分前统一身份。
3. CT 是可确认 HU 的三维 NIfTI，掩膜必须与其物理网格一致、为整数 0/1。在配置 `training.assume_nifti_hu` 中作来源声明；仅在已确认且 NIfTI 未写单位时设置 `assume_nifti_mm`。声明不会修复错误数据。训练当前只接收 NIfTI；DICOM 可用于 infer/extract。
4. 根据训练队列制定空间采样、HU 窗和补丁配置；示例值只是工程起点。使用完整的人工标签，记录漏标、融合结节和小结节处理规则。二值标签表示分割前景，不表示病理阳性。

```bash
ctln train --manifest cases.csv --config train.yaml --output models/ln.pt --device mps
```

输出检查点及 `models/ln.training.json`；记录真实优化步数、选择的 epoch 和开发验证 Dice。该 Dice 不是独立测试集结果。代码检查规范化患者 ID、重复扫描 ID、跨集合相同 CT 路径，以及 train/val 的相同文件字节哈希；不能识别重新压缩、换格式、改伪 ID 后的全部重复病例。独立测试、跨中心验证、重复标注一致性和临床评价仍须另行完成。

## 推理与人工复核

```bash
ctln infer --image ct.nii.gz --checkpoint models/ln.pt --output results/case001 \
  --device mps --assume-hu --roi-size 64 64 64 --batch-size 1 --overlap 0.5 --scan-id case001
```

目录 DICOM 可用 `--image /path/to/series`；多序列目录必须指定 `--series-uid`。仅支持能核验规则几何的经典单帧 CT；不猜测缺失 HU/几何元数据或把非规则层间距当作规则体积。

输出 `segmentation_probability.nii.gz`、`candidate_mask.nii.gz`、`candidate_instances.nii.gz`、`candidate_review.json` 和 `run_manifest.json`。三幅影像回到原 CT 网格，原 CT 不会被归一化后覆盖。输出已存在时默认拒绝，确需覆盖须显式 `--overwrite`。

在影像软件中叠加复核、修订掩膜；拆分/合并候选后同步更新整数 label 与 review JSON。审核数组必须覆盖掩膜的每一个非零 label，审核通过至少填写 `node_id`、`review.status=approved`、`review.mask_approved=true`、审核者与时间。不通过的候选显式标记 `rejected`。站别/病理证据仍可未知；影像审核通过不自动产生病理真值。字段说明与完整记录示例见 [Skill 集成说明](skill/thoracic-lymph-node-ct/references/integration.md)。

## 可选：放射组学提取

本包将 CPU 提取与神经网络的 GPU 环境分开。**PyRadiomics 3.0.1 缺少本项目 Python 版本与 Apple arm64 的预编译 wheel**，其旧构建系统在本次 Windows/Python 3.12 实测安装失败。因此核心安装不包含它；真实 PyRadiomics 端到端提取尚未验收，不能宣称 Mac 放射组学安装已经解决。详见 [依赖限制与验收](docs/APPLE_SILICON.md#可选放射组学依赖)。

在已验证可构建 PyRadiomics 的独立 CPU 环境中安装本包及该依赖，先运行 `python -m pytest tests/test_radiomics_integration.py -q`，确认没有跳过并通过后再提取。`.[radiomics]` 声明所需版本，不保证旧包能在当前平台编译成功。

```bash
ctln extract --image ct.nii.gz --labels results/case001/candidate_instances.nii.gz \
  --review results/case001/candidate_review.json --params configs/radiomics.example.yaml \
  --output features/case001 --assume-hu
```

只能使用原始 HU CT，不能使用网络的归一化输入；不得以概率图充当整数 ROI。保存 CSV、参数与版本、图像/配置哈希、审核记录和病理是否可用于逐结训练的标志。特征集合和离散化参数均须在训练前固定；后续分类、特征选择、归一化和调参须在患者分组的训练折内拟合，当前代码没有自动实现转移分类模型。

## 验证与迁移

```bash
python -m pytest tests -q
ctln --help
```

本包的合成测试覆盖参数更新到原网格推理、物理坐标、HU 保留、分割标签、审核门禁、数据分组、内存预算与设备失败行为。真实执行记录放在 [validation](validation/)；GPU、真实 PyRadiomics 与临床验证的缺口均显式记录。见 [旧项目迁移](docs/MIGRATION.md)、[验收要求](docs/VALIDATION.md)。

原仓库见 [dbouget/ct_mediastinal_structures_segmentation](https://github.com/dbouget/ct_mediastinal_structures_segmentation)。保留 BSD 2-Clause 许可及归属；新架构没有宣称复现原论文精度。公共版不包含私人教学视频、关键帧或其逐时段笔记；材料与模型许可范围见 [NOTICE.md](NOTICE.md)。
