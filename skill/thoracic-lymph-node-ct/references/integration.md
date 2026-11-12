# 与现代 CTLN 执行层结合

本技能负责输入资格、任务定义、解剖/标签约束和结果审查；项目 `ctln` 命令执行设备检查、分割、候选实例生成、特征提取及研究训练。二者通过影像文件、审核 JSON、配置 YAML 和运行报告对接。

这是研究用流程。新 PyTorch 架构需要匹配的训练检查点；旧 TensorFlow H5 不能直接复用。随机演示检查点只验证程序，不能产生真实病例结论。MPS 代表 Mac GPU 执行路径，不表示自动使用 ANE，也不表示已经在 M5 Max 上测得加速。

## 1. 启动前确认

读取当前项目 README 和 `ctln --help`。使用独立环境，先明确 `cpu`、`cuda` 或 `mps`；保存设备诊断：

```bash
ctln doctor --device auto --output reports/doctor.json
ctln validate-device --device mps --output reports/parity-mps.json
```

Mac GPU 验收应显式要求 `mps`，保存原生 arm64 环境的诊断和经过预热、设备同步的测速 JSON；CPU/MPS 对照使用相同输入、网络和参数。完整项目另有 `docs/APPLE_SILICON.md`。硬件未实际运行的性能不能写成测得结果。MPS 加速范围是所实现的神经网络计算；影像读写、CPU 融合和 PyRadiomics 提取不会因此自动全部转到 GPU。

核对原始 CT 的 HU、毫米单位和几何来源；在同一患者多序列 DICOM 中先选定序列，不混合期相，目录含多个序列时传 `--series-uid`。`--assume-hu` 需要转换来源支持；`--assume-mm` 仅在 NIfTI 头单位未知、来源却已证明为 mm 时使用，已有可信单位时可省略。这两个声明不进行标定或单位修复。原影像与网络归一化输入分开保存，原数据不得改写。

当前 DICOM 读取器只接受至少两层、规则正交网格的 classic 单帧 CT。Enhanced/多帧、非均匀层距和机架倾斜等不属于已实现路径，需有验证记录的外部转换；不把拒绝读取改成静默猜测几何。

## 2. 分割生成候选

使用来源、训练范围与配置匹配的检查点运行：

```bash
ctln infer --image ct.nii.gz --checkpoint model.pt --output results \
  --device mps --roi-size 64 64 64 --batch-size 1 --overlap 0.5 \
  --scan-id EXAMPLE_SCAN001 --assume-hu --assume-mm
```

CPU 或 NVIDIA 环境用对应设备值，`--scan-id` 换成去标识且唯一的扫描编号。不要因 GPU 不可用而私自把输出报告仍标为 GPU。滑窗和重叠是计算参数，不是已验证的医学阈值；改变设置后保留配置并检查输出一致性。命令默认保护已有输出；重跑优先用新目录，确需替换派生结果时才使用 `--overwrite`，不会因此允许覆盖输入文件。

预期输出的职责分开：

| 文件 | 含义 | 必须执行的审查 |
|---|---|---|
| `segmentation_probability.nii.gz` | 逐体素淋巴结结构分割分数 | 不解释为转移概率；检查有限数值、范围和物理网格 |
| `candidate_mask.nii.gz` | 分割阈值后的候选掩膜 | 与原 CT 叠加检查，不能直接当专家真值 |
| `candidate_instances.nii.gz` | 连通成分编号的候选 | 连通成分可能是误检、融合团块或一枚结节的碎片，不保证逐枚对应 |
| `candidate_review.json` | 待人工审核的候选记录 | 根据实际模板保留 ID、审核人、结论和争议，不自动填阳性病理 |
| `run_manifest.json` | 实际设备、配置及文件来源记录 | 核对实际运行路径、版本和检查点来源；留存以复现 |

不要把掩膜中的“前景”翻译为恶性或转移。体积/质心可从可信几何计算，但程序的轴向包围盒或三维径不能冒充人工临床短径。

## 3. 人工分站和轮廓审核

按本技能的分站表，结合连续 CT 和 MPR 核对候选是否确为结节、是否包含血管、是否融合、边界和站位是否清楚。必要时编辑轮廓、拆分或合并候选，并保持审核记录与最终标签图一致。

人工修改 `candidate_review.json` 时遵循当前版本生成的 schema；只批准已完成审核的候选。当前实现要求每个非零 `component_label` 恰有一条记录；获准提取的 ROI 需有唯一 `node_id`、`review.status: approved`、`review.mask_approved: true`、审核人和审核时间。确为误检者标为 `rejected`，保留记录；未完成的 `pending` 会阻止提取。拆分、合并或删改标签后同步更新记录，不能仅改审核状态来绕过核对。

技能中的 `assets/node-annotation-template.json` 是完整教学/研究标注模板，字段结构与 CLI 生成的记录数组不同，不能直接作为 `--review` 文件；需把确有来源的信息映射到当前 CLI schema。

`node_id` 在此也是经审核的 ROI 标识，单独填写它不证明这是可与病理一一配对的独立结节。确认是单枚结节时才把 `roi_unit` 从 `candidate` 改为 `node`。未知站位或病理保持未知，不为通过程序而填写假值。逐结监督标签还需 `reference_standard.label_unit: node`、阳性/阴性结果、`matching_evidence` 和受支持的组织/细胞学 `method`；阴性须 `sampling_adequacy: adequate`。当前受支持方法是 `pathology`、`cytology`、`surgical_pathology`、`biopsy_pathology`、`ebus_tbna`、`eus_fna`。仅有站级病理时保留站级标签，不能扩散到站内所有候选。

轮廓批准本身不验证病理。没有病理配对的已审核 ROI 可以提特征，但 `manifest.json` 中的 `training_label_verified` 应保持 `false`，不能伪造证据使它变为 `true`。

## 4. 从原 CT 提取放射组学

先确认当前环境实际装有可用 PyRadiomics。项目固定的 3.0.1 缺少现代 Python/Apple arm64 wheel，可能需要单独验证源码构建或使用兼容的独立 CPU 提取环境；不能因 MPS 分割成功就认定特征链也已验收。[官方发行文件](https://pypi.org/project/pyradiomics/3.0.1/#files)、[安装说明](https://pyradiomics.readthedocs.io/en/latest/installation.html)

先冻结 `configs/radiomics.example.yaml` 的具体研究参数，再对审核通过的标签提取：

```bash
ctln extract --image ct.nii.gz \
  --labels results/candidate_instances.nii.gz \
  --review results/candidate_review.json \
  --params configs/radiomics.example.yaml \
  --output features --assume-hu --assume-mm
```

提取必须使用保留 HU 的 CT，而非神经网络裁剪、缩放和归一化后的张量。概率图不是整数实例标签。参数文件应明确二维/三维、重采样、离散化、滤波及聚合设置；例子不是通用最优方案。完整成功时生成 `features.csv` 和 `manifest.json`，记录原图/标签指纹、参数哈希、提取器版本及审核来源。失败时保留错误原因并修正输入，不把失败项填 0；当前命令不会在某枚 ROI 失败后发布一份冒充完整成功的特征表。

## 5. 准备训练与评价

训练 CSV 按 `configs/cases.example.csv` 提供 `patient_id,scan_id,image,mask,split`，影像路径与二值结节掩膜应配对且物理几何一致。人工审核状态、结节 ID、癌种、期相和病理配对层级保存在通过这些 ID 关联的研究清单中；不能靠文件名猜病理。训练与验证按患者隔离，同一患者的结节、检查及重建序列不跨分区。`test` 留作独立评价，不参与本命令的训练或模型选择。

复制并完成训练 YAML，再运行下面示例。`training.dataset_description` 必须真实填写，示例中的 `null` 会被拒绝；这项来源声明本身不验证数据或医学效果。NIfTI 训练 CT 仍需确认 HU：有依据后设置 `training.assume_nifti_hu`；只有空间单位也需有据声明时才设置 `training.assume_nifti_mm`。确认网络/预处理、补丁尺寸、训练轮数及验证配置后冻结该文件。

```bash
ctln train --manifest cases.csv --config configs/train.example.yaml \
  --output model.pt --device mps
```

该命令训练的是项目所实现的分割网络，输出检查点及对应的 `.training.json`；按开发验证集 Dice 选择检查点不构成独立测试。训练分割模型使用结节轮廓；训练转移分类器还需要可靠的转移标签和额外建模流程，不能由分割损失或前景分数代替。保留未参与训练、特征选择和调参的测试数据；实际评价范围按项目实现与报告说明。

## 6. 对结果作出恰当表述

- 设备探针通过：仅说明指定软件/硬件下测试算子和模型路径可运行。
- `validate-device`：同一合成小网络与 CPU 比较 FP32 前向、输入梯度和参数梯度；报告容差、误差与实际设备。CPU 自检不算 GPU 验证，数值通过也不算临床验证。
- 合成演示成功：仅说明文件、几何及程序链路通过测试。
- 特征导出成功：说明指定 CT、掩膜和参数下产生了数值，不证明该特征能预测转移。
- 分割评价良好：说明在相应人群与标注下的分割表现；不等于病理转移判断准确。
- 只有经过独立数据验证和校准的具体转移模型，才可在其适用范围内报告转移风险；本项目的分割概率不得用于这项表述。

报告中分别保留影像资料、公开文献、人工判定和软件计算的来源。当前条件不足时，完成可用的定性教学或数据质控，明确尚未运行的环节，不补造结果。
