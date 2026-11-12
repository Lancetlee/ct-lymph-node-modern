# 胸部淋巴结 CT 放射组学：核对后的证据与训练约束

检索日期：2026-11-12。用途：供本技能进行特征选择、研究解释与训练设计时引用；这是精选证据，不是系统综述，也不是经过前瞻性验证的诊断规则。以下只把已查阅的原始论文/官方规范作为证据，不把作者的临床应用愿景当成已证实疗效。

## 一、直接研究淋巴结本身的 CT 证据（优先）

| 来源 | 人群、ROI、真值及切分 | 已核对结果 | 对技能的具体影响 |
| --- | --- | --- | --- |
| [R1 Sha 等，2020，BMC Medical Imaging](https://link.springer.com/article/10.1186/s12880-020-0416-3)，DOI 10.1186/s12880-020-0416-3；[全文 PMC7003415](https://pmc.ncbi.nlm.nih.gov/articles/PMC7003415/) | 86 名 NSCLC 患者、231 枚有手术病理信息的纵隔 LN。61/25 名患者按时间分训练/验证，对应 163/68 枚 LN。LN ROI 在动脉期勾画，并配准到平扫、静脉期；有单期和跨期模型。 | 平扫 LN 模型训练/验证 AUC 0.926/0.925；这是同中心时间验证，不是外部中心验证。 | 逐 LN 任务的可行性证据；相位、配准和勾画条件必须保留。不要据此宣称平扫在所有任务都优于增强。摘要写 846、正文写 841 项，复现应核查补充材料，而非静默选择一个数。 |
| [R2 Dong 等，2021，Frontiers in Oncology](https://www.frontiersin.org/journals/oncology/articles/10.3389/fonc.2020.558428/full)，DOI 10.3389/fonc.2020.558428 | 129 名患者、201 枚病理证实的纵隔 LN，训练/测试 140/61 枚；静脉期最大轴位 **2D** LN ROI，A.K. 软件。恶性组混合肺腺癌、肺鳞癌、SCLC、乳腺/肾癌；良性含结节病、结核、炎症。 | 内部测试 LR AUC 0.850（95% CI 0.747–0.953）。最终八项：Percentile20、SumAverage、GLCMEnergy、ClusterShade、Inertia、HighGrayLevelRunEmphasis、LongRunHighGrayLevelEmphasis、SurfaceArea。作者报告 Percentile20 与 Inertia 为负向，GLCMEnergy 与 SurfaceArea 为正向。 | 不能称为纯 NSCLC 转移研究；不能把 2D 模型当成 3D 模型。多变量模型的系数受相关变量与实现影响，不能简化成“越不均匀必越恶性”。研究内系数不能移植成临床阈值；患者内相关性/切分需复查。 |
| [R3 Kang 等，2025，BMC Medical Imaging](https://link.springer.com/article/10.1186/s12880-025-01686-1)，DOI 10.1186/s12880-025-01686-1 | 356 名 pN0–pN2 NSCLC 患者；207/90/59 名训练/内部测试/独立测试。第 4 站 LN 静脉期 **3D** ROI，病理对应，最多每人选三枚。来自同院两院区，独立组采用不同扫描设备。 | RF 融合模型内部测试 AUC 0.889；独立测试 0.883（95% CI 0.795–0.972）。入模八项见下节。独立测试不等于独立外院验证；论文在患者与多枚 LN 的聚合描述上需复现者进一步澄清。 | 接近第 4 站定位训练目标，但不能推广至所有站。作者使用 box mean、shot noise、speckle noise 等变换且部分参数保留默认值；复现前必须取得实现、参数、随机性控制，不能直接复制 Radscore 公式。 |

## 二、具体特征：保留完整图像变换/ROI 语境

R3 的八项按论文图 6/8 和 Radscore 描述核对：

1. wavelet-HHH / GLRLM RunVariance。
2. box mean / GLDM GrayLevelNonUniformity。
3. shot noise / GLRLM RunEntropy。
4. LoG sigma 1 mm / GLRLM RunEntropy。
5. shot noise / GLDM LargeDependenceEmphasis。
6. speckle noise / GLDM DependenceNonUniformity。
7. LoG sigma 4 mm / GLRLM RunEntropy。
8. original / shape SurfaceVolumeRatio。

论文中的前七项系数为正、SurfaceVolumeRatio 为负，只能解释为该训练集、预处理与多变量模型中的方向；不构成跨中心、跨癌种或跨期相的独立诊断规律。此处保留名称帮助查证，不能声称这些特征已在本地实际计算或验证。

计算含义采用 [R6 IBSI 官方标准入口](https://theibsi.github.io/)校准，以下为训练用简释：

| 类别/示例 | 简释 | 训练时避免的误解 |
| --- | --- | --- |
| 形状：Volume、SurfaceArea、SurfaceVolumeRatio、球形度、最大 3D 径 | ROI 的几何规模、表面积及紧致程度 | 最大 3D 径不等于轴位短径；SurfaceVolumeRatio 强烈受尺寸和分割影响。 |
| 一阶：HU 均值/中位数、分位数、偏度、峰度、熵、能量 | 单个体素强度的分布，不含空间邻接 | 一阶 energy 不等于 GLCM energy；不能从屏幕灰度反推 HU。 |
| GLCM：Energy、ClusterShade、Inertia/Contrast、IDN、MCC | 灰度对的联合分布；涉及均匀性、非对称性、差异或相关结构 | 名称相近也须核对公式、距离、方向与聚合；“homogeneity”不是良性标签。 |
| GLRLM：RunVariance、RunEntropy、HGLRE、LRHGLE | 同灰度连续游程的长度、分布及灰度权重 | 游程长度依赖体素间距、方向与离散化。 |
| GLSZM | 连通的同灰度区域的大小与分布 | zone 不是 lymph-node station，也不是肿瘤组织学分区。 |
| GLDM：GLNU、DNU、LDE | 邻域依赖的灰度分布、非均匀性与大依赖权重 | 依赖定义、阈值与邻域设置影响数值。 |
| NGTDM：Coarseness、Contrast 等 | 体素与邻域平均灰度差异 | 不能肉眼估值，亦无通用“转移 cut-off”。 |
| LoG、wavelet、box/noise 变换 | 先改变输入图像，再提取上述量 | 变换后的 feature 是不同变量；必须记录滤波尺度、边界处理、实现和种子。 |

实现时再对照[软件的特征定义](https://pyradiomics.readthedocs.io/en/latest/features.html)，不能直接按缩写字符串跨软件合并。一阶 Energy 与 GLCM JointEnergy 计算对象不同；不同软件的 kurtosis 是否减 3、负 HU 的处理也可能不同。网页 `latest` 在本次读取时标为开发版，不把其版本号当成本技能已安装的软件版本。

## 三、必须与逐 LN 诊断分开的患者级任务

| 来源 | 任务、ROI、标签及验证 | 数值与适用边界 |
| --- | --- | --- |
| [R4 Jiang 等，2024，Respiratory Research](https://pmc.ncbi.nlm.nih.gov/articles/PMC11138070/)，DOI 10.1186/s12931-024-02852-9 | 四中心 242 名 cT1–2N0M0 **SCLC** 手术患者；158 训练、84 外部验证；增强 CT 原发瘤 GTV 和约 15 mm 瘤周环；术后病理确定患者隐匿 LN 转移。 | 融合模型外部 AUC 0.772（95% CI 0.656–0.887），纯 GTV+PTV 模型 0.703。最终使用 GTV MCC、PTV median/IDN，并加吸烟与原发瘤形状。此处的 round/oval 是**原发瘤**形状，不是 LN 形状；不能用于判某一枚 LN 的真假转移。 |
| [R5 Yin 等，2025，Chinese Journal of Cancer Research](https://pmc.ncbi.nlm.nih.gov/articles/PMC11893343/)，DOI 10.21147/j.issn.1000-9604.2025.01.02 | 两中心 358 名早期、影像 cN0 肺腺癌患者；186/48/124 训练/内部验证/外部测试；平扫薄层 CT **原发瘤** ROI；手术及清扫病理为患者级标签。 | 深度学习+放射组学+临床融合外部 AUC 0.907；不能冒称为单纯手工 radiomics AUC。训练来源中心做了阳性全纳入/阴性抽样，患病率与概率校准需注意；不宜把其输出当成逐枚 LN 检出或 SCLC 风险。 |

多模态拓展，仅一条：Zhai 等 2025（[PMID39983386](https://pubmed.ncbi.nlm.nih.gov/39983386/)，DOI 10.1016/j.crad.2025.106832）对中心 1 的 88 人/559 枚 LN、中心 2 的 75 人/543 枚 LN 建模，PET/CT 融合模型中心 2 AUC 报告 0.923/0.927。含 PET 的证据不能用于声称 CT 单模态达到该性能；本文只核对到摘要，病理逐结对应与患者分组细节须读全文确认。

## 四、为后续 CT 训练落实的规则（须验证的工程建议）

1. 先声明任务：逐枚 LN 分类、逐站分类、患者是否存在转移、隐匿转移风险、治疗后残留病变必须分开。训练检测/分割与训练转移判断也应分开。
2. 一枚 LN 的标签只能来自可靠的同枚/同站匹配；只有“该站阳性”时不能把站内每枚全部标阳性。未采样不能标阴性；病理阴性须记录取样方式与充分性。弱标签单独存储并标注来源。
3. 区分原发灶、LN、LN 周围、原发瘤周围的 ROI；每个特征加 ROI 类型、期相、日期、站号、掩膜版本和实现版本。不要将 GTV/PTV 研究的特征名套到 LN 上。
4. 原始 DICOM HU 和几何信息可用时才做定量 radiomics。MP4、JPEG、窗宽窗位后的截图可做解剖/视觉标记学习，不能作为可复现 HU/纹理特征来源。
5. 所有节点、检查时点、重建序列、增广片段按患者隔离切分。先划分患者，再在训练折内做缺失处理、标准化、批次校正、特征筛选、重采样与调参；测试集不参与选择。模型指标以患者聚类处理相关性。
6. CT 必须保留平扫/增强及期相、扫描仪、重建核、层厚、层间距、像素间距、造影剂时序。预处理保存 YAML/JSON；不把某论文 1 mm 或 bin width=25 HU 当无条件通用设置。
7. 分割重复性/扰动稳健性与模型外部表现分别检验；二者不互相替代。小 LN 的部分容积、掩膜少体素、黏连/融合、血管边界尤其应人工复核。
8. 除 ROC-AUC，还报告灵敏度/特异度、PPV/NPV、PR-AUC、置信区间、校准及适用阈值，分癌种、站号、大小、期相、中心、治疗状态报告性能。阈值由训练/验证过程预定，不能在独立测试集择优。
9. 必备基线包括短径等临床/语义特征模型；检验 radiomics 的增量价值，不将一个更高 AUC 或不显著的 Hosmer–Lemeshow 结果等同于临床可用性。

上述研究报告规范可参照：

- [R7 CLEAR，2023，原始报告规范](https://pmc.ncbi.nlm.nih.gov/articles/PMC10160267/)，DOI 10.1186/s13244-023-01415-8：58 项用于让图像、分割、特征、模型、验证及开放资源可审查；它是报告标准，不是性能认证。
- [R8 TRIPOD+AI，2024，原始报告规范](https://www.bmj.com/content/385/bmj-2023-078378)，DOI 10.1136/bmj-2023-078378：临床预测模型的透明报告标准，涵盖回归与机器学习；不能当作具体研究设计的替代或外部有效性的保证。
- [R6 IBSI](https://theibsi.github.io/)：特征/图像处理的标准定义和参考基准。研究技能应要求对照基准校准实现；使用自称 IBSI-compliant 的软件并不自动证明模型经过临床验证。

## 五、未夸大的结论

可以写：LN 的几何、一阶强度、GLCM、GLRLM、GLSZM、GLDM、NGTDM 和多尺度变换特征，在特定回顾性胸部研究中有预测价值；前提是任务、真值、ROI、协议、预处理和验证范围一致。

不要写：entropy 越高必为转移；一个轮廓圆/饱满的 LN 就能定性；所有论文均已多中心外部验证；NSCLC 系数适用于 SCLC；原发瘤的患者级 OLNM 模型可以给每枚 LN 自动贴标签；仅凭截图或视频就能复现定量 CT 特征。

## 源读取与复现说明

R1、R2、R3、R4、R5 已查阅全文正文；R2 实际特征在正文 Discussion 中核对，R3 在正文 Radscore 及图注核对。R1 的具体特征权重位于补充 DOCX，本次未完成补充表解析，故不编造具体变量名/系数。PET/CT 拓展仅限摘要。文中 AUC 是作者原报告，未在本地重新训练或验证。
