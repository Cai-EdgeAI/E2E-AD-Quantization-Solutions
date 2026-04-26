# E2E-AD-Quantization-Solutions: 端到端自动驾驶模型量化与部署调优实践

本项目记录了在端到端自动驾驶大模型（如基于 Transformer 的 BEV 感知与规控架构）部署过程中，针对模型量化（INT8 / 混合精度）与底层硬件适配的工程实践。

项目主要探讨了在纯训练后量化（PTQ）路线下，如何排查和解决长尾数据截断、异常值扰动、异构访存瓶颈以及芯片工具链算子回退等实际部署问题。

## 📂 项目结构与核心内容

本项目分为 4 个子模块，分别针对量化部署链条中的不同工程挑战：

### [01. Mixed Precision Planning (规划头长尾截断与混合精度部署)](./01_Mixed_Precision_Planning)
* **Situation:** 在将 E2E 模型转换为 INT8 时，横向规划精度正常，但纵向长距离规划（如 10m 目标）被截断至 7m，导致车辆无法正常加速。
* **Action:** 实施逐层量化敏感度分析（Layer-wise Sensitivity Profiling）。定位到末端 Planning MLP 层因物理坐标的长尾分布导致 Scale 计算妥协。在编译配置中将该最终输出层 Fallback 至 FP16，执行混合精度编译。
* **Result:** 在整体推理耗时增加极小（<1%）的前提下，消除了量化溢出截断，轨迹规划精度对齐 FP32 原模型，替代了高成本的 QAT 方案。

### [02. BEV Outlier Ghost Braking (BEV 异常值截断与校准器优化)](./02_BEV_Outlier_Ghost_Braking)
* **Situation:** INT8 量化模型在强光或积水等特定场景下，View Transformer 产生激活异常值（Outliers），引发高置信度的假阳性检测（幽灵刹车）。
* **Action:** 弃用默认的 Min-Max 校准策略以避免 Scale 被异常值拉偏。引入基于数据统计分布的校准方法（如 Percentile 99.9% 截断），主动过滤极少数极端噪声。
* **Result:** 恢复了 BEV 空间的 INT8 正常特征分辨率，有效消除了由量化误差导致的异常检测框。

### [03. Cabin-Driving Co-Deployment (舱驾并发与 Roofline 访存分析)](./03_Cabin_Driving_Co_Deployment_Roofline)
* **Situation:** 模拟舱驾一体同 SoC 部署压测时，座舱 VLM 模型在文本生成（Decode 阶段）大量占用系统内存带宽，导致 E2E 视觉主干网络访存阻塞，帧率波动。
* **Action:** 结合 Roofline 模型对两者进行定性分析。对访存受限（Memory-Bound）的 VLM 实施 W4A16 权重压缩与 KV-Cache INT8 量化，利用计算单元的空闲周期处理反量化。
* **Result:** 显著降低了 VLM 的访存带宽需求，缓解了系统总线压力，保障了计算受限（Compute-Bound）的 E2E 模型在并发场景下的运行稳定性。

### [04. Horizon BPU Deployment (地平线 BPU 算子适配与编译)](./04_Horizon_BPU_Deployment)
* **Situation:** 初始 ONNX 模型导入地平线 OpenExplorer 工具链时，存在动态维度及不兼容算子，导致大量节点 CPU Fallback，异构调度开销大。
* **Action:** 执行部署导向设计（DOD）。固定动态 Shape，将复杂的激活函数及后处理操作替换为 INT8 LUT 查表或硬件原生支持的算子，优化 `deploy_config.yaml` 编译配置。
* **Result:** 提升了 BPU 算子运行占比，大幅减少了 CPU 与 BPU 之间的内存拷贝（D2H/H2D），优化了端到端推理的实际 Latency。

## 👨‍💻 专注方向
* 模型量化调优 (PTQ Calibration, 混合精度策略)
* 大模型端侧部署与性能 Profiling
* 自动驾驶端到端架构解析 (Transformer, BEV)
