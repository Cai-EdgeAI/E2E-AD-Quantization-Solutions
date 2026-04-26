# Project 02: 攻克 BEV 空间异常值引发的“幽灵障碍物”

## 📑 业务背景
在基于 BEVFormer/UniAD 架构的端到端大模型部署中，将网络转换为 INT8 后，空旷或高光场景常发生“幽灵刹车”。
**根源：** 图像高光噪音经过 Deformable Attention 放大，产生了极端的激活异常值 (Activation Outliers)。

## 🔍 病理分析 (Scale Hijacking)
1. **刻度绑架：** 默认 Min-Max 策略会被个别极大值（如光斑）绑架，导致缩放因子 (Scale) 极度拉大。
2. **特征湮灭：** 正常的车辆特征（数值小）除以巨大的 Scale 后，在 INT8 空间变成了 0，导致检测幻觉。

## 💡 解决方案：高级数据分布校准
本项目摒弃 Min-Max，采用**智能截断策略 (Smart Clipping)**：
* **Percentile Calibration (百分位截断)：** 强制截断 99.9% 之后的数据，保护 99.9% 的正常特征分辨率。
* **KL-Divergence / Entropy Calibration (熵校准)：** 通过寻找信息损失最小的截断点，平衡精度与截断误差。

## 📊 实验结论
通过将 View Transformer 算子的校准策略从 Min-Max 切换为 **Percentile(99.9%)**，成功消除了高光场景下的幽灵障碍物，且零推理开销。