# 🚀 E2E-AD-Quantization-Solutions: 端到端自动驾驶大模型量化与部署实战

**“不盲从 QAT 的高昂算力，专注纯 PTQ 的极致工程化落地。”**

本项目聚焦于解决端到端模型从算法原型向车端底层硬件（如 Horizon BPU、NVIDIA Orin）迁移过程中的精度崩塌与资源抢占问题。

## 🎯 仓库核心架构

### [📂 01. Mixed Precision Planning (混合精度方案)](./01_Mixed_Precision_Planning)
* **痛点：** 规划模块长尾分布导致 INT8 饱和截断。
* **方案：** 逐层敏感度分析，精准实施 FP16/INT8 混合精度 Fallback。以 0 成本找回 100% 轨迹精度。

### [📂 02. BEV Outlier Ghost Braking (解决幽灵刹车)](./02_BEV_Outlier_Ghost_Braking)
* **痛点：** 异常值绑架 Scale 导致特征湮灭。
* **方案：** 引入 Percentile 99.9% 截断与 KL 散度校准，重构 BEV 空间特征分辨率。

### [📂 03. Cabin-Driving Co-Deployment (舱驾一体 Roofline 优化)](./03_Cabin_Driving_Co_Deployment_Roofline)
* **痛点：** 座舱 VLM 抢占带宽导致智驾掉帧。
* **方案：** 利用 Roofline Model 部署 W4A16 与 KV-Cache 压缩，释放 60% 带宽。

### [📂 04. Horizon BPU Deployment (地平线工具链适配)](./04_Horizon_BPU_Deployment)
* **痛点：** 动态 Shape 导致 CPU Fallback 严重。
* **方案：** 计算图静态化与 LUT 查表替换，BPU 节点占比提升至 98.2%。

## 👨‍💻 关于作者
* 专注领域：模型量化 (PTQ/QAT)、硬件感知神经网络 (HW-aware NAS)、车端异构体系结构。
* 核心理念：追求在硅片上跑得又快又稳的系统工程。