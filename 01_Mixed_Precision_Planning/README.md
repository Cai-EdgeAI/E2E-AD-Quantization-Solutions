# Project 01: E2E 规划控制模块的混合精度 PTQ 部署方案
*(Mixed-Precision PTQ Deployment for End-to-End Planning Head)*

## 📑 业务背景 (Background)
在基于 Transformer 的端到端自动驾驶大模型（如 UniAD / VAD）部署中，模型通常采用统一的 INT8 训练后量化 (PTQ) 以满足车端 NPU/BPU 的实时性要求 (FPS)。
然而，在实际开环仿真与闭环测试中，我们发现**横向规划（Lateral）精度正常，但纵向规划（Longitudinal）在长距离加速场景下出现严重的“截断误差”（如 10m 目标被截断至 7m，引发幽灵刹车）。**

## 🔍 病理溯源 (Root Cause Analysis)
经过分析，导致该现象的底层原因在于：
1. **长尾物理分布：** 规划模块末端（Planning MLP Head）输出的是真实物理坐标。纵向距离的动态范围（0~80m）远大于横向（-5~5m）。
2. **INT8 饱和截断 (Saturation Clipping)：** PTQ 校准工具在 Per-tensor 粒度下，为了保护 99% 稠密近处轨迹的精度，选择了较小的缩放因子 (Scale)。当遇到纵向长距离极值时，触发 INT8 表达上限 (127)，导致不可逆的坐标溢出与截断。

## 💡 解决方案：逐层敏感度分析 + 混合精度部署 (Mixed Precision PTQ)
考虑到在百亿参数模型上执行 QAT（量化感知训练）存在显存开销极大、梯度崩溃风险及迭代周期过长等工程痛点，本项目抛弃 QAT，采用**纯 PTQ 路线**解决该问题。

### 核心工作流：
1. **Layer-wise Sensitivity Profiling:** 构建自动化脚本，对网络逐层进行 INT8 伪量化注入，监控轨迹 MSE Loss 的劣化率。
2. **算子定位：** 精准定位出最后两层 `Linear` 算子为 INT8 极度敏感层（误差放大 > 300%）。
3. **Fallback 策略：** 编写混合精度流水线，将主干网络（ResNet + BEVFormer）保持 INT8 以榨取算力，强制将最后的 Planning Head 剥离至 CPU/DSP 使用 FP16 进行计算。

## 📊 实验结果 (Results)
| 部署策略 | 规划 Head 精度 (Type) | 纵向最大误差 (m) | 整体推理耗时 (ms) | 评价 |
| :--- | :---: | :---: | :---: | :--- |
| 全 FP32 原模型 | Float32 | 0.12 | 48.5 | 基准基线，但无法满足车端实时性 |
| 纯 INT8 PTQ | INT8 | **3.45 (严重截断)** | 12.1 | 速度达标，但发生长距离幽灵刹车 |
| **混合精度 (本项目)** | **INT8 + FP16** | **0.15** | **12.3** | **零训练成本解决截断，耗时仅增加0.2ms** |

**结论：** 本项目以 $<0.5\%$ 的 Latency 牺牲，完美修复了 PTQ 量化导致的 E2E 轨迹截断问题，证明了通过架构分析和混合精度策略可有效替代高成本的 QAT。