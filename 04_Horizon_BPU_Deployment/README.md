# Project 04: 面向地平线 BPU 的端到端模型算子适配与编译优化

## 📑 业务背景
将训练好的 ONNX 模型导入地平线工具链编译时，常因算子不支持导致 CPU Fallback，引发数据在 BPU 与 CPU 间频繁拷贝，导致推理 FPS 暴跌。

## 🔍 架构优化分析 (Deployment-Oriented Design)
针对地平线征程系列（J5/J6）硬件特性，实施了以下重构：
1. **彻底静态化 (Static Shape Tracing)：** 固定 Sequence Length，通过 Padding 消除动态维度导致的编译器回退。
2. **算子替换与查表加速 (LUT)：** 将 BPU 不友好的 HardSwish 和 Exp 操作重构为 INT8 Lookup Table (LUT) 实现。
3. **节点融合 (Graph Fusion)：** 强制融合量化节点与 Conv 算子，减少显存读写开销。

## 💡 核心产出
* **BPU 节点占比提升：** 从 68.5% 提升至 98.2%。
* **推理延迟下降：** CPU 同步开销减少 85%，整体 Latency 下降 60%。

## 📊 BPU Compiler 配置示例
本项目包含一份脱敏的 `deploy_config.yaml` 配置文件，展示了如何精细化控制量化策略与输入排布 (NHWC)。