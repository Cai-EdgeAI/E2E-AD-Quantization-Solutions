# Project 03: 舱驾一体架构下的 VLM 与 E2E AD 资源抢占剖析与 Roofline 优化
*(Resolving Resource Contention between VLM and E2E Models via Roofline Profiling & KV-Quantization)*

## 📑 业务背景 (Background)
在下一代“舱驾一体”架构中，车端 SoC (如 NVIDIA Orin / Horizon J6) 需要同时运行端到端自动驾驶模型 (E2E AD) 与座舱视觉语言大模型 (VLM)。
在并发压测中，我们发现：当座舱 VLM 执行长文本生成的 Decode 阶段时，会大量抢占 DRAM 内存带宽，导致 E2E 模型的图像主干网络发生严重的 Cache Miss 与数据饥饿，最终导致自动驾驶轨迹更新帧率 (FPS) 从 30 暴跌至 12。

## 🔍 架构级病理分析 (Roofline Model Analysis)
通过构建硬件的 Roofline Model（屋顶线模型），我们对两者的物理瓶颈进行了定性与定量分析：
1. **E2E 模型 (Compute-Bound)：** 具有极高的算术强度 (Arithmetic Intensity)。其瓶颈在于 BPU/GPU 的矩阵乘加算力池 (MACs)。
2. **VLM 模型 (Memory-Bound)：** 在 Decode 阶段，算术强度极低 (GEMV 运算)。其瓶颈在于内存带宽 (Memory Bandwidth)。此外，随着上下文长度增加，庞大的 KV Cache 会引发严重的 VRAM OOM 风险。

## 💡 终极部署战略 (Deployment Strategy)
为了在同一块 SoC 上实现硬件资源的时空隔离与极致榨取，本项目实施了以下联合部署策略：
1. **对于 E2E 模型：** 采用 INT8 / 混合精度 PTQ，极致压缩 Activation 维度，提升算术吞吐量。
2. **对于 VLM 模型：** - 引入 **W4A16 量化**：将 Weight 访存量降低 75%，利用 Decode 阶段的“免费空闲算力”隐藏反量化开销。
   - 引入 **KV Cache INT8 量化**：将长文本的显存占用减半，彻底解决车机显存墙导致的 OOM。

## 📊 实验与核心结论 (Results)
本项目实现了一个轻量级的并发 Profiler 模拟器。结果表明：在未优化状态下，VLM 的生成会导致 E2E 延迟恶化 150%。在施加 W4A16 + KV-INT8 联合优化后，系统内存带宽占用率下降 60%，自动驾驶 E2E 模型重新稳定在 30 FPS 的安全底线之上。证明了基于 Roofline 理论的异构量化策略在舱驾一体部署中的绝对必要性。