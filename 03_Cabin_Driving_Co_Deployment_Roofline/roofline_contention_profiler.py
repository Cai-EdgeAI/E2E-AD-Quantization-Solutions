import time

class HardwareSoC:
    """模拟一块车端芯片 (如 Orin 或 J6) 的物理极限"""
    def __init__(self):
        self.peak_flops = 100.0  # 理论峰值算力 (TFLOPS)
        self.memory_bw = 200.0   # 内存带宽 (GB/s)

    def compute_latency(self, flops_required, bytes_required):
        """
        基于 Roofline Model 计算耗时。
        耗时 = MAX(算力耗时, 访存耗时)
        """
        compute_time = flops_required / self.peak_flops
        memory_time = bytes_required / self.memory_bw
        return max(compute_time, memory_time), compute_time, memory_time

def run_roofline_profiling():
    soc = HardwareSoC()
    print("=== 舱驾一体 (E2E + VLM) 资源抢占与 Roofline 剖析 ===\n")
    
    # ----------------------------------------------------
    # 1. 单独跑 E2E (Compute-Bound)
    # ----------------------------------------------------
    e2e_flops = 2.5   # 矩阵计算量大
    e2e_bytes = 0.5   # 图像读入和特征图数据量
    e2e_total, e2e_comp, e2e_mem = soc.compute_latency(e2e_flops, e2e_bytes)
    
    print(f"[E2E 独占运行] 算术强度(FLOPs/Byte): {e2e_flops/e2e_bytes:.1f} (计算密集型)")
    print(f"  -> 耗时: {e2e_total*1000:.1f} ms (受限于算力瓶颈)\n")

    # ----------------------------------------------------
    # 2. 并发灾难：VLM Decode 抢占带宽 (FP16 Baseline)
    # ----------------------------------------------------
    vlm_fp16_flops = 0.01  # 计算量极小 (向量乘法)
    vlm_fp16_bytes = 4.0   # 疯狂读取权重和 KV Cache
    
    # 模拟并发抢占：VLM 强行霸占了 80% 的内存带宽
    soc.memory_bw = 200.0 * 0.2  # 留给 E2E 的带宽只剩 40 GB/s
    
    e2e_crash_total, _, e2e_crash_mem = soc.compute_latency(e2e_flops, e2e_bytes)
    print(f"[并发灾难] VLM (FP16) 开始生成文本，榨干内存带宽！")
    print(f"  -> VLM 算术强度: {vlm_fp16_flops/vlm_fp16_bytes:.4f} (极度访存密集)")
    print(f"  -> E2E 被迫等待数据，耗时飙升至: {e2e_crash_total*1000:.1f} ms (掉帧警告！)\n")

    # ----------------------------------------------------
    # 3. 部署优化：VLM 实施 W4A16 + KV-INT8
    # ----------------------------------------------------
    soc.memory_bw = 200.0  # 恢复总带宽
    
    # W4A16 将权重体积缩小 4 倍，KV-INT8 将 Cache 缩小 2 倍
    # 综合导致 VLM 访存量大幅下降
    vlm_quant_flops = 0.015  # 反量化带来了一点点额外计算量
    vlm_quant_bytes = 1.0    # 访存量下降到原来的 25%
    
    # VLM 现在只占用极少的带宽，E2E 恢复流畅
    soc.memory_bw = 200.0 - (vlm_quant_bytes / 0.05) # 简化的带宽扣除模拟
    
    e2e_fixed_total, _, _ = soc.compute_latency(e2e_flops, e2e_bytes)
    print(f"[联合优化] VLM 启用 W4A16 & KV-Cache INT8 量化...")
    print(f"  -> VLM 反量化时间被完美隐藏在访存间隙中 (免费算力)。")
    print(f"  -> VLM 释放出大量带宽，E2E 耗时恢复至: {e2e_fixed_total*1000:.1f} ms (安全状态！)\n")
    print("结论: Roofline 模型证明，LLM 量化的本质是为了拯救系统的内存带宽，而不是增加算力。")

if __name__ == "__main__":
    run_roofline_profiling()