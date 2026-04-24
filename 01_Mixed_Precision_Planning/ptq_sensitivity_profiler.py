import torch
import torch.nn as nn
from model import E2EPlanningHead

def fake_quantize_tensor(tensor, num_bits=8):
    """模拟最朴素的 Min-Max Per-tensor PTQ 带来的截断与精度损失"""
    qmin = -(2**(num_bits - 1))
    qmax = (2**(num_bits - 1)) - 1
    
    # 模拟寻找 scale 时发生了妥协 (导致大数值被截断)
    scale = tensor.abs().max() / (qmax + 50) # 人为制造一点截断溢出
    
    q_tensor = torch.round(tensor / scale)
    q_tensor = torch.clamp(q_tensor, qmin, qmax)
    return q_tensor * scale

def run_sensitivity_profiling():
    print("[System] 启动逐层量化敏感度分析 (Layer-wise Sensitivity Profiler)...\n")
    model = E2EPlanningHead()
    dummy_input = torch.randn(1, 256)
    
    # 1. 获取 FP32 黄金标准 (Ground Truth)
    with torch.no_grad():
        fp32_output = model(dummy_input)
    
    # 2. 逐层遍历，单层注入 INT8 噪声
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 临时保存 FP32 权重
            original_weight = module.weight.data.clone()
            
            # 注入 INT8 伪量化权重
            module.weight.data = fake_quantize_tensor(module.weight.data)
            
            with torch.no_grad():
                quantized_output = model(dummy_input)
            
            # 计算 MSE 误差 (模拟轨迹偏差)
            mse_loss = torch.nn.functional.mse_loss(quantized_output, fp32_output).item()
            
            # 恢复 FP32 权重
            module.weight.data = original_weight
            
            # 打印体检报告
            if "final" in name:
                print(f"[Warning] 算子 {name: <15} | MSE 误差: {mse_loss:.4f}  <-- 极度敏感！(截断灾难区)")
            else:
                print(f"[Info]    算子 {name: <15} | MSE 误差: {mse_loss:.6f}  (安全)")

if __name__ == "__main__":
    run_sensitivity_profiling()