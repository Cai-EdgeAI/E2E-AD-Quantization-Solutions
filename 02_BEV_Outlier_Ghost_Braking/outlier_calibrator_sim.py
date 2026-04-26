import torch
import numpy as np

def simulate_bev_features():
    """
    模拟 BEV 空间中的特征分布：
    99.9% 是正常路面/车辆特征 (分布在 0~5 之间)
    0.1%  是逆光造成的极端异常值 (数值高达 150)
    """
    # 模拟 10 万个正常特征
    normal_features = torch.rand(100000) * 5.0 
    # 模拟 100 个异常光斑特征
    outliers = torch.ones(100) * 150.0 
    
    # 组合成一个 BEV 特征图
    bev_feature_map = torch.cat([normal_features, outliers])
    # 打乱顺序
    bev_feature_map = bev_feature_map[torch.randperm(bev_feature_map.size(0))]
    return bev_feature_map

def min_max_quantize(tensor, num_bits=8):
    """【失败案例】朴素的 Min-Max 量化"""
    qmax = (2**(num_bits - 1)) - 1 # 127
    
    # 寻找绝对最大值（被 150 绑架了！）
    max_val = tensor.abs().max()
    scale = max_val / qmax
    print(f"  [Min-Max 策略] 选定的最大值: {max_val:.2f}, 计算得 Scale: {scale:.4f}")
    
    q_tensor = torch.round(tensor / scale)
    q_tensor = torch.clamp(q_tensor, -qmax, qmax)
    return q_tensor, scale

def percentile_quantize(tensor, percentile=99.9, num_bits=8):
    """【工业界标准】百分位截断量化 (丢卒保车)"""
    qmax = (2**(num_bits - 1)) - 1 # 127
    
    # 排序并寻找 99.9% 处的数值
    sorted_tensor, _ = torch.sort(tensor.abs())
    idx = int(len(sorted_tensor) * (percentile / 100.0))
    clip_val = sorted_tensor[idx]
    
    scale = clip_val / qmax
    print(f"  [百分位 策略] 截断 {100-percentile:.1f}% 异常值, 选定的最大值: {clip_val:.2f}, 计算得 Scale: {scale:.4f}")
    
    q_tensor = torch.round(tensor / scale)
    q_tensor = torch.clamp(q_tensor, -qmax, qmax)
    return q_tensor, scale

def run_experiment():
    print("=== BEV 异常值量化灾难与修复模拟 ===\n")
    
    # 1. 获取特征图并挑选一个特定的“正常车辆特征”来跟踪
    bev_features = simulate_bev_features()
    # 假设某辆车的特征原始值是 4.2
    target_car_feature = 4.2
    print(f"原始 FP32 状态下，某车辆的特征值为: {target_car_feature}\n")
    
    # 2. 发生灾难：Min-Max 量化
    print(">>> 启动 Min-Max PTQ 量化 (遇到高光异常值)...")
    _, scale_mm = min_max_quantize(bev_features)
    
    # 看看车辆特征变成了什么样子？
    q_car_mm = torch.round(torch.tensor(target_car_feature) / scale_mm)
    dq_car_mm = q_car_mm * scale_mm
    print(f"  --> 量化后，该车辆特征被反量化为了: {dq_car_mm:.2f} (原值 4.2)")
    print(f"  --> 诊断: 误差巨大！INT8 的刻度太粗，导致特征分辨率彻底丢失，极易诱发分类幻觉（幽灵障碍物）。\n")
    
    # 3. 拯救世界：百分位截断量化
    print(">>> 切换为工业级 Percentile(99.9%) 截断校准...")
    _, scale_pct = percentile_quantize(bev_features, percentile=99.9)
    
    q_car_pct = torch.round(torch.tensor(target_car_feature) / scale_pct)
    dq_car_pct = q_car_pct * scale_pct
    print(f"  --> 量化后，该车辆特征被反量化为了: {dq_car_pct:.2f} (原值 4.2)")
    print(f"  --> 诊断: 完美保留！通过牺牲 0.1% 的无用光斑，成功保护了 99.9% 正常特征的分辨率！")

if __name__ == "__main__":
    run_experiment()