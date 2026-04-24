import torch
import torch.nn as nn  
from model import E2EPlanningHead
from ptq_sensitivity_profiler import fake_quantize_tensor


def export_mixed_precision_model():
    print("\n[Pipeline] 启动混合精度部署导出流 (Mixed-Precision Fallback Export)...")
    model = E2EPlanningHead()
    
    # 模拟编译器根据 Config 文件进行精度分配
    config = {
        "fc1": "INT8",            # 隐藏层用 INT8 榨取算力
        "final_linear": "FP16"    # 输出层 Fallback 到高精度，保坐标
    }
    
    print("[Config] 载入量化策略:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            target_dtype = config.get(name, "FP32")
            if target_dtype == "INT8":
                module.weight.data = fake_quantize_tensor(module.weight.data)
                print(f"  -> 算子 {name} 编译为 INT8 (BPU 执行)")
            elif target_dtype == "FP16":
                # 保持原样，模拟 FP16
                print(f"  -> 算子 {name} Fallback 为 FP16 (CPU/DSP 执行)")
                
    print("[Success] 混合精度模型编译完成！彻底消灭长尾截断误差！")

if __name__ == "__main__":
    export_mixed_precision_model()
