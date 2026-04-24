import torch
import torch.nn as nn

class E2EPlanningHead(nn.Module):
    """
    模拟 UniAD / VAD 等端到端模型最后的 MLP 预测头。
    输入：Transformer Decoder 输出的 256 维隐向量 (Ego-Query)
    输出：未来 6 个时间步的 (X, Y) 坐标，共 12 维。
    """
    def __init__(self, in_dim=256, hidden_dim=64, out_dim=12):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        # 最后一层：物理降维，极容易发生量化截断！
        self.final_linear = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.final_linear(x)