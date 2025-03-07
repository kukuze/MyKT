import torch
from torch import nn

from constant import Constants
from constant.Constants import DIFFS_MAX, DIFFS_MIN


class DifficultyEncoder(nn.Module):
    def __init__(self, min_diff=DIFFS_MIN, max_diff=DIFFS_MAX, difficulty_dim=Constants.DIFFICULTY_EMB_DIM):
        super().__init__()
        self.min_diff = min_diff
        self.max_diff = max_diff

        # 可学习的线性变换
        self.difficulty_proj = nn.Linear(1, difficulty_dim)
        # 将模型移动到 GPU（如果可用）
        self.device = torch.device(f'cuda:{Constants.CUDA}' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, difficulty):
        # 确保输入是tensor
        if not isinstance(difficulty, torch.Tensor):
            difficulty = torch.tensor(difficulty, dtype=torch.float32)
        # 将输入移动到与模型相同的设备
        difficulty = difficulty.to(self.device)
        # 归一化
        normalized_diff = (difficulty - self.min_diff) / (self.max_diff - self.min_diff)

        # 转换为张量并展开
        normalized_diff = normalized_diff.unsqueeze(-1)

        # 线性变换
        return self.difficulty_proj(normalized_diff)

# # 使用示例
# difficulty_encoder = DifficultyEncoder()
# difficulty_features = difficulty_encoder(1600)
# print(difficulty_features)
