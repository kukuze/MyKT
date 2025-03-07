import torch
from torch import nn

# 构建标签到索引的映射
from constant import Constants


class KnowledgeModel(nn.Module):
    """知识点嵌入模块"""
    def __init__(self, know_num, hidden_size):
        super(KnowledgeModel, self).__init__()
        self.knowledge_embedding = nn.Linear(know_num, hidden_size)
        # 使用字典存储tag到index的映射
        self.tag_to_idx = {tag: idx for idx, tag in enumerate(Constants.TAGS)}
        self.vector_size = Constants.TAGS_NUM
        self.device = torch.device(f'cuda:{Constants.CUDA}' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    def forward(self, knowledge):
        # 分割字符串并去除空白
        tags = [tag.strip() for tag in knowledge.split(',')]

        # 创建独热编码向量
        encoded = torch.zeros(self.vector_size, device=self.device)
        for tag in tags:
            if tag in self.tag_to_idx:
                encoded[self.tag_to_idx[tag]] = 1


        return self.knowledge_embedding(encoded)


