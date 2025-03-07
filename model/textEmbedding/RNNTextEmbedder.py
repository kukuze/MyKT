import torch
from torch import nn

from constant import Constants


class RNNTextEmbedder(nn.Module):
    """
    单层双向GRU建模题面
    参数:
        wcnt: 词表大小
        emb_size: 词嵌入维度
        topic_size: 最终输出维度
    """

    def __init__(self, wcnt, emb_size, topic_size):
        super(RNNTextEmbedder, self).__init__()

        # 词嵌入层
        self.embedding = nn.Embedding(wcnt, emb_size, padding_idx=0)

        # 单层双向GRU
        self.emb_size = topic_size // 2
        self.rnn = nn.GRU(emb_size, topic_size // 2, 1,
                          bidirectional=True)
        self.device = torch.device(f'cuda:{Constants.CUDA}' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    def forward(self, input, hidden):
        """
        前向传播
        参数:
            input: [seq_len] 的输入序列
            hidden: [2, 1, topic_size//2] 的初始隐藏状态
        返回:
            y: [topic_size] 经过max pooling的序列表示
            h1: [2, 1, topic_size//2] 更新后的隐藏状态
        """
        # 增加batch维度
        x = self.embedding(input.unsqueeze(1))  # [seq_len, 1, emb_size]
        y, h1 = self.rnn(x, hidden)  # y: [seq_len, 1, topic_size]
        y, _ = torch.max(y, 0)  # [1, topic_size]
        return y.squeeze(0), h1  # [topic_size], [2, 1, topic_size//2]

    def default_hidden(self):
        """
        生成默认的初始隐藏状态
        """
        return torch.zeros(2, 1, self.emb_size, device=self.device)  # [2, 1, topic_size//2]

    def load_emb(self, emb):
        """
        加载预训练的词向量
        """
        self.embedding.weight.data.copy_(torch.from_numpy(emb))