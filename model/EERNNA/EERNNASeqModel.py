import torch
from torch import nn

from constant import Constants


class EERNNASeqModel(nn.Module):
    """
    做题记录序列的RNN+attention模型单元（alpha：题面embedding点乘）
    """

    def __init__(self, text_feature_dim, hidden_dim):
        """
        Args:
            text_feature_dim: BERT输出的维度 (768)
            hidden_dim: RNN隐层维度
        """
        super(EERNNASeqModel, self).__init__()
        self.text_feature_dim = text_feature_dim
        self.hidden_dim = hidden_dim

        # RNN层 - 输入维度为 text_feature_dim*2 (拼接模式)
        self.rnn = nn.GRU(text_feature_dim * 2, hidden_dim)
        # 预测层
        self.score_layer = nn.Linear(text_feature_dim + hidden_dim, 1)
        self.device = torch.device(f'cuda:{Constants.CUDA}' if torch.cuda.is_available() else 'cpu')
        # 将模型移到正确的设备上
        self.to(self.device)
    def forward(self, text_v, result, hidden):
        """
        Args:
            text_v: 当前题目BERT向量 [1, text_feature_dim]
            result: 当前题目得分 [1]
            hidden: 包含(vs, hs)的元组,其中:
                vs: 历史题目向量 [历史长度, text_feature_dim]
                hs: 历史隐状态 [历史长度, hidden_dim]
        Returns:
            score: 预测分数 [1]
            h: 当前题目的隐状态 [1, hidden_dim]
        """
        if hidden is None:
            # 第一道题,没有历史信息
            h = self.default_hidden()
            attn_h = h
        else:
            text_s, hidden_s = hidden
            # 获取最后一个隐状态作为RNN的输入状态
            h = hidden_s[-1:, :]  # [1, hidden_dim]
            # Step 1: 计算相似度
            alpha = torch.mm(text_s, text_v.T)  # vs: [历史长度, 向量维度], text_v.T: [向量维度, 1]
            # alpha 的形状：[历史长度, 1]
            # Step 2: Softmax归一化相似度
            alpha = torch.nn.functional.softmax(alpha, dim=0)  # 保持每个历史题目分数归一化
            # alpha 的形状：[历史长度, 1]
            # Step 3: 加权求和历史状态
            # hs 的形状：[历史长度, hidden_dim]
            # alpha.T 将相似度转置为 [1, 历史长度]
            attn_h = torch.mm(alpha.T, hidden_s)  # 加权求和后 h_att 的形状：[1, hidden_dim]

        # 拼接当前题目向量和attention后的历史状态
        pred_v = torch.cat([text_v, attn_h], dim=1)  # [1, text_feature_dim + hidden_dim]

        score = self.score_layer(pred_v)  # [1,1]

        zeros = torch.zeros_like(text_v)  # [1, text_feature_dim]
        x_r = torch.where(
            result >= 0.5,  # 如果做对了
            torch.cat([text_v, zeros], dim=1),  # [1, text_feature_dim * 2]
            torch.cat([zeros, text_v], dim=1)  # [1, text_feature_dim * 2]
        )
        _, h = self.rnn(x_r, h)
        # h[1, rnn_hidden_size]
        # score.squeeze(0) [1]
        return score.squeeze(0), h

    def default_hidden(self):
        return torch.zeros(1, self.hidden_dim, device=self.device)
