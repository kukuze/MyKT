import torch
from torch import nn
from constant import Constants

class EERNNMSeqModel(nn.Module):
    """
    用于做题记录序列的 RNN（GRU）模型
    输入仅包含特征，没有 batch 和 sequence 长度的概念
    """

    def __init__(self, text_feature_size, rnn_hidden_size):
        super(EERNNMSeqModel, self).__init__()
        self.text_feature_size = text_feature_size  # 题目文本特征维度
        self.rnn_hidden_size = rnn_hidden_size  # RNN 隐藏状态维度
        # 定义 GRU 层，输入维度为 text_feature_size * 2（拼接后的文本特征）
        self.rnn = nn.GRU(text_feature_size * 2, rnn_hidden_size)
        # 定义用来预测分数的线性层
        self.score_layer = nn.Linear(rnn_hidden_size + text_feature_size, 1)
        self.device = torch.device(f'cuda:{Constants.CUDA}' if torch.cuda.is_available() else 'cpu')
        # 将模型移到正确的设备上
        self.to(self.device)
    def forward(self, X, r, h):
        """
        X: 题目文本特征，形状为 [1, text_feature_size]
        r: 学生的回答结果，0 或 1，表示是否做对，形状为 [1]
        h: RNN 隐状态，形状为 [1, rnn_hidden_size]（上一时刻的隐状态）
        """

        if h is None:
            h = self.default_hidden()  # 如果没有隐状态，则使用默认初始化

        # 拼接题目特征 X 和隐状态 h，作为预测的输入
        pred_X = torch.cat([X, h], dim=1)  # 拼接后维度为 [1,text_feature_size + rnn_hidden_size]

        # 通过线性层计算分数
        score = self.score_layer(pred_X)  # 输出分数，形状为 [1,1]

        # 创建零向量与 X 形状相同，用于后续拼接
        zeros = torch.zeros_like(X)  # 形状为 [1, text_feature_size]

        # 根据学生回答结果 r 拼接题目特征 X 和零向量
        x_r = torch.where(
            r >= 0.5,  # 如果 r >= 0.5，表示做对了
            torch.cat([X, zeros], dim=1),  # 拼接 [X, zeros]，形状为 [1,text_feature_size * 2]
            torch.cat([zeros, X], dim=1)  # 拼接 [zeros, X]，形状为 [1,text_feature_size * 2]
        )
        # 传入 GRU 计算更新后的隐状态
        _, h = self.rnn(x_r, h)  # GRU 输出 [1, rnn_hidden_size] 的隐状态
        # h[1, rnn_hidden_size]
        # score.squeeze(0) [1]
        return score.squeeze(0), h  # 返回预测的分数和新的隐状态

    def default_hidden(self):
        """
        初始化 RNN 隐状态，返回形状为 [1, rnn_hidden_size]，用于单样本推理
        """
        return torch.zeros(1, self.rnn_hidden_size, device=self.device)
