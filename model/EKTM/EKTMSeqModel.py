import torch
from torch import nn
from constant import Constants

class EKTMSeqModel(nn.Module):
    def __init__(self, text_feature_size, know_emb_size, know_length, seq_hidden_size):
        super(EKTMSeqModel, self).__init__()
        self.text_feature_size = text_feature_size
        self.know_emb_size = know_emb_size
        self.seq_hidden_size = seq_hidden_size
        self.know_length = know_length
        # Knowledge memory matrix
        # knowledge_memory 的维度: [know_length, know_emb_size]
        self.knowledge_memory = nn.Parameter(torch.zeros(self.know_length, self.know_emb_size))
        # 将 knowledge_memory 的每个元素初始化为 [-1, 1] 区间内的均匀分布随机值
        self.knowledge_memory.data.uniform_(-1, 1)
        # GRU: 输入维度 [batch_size, seq_len, text_feature_size * 2]
        #      隐藏状态维度: [batch_size, seq_hidden_size]
        self.rnn = nn.GRU(self.text_feature_size * 2, seq_hidden_size, batch_first=True)
        # the first student state
        # h_initial 的维度: [know_length, seq_hidden_size]
        self.h_initial = nn.Parameter(torch.zeros(self.know_length, self.seq_hidden_size))
        # 初始化 h_initial 为 [-1, 1] 区间的均匀分布值
        self.h_initial.data.uniform_(-1, 1)
        # prediction layer
        # score_layer 的输入维度: [text_feature_size + seq_hidden_size]
        # score_layer 的输出维度: [1] (单个预测分数)
        self.score_layer = nn.Linear(text_feature_size + seq_hidden_size, 1)
        self.device = torch.device(f'cuda:{Constants.CUDA}' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    def forward(self, text_v, knowledge_v, result, h, beta=None):
        # text_v: [1, text_feature_size]
        # knowledge_v: [know_emb_size]
        # result: [1]
        # h: [1, know_length, seq_hidden_size] (初始隐藏状态)
        # beta: [know_length] (知识点权重)
        # 如果 h 是 None，使用初始隐藏状态
        text_v=text_v.unsqueeze(0) #text_v: [1, text_feature_size]
        if h is None:
            # h 的维度: [1, know_length, seq_hidden_size]
            h = self.h_initial.view(1, self.know_length, self.seq_hidden_size)

        # Step 1: 计算知识点权重 beta
        if beta is None:
            # 知识点记忆矩阵: knowledge_memory 的维度: [know_length, know_emb_size]
            # 知识点向量: knowledge_v 的维度: [know_emb_size]
            # 通过矩阵-向量乘法计算 beta: [know_length]
            beta = torch.matmul(self.knowledge_memory, knowledge_v)

            # 对 beta 进行 Softmax，得到权重 beta: [know_length]
            beta = nn.functional.softmax(beta, dim=0)
        # 每个知识状态都有权重，乘完后累加
        # [seq_hidden_size]
        s = torch.sum(beta.view(-1, 1) * h.view(self.know_length, self.seq_hidden_size), dim=0)
        pred_v = torch.cat([text_v.squeeze(0), s])
        predict_score = self.score_layer(pred_v)

        zeros = torch.zeros_like(text_v)
        x_r = torch.where(
            result >= 0.5,  # 如果 r >= 0.5，表示做对了
            torch.cat([text_v, zeros], dim=1),  # 拼接 [X, zeros]，形状为 [1,text_feature_size * 2]
            torch.cat([zeros, text_v], dim=1)  # 拼接 [zeros, X]，形状为 [1,text_feature_size * 2]
        )
        # x_r 的维度扩展为 [1, text_feature_size * 2]，与 beta 相乘后扩展
        # beta.view(-1, 1) -> [know_length, 1]
        # x_r.expand(know_length, -1) -> [know_length, text_feature_size * 2]
        x_r_b = beta.view(-1, 1) * x_r.expand(self.know_length, -1)  # [know_length, text_feature_size * 2]
        # 扩展 x 为 RNN 输入格式: [batch_size=know_length, seq_len=1, text_feature_size * 2]
        x_r_b = x_r_b.unsqueeze(1)  # [know_length, 1, text_feature_size * 2]
        # Step 4: 使用 RNN 更新隐藏状态
        # RNN 输入 x: [batch_size=know_length, seq_len=1, text_feature_size * 2]
        # RNN 初始隐藏状态 h: [1, know_length, seq_hidden_size]
        _, h = self.rnn(x_r_b, h)  # 更新 h 的维度 [1, know_length, seq_hidden_size]

        return predict_score, h

