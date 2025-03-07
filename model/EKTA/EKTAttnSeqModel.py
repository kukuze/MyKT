import torch
from torch import nn
from constant import Constants

class EKTAttnSeqModel(nn.Module):
    """
    Student seq modeling combined with exercise texts and knowledge point
    """

    def __init__(self, text_feature_size, know_emb_size, know_length, seq_hidden_size):
        super(EKTAttnSeqModel, self).__init__()
        self.text_feature_size = text_feature_size
        self.know_emb_size = know_emb_size
        self.seq_hidden_size = seq_hidden_size
        self.know_length = know_length

        # Knowledge memory matrix
        self.knowledge_memory = nn.Parameter(torch.zeros(self.know_length, self.know_emb_size))
        self.knowledge_memory.data.uniform_(-1, 1)

        self.rnn = nn.GRU(self.text_feature_size * 2, seq_hidden_size, batch_first=True)
        # the first student state
        # h_initial 的维度: [know_length, seq_hidden_size]
        self.h_initial = nn.Parameter(torch.zeros(self.know_length, self.seq_hidden_size))
        # 初始化 h_initial 为 [-1, 1] 区间的均匀分布值
        self.h_initial.data.uniform_(-1, 1)

        self.score_layer = nn.Linear(text_feature_size + seq_hidden_size, 1)
        # 将模型移动到 GPU（如果可用）
        self.device = torch.device(f'cuda:{Constants.CUDA}' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, text_v, knowledge_v, result, hidden):
        # text_v: [1, text_feature_size]
        # knowledge_v: [know_emb_size]
        # result: [1]
        # hidden: 包含(text_s, hidden_s)的元组,其中:
        #                 text_s: 历史题目向量 [历史长度 , text_feature_dim]
        #                 hidden_s: 历史隐状态 [历史长度, kenow_length, hidden_dim]
        if hidden is None:
            #h代表着学生学完上一道题目的知识状态矩阵
            h = self.h_initial.view(1, self.know_length, self.seq_hidden_size)
            #H_att代表着过去知识状态的加权
            H_att = self.h_initial
        else:
            text_s, hidden_s = hidden
            h = hidden_s[-1:, :]  # [1,know_length,hidden_dim]
            sim = torch.matmul(text_s, text_v.T)  # [历史长度, 1]
            # 对相似度进行 softmax
            alpha = torch.nn.functional.softmax(sim, dim=0)  # [历史长度, 1]
            # 扩展 alpha 的维度，使其能广播到 hs
            alpha = alpha.view(-1, 1, 1)  # [历史长度, 1, 1]

            # hs: [历史长度, know_length, hidden_dim]
            # 对 hs 加权求和
            H_att = torch.sum(alpha * hidden_s, dim=0)  # [know_length, hidden_dim]

        beta = torch.matmul(self.knowledge_memory, knowledge_v)

        # 对 beta 进行 Softmax，得到权重 beta: [know_length]
        beta = nn.functional.softmax(beta, dim=0)
        # s [hidden_dim]
        s = torch.sum(beta.view(-1, 1) * H_att, dim=0)
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
        _, h = self.rnn(x_r_b, h)
        return predict_score, h
