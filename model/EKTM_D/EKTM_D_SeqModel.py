import torch
from torch import nn
from constant import Constants
import torch.nn.functional as F


class EKTM_D_SeqModel(nn.Module):
    def __init__(self, text_feature_size, know_emb_size, know_length, diff_emb_size, diff_length, seq_hidden_size):
        super(EKTM_D_SeqModel, self).__init__()
        self.text_feature_size = text_feature_size
        self.know_emb_size = know_emb_size
        self.seq_hidden_size = seq_hidden_size
        self.know_length = know_length
        self.diff_emb_size = diff_emb_size
        self.diff_length = diff_length

        # Knowledge memory matrix
        # knowledge_memory 的维度: [know_length, know_emb_size]
        self.knowledge_memory = nn.Parameter(torch.zeros(self.know_length, self.know_emb_size))
        # 将 knowledge_memory 的每个元素初始化为 [-1, 1] 区间内的均匀分布随机值
        self.knowledge_memory.data.uniform_(-1, 1)
        # Difficulty memory matrix
        # difficulty_memory 的维度: [diff_length, diff_emb_size]
        self.difficulty_memory = nn.Parameter(torch.zeros(self.diff_length, self.diff_emb_size))
        # 将 difficulty_memory 的每个元素初始化为 [-1, 1] 区间内的均匀分布随机值
        self.difficulty_memory.data.uniform_(-1, 1)

        # GRU 的注释更新
        # GRU: 输入维度 [batch_size=know_length*diff_length, seq_len=1, text_feature_size*2]
        #      隐藏状态维度: [1, know_length*diff_length, seq_hidden_size]
        self.rnn = nn.GRU(100 * 2, seq_hidden_size, batch_first=True)
        # the first student state
        # h_initial 的维度: [know_length, diff_length, seq_hidden_size]
        self.h_initial = nn.Parameter(torch.zeros(self.know_length, self.diff_length, self.seq_hidden_size))
        # 初始化 h_initial 为 [-1, 1] 区间的均匀分布值
        self.h_initial.data.uniform_(-1, 1)
        self.text_transform = nn.Linear(text_feature_size, 100)
        # prediction layer
        # score_layer 的输入维度: [text_feature_size + seq_hidden_size]
        # score_layer 的输出维度: [1] (单个预测分数)
        self.score_layer = nn.Linear(100 + know_emb_size + diff_emb_size + seq_hidden_size, 1)
        self.device = torch.device(f'cuda:{Constants.CUDA}' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, text_v, knowledge_v, difficulty_v, result, h, get_status=False):
        # text_v: [1,know_emb_size]
        # knowledge_v: [know_emb_size]
        # difficulty_v: [diff_emb_size]
        # difficulty_memory: [diff_length, diff_emb_size]
        # knowledge_memory: [know_length, know_emb_size]
        # h: [know_length, diff_length, seq_hidden_size] 或 None
        # 对 text_v 进行可学习的变换
        text_v = self.text_transform(text_v)
        if h is None:
            # h 的维度: [know_length, diff_length, seq_hidden_size]
            h = self.h_initial
        if get_status:
            return self.compute_all_knowledge_states(h)
        # 调用 compute_beta 函数
        # beta [know_length, diff_length]
        beta = self.compute_beta(knowledge_v, difficulty_v)
        # 每个知识状态都有权重，乘完后累加
        # s: [seq_hidden_size]
        s = torch.sum(
            beta.unsqueeze(-1) * h,
            dim=(0, 1)
        )
        pred_v = torch.cat([text_v.squeeze(0), knowledge_v, difficulty_v, s])
        # predict_score: [1]
        predict_score = self.score_layer(pred_v)

        zeros = torch.zeros_like(text_v)
        x_r = torch.where(
            result >= 0.5,  # 如果 r >= 0.5，表示做对了
            torch.cat([text_v, zeros], dim=1),  # 拼接 [X, zeros]，形状为 [1,text_feature_size * 2]
            torch.cat([zeros, text_v], dim=1)  # 拼接 [zeros, X]，形状为 [1,text_feature_size * 2]
        )
        # 1. 扩展 x_r 到知识点和难度的维度
        x_r_expanded = x_r.expand(self.know_length, self.diff_length, -1)
        # [know_length, diff_length, text_feature_size * 2]
        # 2. 扩展 beta 以匹配 x_r 的维度
        beta_expanded = beta.unsqueeze(-1)
        # [know_length, diff_length, 1]
        # 3. 计算加权的输入向量
        x_r_b = beta_expanded * x_r_expanded
        # [know_length, diff_length, text_feature_size * 2]

        # 4. 重塑为 RNN 输入格式
        x_r_b = x_r_b.view(-1, 1, x_r.size(-1))
        # x_r_b: [know_length * diff_length, 1, text_feature_size * 2]
        # 5. 重塑 h 以匹配新的输入格式
        h = h.view(-1, self.seq_hidden_size).unsqueeze(0)
        # [1, know_length * diff_length, seq_hidden_size]
        _, h = self.rnn(x_r_b, h)  # 更新 h 的维度 [1, know_length, seq_hidden_size]
        # 7. 将 h_new 重塑回原始维度
        h = h.view(self.know_length, self.diff_length, self.seq_hidden_size)
        # h: [know_length, diff_length, seq_hidden_size]

        return predict_score, h

    def compute_beta(self, knowledge_v, difficulty_v):
        # 计算知识点相似度
        knowledge_similarity = torch.matmul(self.knowledge_memory, knowledge_v)

        # 计算难度相似度
        difficulty_similarity = torch.matmul(self.difficulty_memory, difficulty_v)

        # 使用外积创建 beta
        beta = torch.mul(
            knowledge_similarity.unsqueeze(1),
            difficulty_similarity.unsqueeze(0)
        )
        # 对展平的向量进行 softmax
        beta_flat = beta.view(-1)
        beta_softmax_flat = F.softmax(beta_flat, dim=0)

        # 将一维向量恢复为原始的矩阵形状
        beta = beta_softmax_flat.view(beta.size())

        return beta

    def compute_all_knowledge_states(self, h):
        """
        封装 get_status=True 的逻辑，用于计算所有知识状态的掌握程度。
        """
        # 创建零向量作为占位符
        text_v_zero = torch.zeros((self.know_length, self.diff_length, 100), device=h.device)
        knowledge_v_zero = torch.zeros((self.know_length, self.diff_length, self.know_emb_size), device=h.device)
        difficulty_v_zero = torch.zeros((self.know_length, self.diff_length, self.diff_emb_size), device=h.device)

        # 拼接特征
        combined_input = torch.cat([
            text_v_zero,  # [know_length, diff_length, 100]
            knowledge_v_zero,  # [know_length, diff_length, know_emb_size]
            difficulty_v_zero,  # [know_length, diff_length, diff_emb_size]
            h  # [know_length, diff_length, seq_hidden_size]
        ], dim=-1)  # [know_length, diff_length, total_input_dim]

        # 使用 score_layer 处理每个位置
        return self.score_layer(combined_input), 0  # [know_length, diff_length, 1]
