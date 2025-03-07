import torch
from torch import nn
from constant import Constants
import torch.nn.functional as F


class EKTAttn_DSeqModel(nn.Module):
    """
    Student seq modeling combined with exercise texts and knowledge point
    """

    def __init__(self, text_feature_size, know_emb_size, know_length, seq_hidden_size, diff_emb_size, diff_length):
        super(EKTAttn_DSeqModel, self).__init__()
        """
            Args:
                text_feature_size: 文本特征维度
                know_emb_size: 知识点嵌入维度
                know_length: 知识点数量
                seq_hidden_size: 序列隐藏状态维度
                diff_emb_size: 难度嵌入维度
                diff_length: 难度等级数量
            """
        self.text_feature_size = text_feature_size
        self.know_emb_size = know_emb_size
        self.seq_hidden_size = seq_hidden_size
        self.know_length = know_length
        self.diff_emb_size = diff_emb_size
        self.diff_length = diff_length
        # Knowledge memory matrix
        self.knowledge_memory = nn.Parameter(torch.zeros(self.know_length, self.know_emb_size))
        self.knowledge_memory.data.uniform_(-1, 1)
        # 添加难度记忆矩阵
        self.difficulty_memory = nn.Parameter(torch.zeros(self.diff_length, self.diff_emb_size))
        self.difficulty_memory.data.uniform_(-1, 1)
        # 添加三种注意力的权重参数
        self.attention_weights = nn.Parameter(torch.ones(3))  # [text, difficulty, knowledge]
        # 添加时间衰减参数
        self.time_decay_factor = nn.Parameter(torch.tensor(0.1))
        self.rnn = nn.GRU(self.text_feature_size * 2, seq_hidden_size, batch_first=True)
        # the first student state
        # 更新初始状态维度
        self.h_initial = nn.Parameter(torch.zeros(self.know_length, self.diff_length,
                                                  self.seq_hidden_size))
        self.h_initial.data.uniform_(-1, 1)
        self.score_layer = nn.Linear(text_feature_size +diff_emb_size+know_emb_size+ seq_hidden_size * 2, 1)
        # 将模型移动到 GPU（如果可用）
        self.device = torch.device(f'cuda:{Constants.CUDA}' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, text_v, knowledge_v, difficulty_v, current_time, result, hidden, history_info=None):
        """
            前向传播函数
            Args:
                text_v: [1, text_feature_size] 当前题目的文本特征
                knowledge_v: [know_emb_size] 当前题目的知识点特征
                difficulty_v: [diff_emb_size] 当前题目的难度特征
                current_time: 当前时间戳
                result: [1] 当前题目的作答结果
                hidden: [know_length, diff_length, seq_hidden_size] Just the previous hidden state
                history_info: tuple (text_s, hidden_s, difficulty_s, knowledge_s, times) Historical information
                包含历史信息的元组 (text_s, hidden_s, difficulty_s, knowledge_s, times)，其中：
                        text_s: [历史长度, text_feature_dim] 历史题目的文本特征序列
                        hidden_s: [历史长度, know_length, diff_length, hidden_dim] 历史隐藏状态序列
                        difficulty_s: [历史长度, diff_emb_size] 历史题目的难度特征序列
                        knowledge_s: [历史长度, know_emb_size] 历史题目的知识点特征序列
                        times: [历史长度,1] 历史题目的时间戳序列
            Returns:
                predict_score: [1] 预测分数
                h: [know_length, diff_length, seq_hidden_size] 更新后的隐藏状态
            """
        if hidden is None:
            h = self.h_initial
            H_att = self.h_initial
        else:
            text_s, hidden_s, difficulty_s, knowledge_s, times = history_info
            h = hidden_s[-1]  # [know_length, diff_length, hidden_dim]
            # 计算时间衰减
            if current_time is not None and times is not None:
                # current_time: [1]
                # times: [history_len,1]
                time_intervals = current_time - times  # [history_len,1]

                # time_decay_factor: 标量
                # time_intervals: [history_len]
                decay_factors = torch.exp(-self.time_decay_factor * time_intervals)  # [history_len,1]

                # 调整维度以便与 hidden_s 广播
                # [history_len] -> [history_len, 1, 1, 1]
                decay_factors = decay_factors.view(-1, 1, 1, 1)
            else:
                decay_factors = 1.0  # 标量

            # 计算综合注意力权重
            alpha = self.compute_attention_weights(
                text_s,  # [history_len, text_feature_dim]
                text_v,  # [1, text_feature_dim]
                difficulty_s,  # [history_len, diff_emb_size]
                difficulty_v,  # [diff_emb_size]
                knowledge_s,  # [history_len, know_emb_size]
                knowledge_v  # [know_emb_size]
            )  # 返回: [history_len, 1]
            # [history_len, 1] -> [history_len, 1, 1, 1]
            alpha = alpha.view(-1, 1, 1, 1)
            # 应用时间衰减和注意力权重
            # hidden_s: [history_len, know_length, diff_length, hidden_dim]
            # alpha: [history_len, 1, 1, 1]
            # decay_factors: [history_len, 1, 1, 1]
            # 通过广播机制:
            # alpha * decay_factors: [history_len, 1, 1, 1]
            # (alpha * decay_factors) * hidden_s: [history_len, know_length, diff_length, hidden_dim]
            # torch.sum(..., dim=0): [know_length, diff_length, hidden_dim]
            H_att = torch.sum(alpha * decay_factors * hidden_s, dim=0)
        # 调用 compute_beta 函数
        # beta [know_length, diff_length]
        beta = self.compute_beta(knowledge_v, difficulty_v)
        # 每个知识状态都有权重，乘完后累加
        # s: [seq_hidden_size]
        s_history = torch.sum(
            beta.unsqueeze(-1) * H_att,
            dim=(0, 1)
        )
        s_current = torch.sum(
            beta.unsqueeze(-1) * h,
            dim=(0, 1)
        )
        pred_v = torch.cat([
            text_v.squeeze(0),  # 当前题目特征
            difficulty_v,  # 当前题目难度特征
            knowledge_v,  # 当前题目知识点特征
            s_history,  # 历史知识状态
            s_current  # 当前知识状态
        ])
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
        # 更新 h 的维度 [1, know_length * diff_length, seq_hidden_size]
        _, h = self.rnn(x_r_b, h)
        # 7. 将 h_new 重塑回原始维度
        h = h.view(self.know_length, self.diff_length, self.seq_hidden_size)
        # h: [know_length, diff_length, seq_hidden_size]

        return predict_score, h

    def compute_beta(self, knowledge_v, difficulty_v):
        """
           计算知识点和难度的联合注意力权重
           Args:
               knowledge_v: [know_emb_size] 知识点向量
               difficulty_v: [diff_emb_size] 难度向量
           Returns:
               beta: [know_length, diff_length] 联合注意力权重
           """
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

    def compute_attention_weights(self, text_s, text_v, difficulty_s, difficulty_v,
                                  knowledge_s, knowledge_v):
        """
        计算注意力权重
        Args:
            text_s: [history_len, dim]
            text_v: [1, dim]
            difficulty_s: [history_len, dim]
            difficulty_v: [dim]
            knowledge_s: [history_len, dim]
            knowledge_v: [dim]
        Returns:
            attention_weights: [history_len, 1]
        """
        # 调整 difficulty_v 和 knowledge_v 的维度为 [1, dim]
        text_s = text_s.squeeze(1)
        difficulty_v = difficulty_v.unsqueeze(0)  # [1, dim]
        knowledge_v = knowledge_v.unsqueeze(0)  # [1, dim]

        # 计算三种相似度
        text_sim = torch.matmul(text_s, text_v.T)  # [history_len, 1]
        difficulty_sim = torch.matmul(difficulty_s, difficulty_v.T)  # [history_len, 1]
        knowledge_sim = torch.matmul(knowledge_s, knowledge_v.T)  # [history_len, 1]

        # 将三种相似度组合
        weights = torch.softmax(self.attention_weights, dim=0)  # [3]
        combined_sim = (weights[0] * text_sim +
                        weights[1] * difficulty_sim +
                        weights[2] * knowledge_sim)  # [history_len, 1]

        return torch.softmax(combined_sim, dim=0)  # [history_len, 1]
