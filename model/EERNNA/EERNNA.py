import torch
from torch import nn

from constant import Constants
from constant.Constants import EMBEDDING_MODEL

from model.EERNNA.EERNNASeqModel import EERNNASeqModel
from model.textEmbedding.BERTEmbedder import BERTEmbedder
from model.textEmbedding.OpenAIEmbedder import OpenAIEmbedder


class EERNNA(nn.Module):
    """
    得分预测的RNN+attention模型
    """

    def __init__(self):
        super(EERNNA, self).__init__()
        # BERT编码器
        if EMBEDDING_MODEL == "openai":
            self.embedder = OpenAIEmbedder()
        else:
            # 默认使用 BERT embedder 或其他 embedder
            self.embedder = BERTEmbedder()
        # Attention序列模型
        self.seq_model = EERNNASeqModel(
            text_feature_dim=Constants.TEXT_FEATURE_DIM,  # BERT输出维度 768
            hidden_dim=Constants.HIDDEN_DIM  # RNN隐层维度 如 256
        )

        # 设备配置
        self.device = torch.device(f'cuda:{Constants.CUDA}' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, item, result, hidden=None):
        """
        输入:
            item: 做题记录信息
            result: 做题结果 [1]
            hidden: 包含(vs, hs)的元组,其中:
                vs: 历史题目向量 [历史长度, text_feature_dim]
                hs: 历史隐状态 [历史长度, hidden_dim]
        """
        text = item.content
        # 移动数据到设备
        result = result.to(self.device)
        # BERT编码得到题目向量 [1, 768]
        text_v = self.embedder.get_embeddings(item.cid, item.qindex, text)
        s, h = self.seq_model(text_v, result, hidden)
        # s 这道题预测对不对，h 学完这道题的状态
        if hidden is None:
            hidden = text_v, h
        else:
            text_s, hidden_s = hidden
            text_s = torch.cat([text_s, text_v])
            hidden_s = torch.cat([hidden_s, h])
            hidden = text_s, hidden_s
        return s, hidden
