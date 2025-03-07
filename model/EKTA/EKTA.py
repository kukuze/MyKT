import torch
from torch import nn

from constant import Constants
from constant.Constants import EMBEDDING_MODEL
from model.EKTA.EKTAttnSeqModel import EKTAttnSeqModel
from model.KnowledgeModel import KnowledgeModel
from model.textEmbedding.BERTEmbedder import BERTEmbedder
from model.textEmbedding.OpenAIEmbedder import OpenAIEmbedder


class EKTA(nn.Module):
    """
    Knowledge Tracing Model with Attention mechnaism combined with exercise texts and knowledge concepts
    """

    def __init__(self):
        super(EKTA, self).__init__()
        if EMBEDDING_MODEL == "openai":
            self.embedder = OpenAIEmbedder()
        else:
            # 默认使用 BERT embedder 或其他 embedder
            self.embedder = BERTEmbedder()
        # 知识点嵌入模块
        self.knowledge_model = KnowledgeModel(Constants.TAGS_NUM, Constants.TAG_EMB_DIM)
        # student seq module
        self.seq_model = EKTAttnSeqModel(text_feature_size=Constants.TEXT_FEATURE_DIM,  # 题目特征维度
                                         know_emb_size=Constants.TAG_EMB_DIM,  # 知识点特征维度
                                         know_length=Constants.TAGS_NUM,  # 知识点总数
                                         seq_hidden_size=Constants.HIDDEN_DIM  # 隐层维度
                                         )
        # 将模型移动到 GPU（如果可用）
        self.device = torch.device(f'cuda:{Constants.CUDA}' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, item, result, hidden=None):
        text = item.content
        knowledge = item.tags
        knowledge_v = self.knowledge_model(knowledge)
        text_v = self.embedder.get_embeddings(item.cid, item.qindex, text)

        s, h = self.seq_model(text_v, knowledge_v, result, hidden)
        if hidden is None:
            hidden = text_v, h
        else:
            text_s, hidden_s = hidden
            text_s = torch.cat([text_s, text_v.detach()])
            hidden_s = torch.cat([hidden_s, h.detach()])
            hidden = text_s, hidden_s
        return s, hidden
