import torch
from torch import nn

from constant import Constants
from constant.Constants import EMBEDDING_MODEL, TEXT_FEATURE_DIM
from model.EKTM.EKTMSeqModel import EKTMSeqModel
from model.KnowledgeModel import KnowledgeModel
from model.textEmbedding.BERTEmbedder import BERTEmbedder
from model.textEmbedding.OpenAIEmbedder import OpenAIEmbedder
from model.textEmbedding.RNNTextEmbedder import RNNTextEmbedder
from model.textEmbedding.TextProcessor import TextProcessor
from util import load_embedding, load_glove_embedding


class EKTM(nn.Module):
    """
    Knowledge Tracing Model with Markov property combined with exercise texts and knowledge concepts
    """

    def __init__(self):
        super(EKTM, self).__init__()
        self.text_processor = TextProcessor(
            emb_file=r"D:\MyKT\model\textEmbedding\glove.6B.300d.txt"
        )
        self.text_model = RNNTextEmbedder(
            wcnt=self.text_processor.wcnt,
            emb_size=self.text_processor.emb_size,
            topic_size=Constants.TEXT_FEATURE_DIM
        )
        self.text_model.load_emb(self.text_processor.embs)  # 使用预训练词向量 + GRU处理文本
        # 知识点嵌入模块
        self.knowledge_model = KnowledgeModel(Constants.TAGS_NUM, Constants.TAG_EMB_DIM)
        self.seq_model = EKTMSeqModel(
            text_feature_size=Constants.TEXT_FEATURE_DIM,  # 题目特征维度
            know_emb_size=Constants.TAG_EMB_DIM,  # 知识点特征维度
            know_length=Constants.TAGS_NUM,  # 知识点总数
            seq_hidden_size=Constants.HIDDEN_DIM  # 隐层维度
        )
        # 将模型移动到 GPU（如果可用）
        self.device = torch.device(f'cuda:{Constants.CUDA}' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, item, result, hidden=None):
        # item: 做题记录信息
        text = item.content
        knowledge = item.tags
        knowledge_v = self.knowledge_model(knowledge)
        # text_v = self.embedder.get_embeddings(item.cid, item.qindex, text)
        # 1. 处理文本
        text_indices = self.text_processor.process_text(text).to(self.device)
        text_h = self.text_model.default_hidden() # [2, 1, TEXT_FEATURE_DIM//2]
        text_v, _ = self.text_model(text_indices, text_h) #[TEXT_FEATURE_DIM]
        s, h = self.seq_model(text_v, knowledge_v, result, hidden)
        return s, h
