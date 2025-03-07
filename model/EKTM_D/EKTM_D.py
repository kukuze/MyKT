import torch
from torch import nn

from constant import Constants
from constant.Constants import EMBEDDING_MODEL
from model.DifficultyModel import DifficultyEncoder
from model.EKTM_D.EKTM_D_SeqModel import EKTM_D_SeqModel
from model.KnowledgeModel import KnowledgeModel
from model.textEmbedding.BERTEmbedder import BERTEmbedder
from model.textEmbedding.OpenAIEmbedder import OpenAIEmbedder


class EKTM_D(nn.Module):
    """
    Knowledge Tracing Model with Markov property combined with exercise texts and knowledge concepts and difficulty
    """

    def __init__(self):
        super(EKTM_D, self).__init__()
        if EMBEDDING_MODEL == "openai":
            self.embedder = OpenAIEmbedder()
        else:
            # 默认使用 BERT embedder 或其他 embedder
            self.embedder = BERTEmbedder()
        # 知识点嵌入模块
        self.knowledge_model = KnowledgeModel(Constants.TAGS_NUM, Constants.TAG_EMB_DIM)
        # 难度编码模块
        self.difficulty_encoder = DifficultyEncoder(
            difficulty_dim=Constants.DIFFICULTY_EMB_DIM,  # 在常量中定义难度嵌入维度
            min_diff=Constants.DIFFS_MIN,
            max_diff=Constants.DIFFS_MAX
        )
        self.seq_model = EKTM_D_SeqModel(
            text_feature_size=Constants.TEXT_FEATURE_DIM,  # 题目特征维度
            know_emb_size=Constants.TAG_EMB_DIM,  # 知识点特征维度
            know_length=Constants.TAGS_NUM,  # 知识点总数
            seq_hidden_size=Constants.HIDDEN_DIM,  # 隐层维度
            diff_emb_size=Constants.DIFFICULTY_EMB_DIM,  # 传入难度嵌入维度
            diff_length=Constants.DIFFS_NUM
        )
        # 将模型移动到 GPU（如果可用）
        self.device = torch.device(f'cuda:{Constants.CUDA}' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, item, result, hidden=None, get_status=False):
        # item: 做题记录信息
        text = item.content
        knowledge = item.tags
        difficulty = item.difficulty

        difficulty_v = self.difficulty_encoder(difficulty)
        knowledge_v = self.knowledge_model(knowledge)
        text_v = self.embedder.get_embeddings(item.cid, item.qindex, text)
        s, h = self.seq_model(text_v, knowledge_v, difficulty_v, result, hidden, get_status)
        return s, h
