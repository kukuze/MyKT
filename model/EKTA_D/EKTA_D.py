import torch
from torch import nn

from constant import Constants
from constant.Constants import EMBEDDING_MODEL
from model.DifficultyModel import DifficultyEncoder
from model.EKTA_D.EKTAttn_DSeqModel import EKTAttn_DSeqModel
from model.KnowledgeModel import KnowledgeModel
from model.textEmbedding.BERTEmbedder import BERTEmbedder
from model.textEmbedding.OpenAIEmbedder import OpenAIEmbedder
from util import check_memory


class EKTA_D(nn.Module):
    """
    Knowledge Tracing Model with Attention mechnaism combined with exercise texts and knowledge concepts
    """

    def __init__(self):
        super(EKTA_D, self).__init__()
        if EMBEDDING_MODEL == "openai":
            self.embedder = OpenAIEmbedder()
        else:
            # 默认使用 BERT embedder 或其他 embedder
            self.embedder = BERTEmbedder()
        # 知识点嵌入模块
        self.knowledge_model = KnowledgeModel(Constants.TAGS_NUM, Constants.TAG_EMB_DIM)
        # 添加难度编码模块
        self.difficulty_encoder = DifficultyEncoder(
            difficulty_dim=Constants.DIFFICULTY_EMB_DIM,
            min_diff=Constants.DIFFS_MIN,
            max_diff=Constants.DIFFS_MAX
        )
        # student seq module
        self.seq_model = EKTAttn_DSeqModel(text_feature_size=Constants.TEXT_FEATURE_DIM,  # 题目特征维度
                                           know_emb_size=Constants.TAG_EMB_DIM,  # 知识点特征维度
                                           know_length=Constants.TAGS_NUM,  # 知识点总数
                                           seq_hidden_size=Constants.HIDDEN_DIM,  # 隐层维度
                                           diff_emb_size=Constants.DIFFICULTY_EMB_DIM,
                                           diff_length=Constants.DIFFS_NUM
                                           )
        self.history = None
        # 将模型移动到 GPU（如果可用）
        self.device = torch.device(f'cuda:{Constants.CUDA}' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.sum_num = 0

    def init_history(self):
        """Initialize empty history container"""
        self.history = {
            'text_s': [],
            'hidden_s': [],
            'diff_s': [],
            'know_s': [],
            'times': []
        }

    def update_history(self, text_v, h, difficulty_v, knowledge_v, current_time):
        """Update history with new information"""
        if self.history is None:
            self.init_history()
        # Add current states to history
        self.history['text_s'].append(text_v.detach())
        self.history['hidden_s'].append(h.detach())
        self.history['diff_s'].append(difficulty_v.detach())
        self.history['know_s'].append(knowledge_v.detach())
        self.history['times'].append(current_time.detach())

    def get_history_tensor(self):
        """Convert history lists to tensor format"""
        if self.history is None or len(self.history['text_s']) == 0:
            return None

        history_tensors = (
            torch.stack(self.history['text_s']).to(self.device),
            torch.stack(self.history['hidden_s']).to(self.device),
            torch.stack(self.history['diff_s']).to(self.device),
            torch.stack(self.history['know_s']).to(self.device),
            torch.stack(self.history['times']).to(self.device)
        )
        return history_tensors
        # # Print memory usage for each tensor in history_info
        # for i, name in enumerate(['text_s', 'hidden_s', 'diff_s', 'know_s', 'times']):
        #     tensor = history_tensors[i]
        #     size_mb = tensor.element_size() * tensor.nelement() / (1024 * 1024)  # Convert to MB
        #     # print(f"{name} tensor shape: {tensor.shape}, Memory usage: {size_mb:.2f} MB")
        #     self.sum_num+=size_mb
        #     print(self.sum_num)

    def forward(self, item, result, hidden=None):
        text_v = self.embedder.get_embeddings(item.cid, item.qindex, item.content)
        knowledge_v = self.knowledge_model(item.tags)
        difficulty_v = self.difficulty_encoder(item.difficulty)
        current_time = torch.tensor([item.submitTime], device=self.device)
        history_info = self.get_history_tensor()
        s, h = self.seq_model(text_v, knowledge_v, difficulty_v, current_time, result, hidden, history_info)
        self.update_history(text_v, h, difficulty_v, knowledge_v, current_time)
        return s, h
