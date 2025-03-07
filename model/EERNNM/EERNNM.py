import torch
import torch.nn as nn

# from model.BERTEmbedder import BERTEmbedder
from constant.Constants import EMBEDDING_MODEL
from model.EERNNM.EERNNMSeqModel import EERNNMSeqModel
from constant import Constants
from model.textEmbedding.BERTEmbedder import BERTEmbedder
from model.textEmbedding.OpenAIEmbedder import OpenAIEmbedder


class EERNNM(nn.Module):
    """
    得分预测的RNN模型
    """

    def __init__(self):
        super(EERNNM, self).__init__()
        # 创建 BERTEmbedder 实例，这样可以在其他地方复用它
        if EMBEDDING_MODEL == "openai":
            self.embedder = OpenAIEmbedder()
        else:
            # 默认使用 BERT embedder 或其他 embedder
            self.embedder = BERTEmbedder()
        self.seq_model = EERNNMSeqModel(Constants.TEXT_FEATURE_DIM, Constants.HIDDEN_DIM)
        # 将模型移动到 GPU（如果可用）
        self.device = torch.device(f'cuda:{Constants.CUDA}' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, item, result, hidden=None):
        # item: 做题记录信息 包括很多信息
        text = item.content
        result = result.to(self.device)
        if hidden is not None:
            hidden = hidden.to(self.device)
        # 使用 BERTEmbedder 获取文本嵌入
        text_v = self.embedder.get_embeddings(item.cid, item.qindex, text)
        s, hidden = self.seq_model(text_v, result, hidden)
        return s, hidden
