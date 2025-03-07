import os
import torch
from transformers import BertTokenizer, BertModel

from constant import Constants
from model.textEmbedding.BaseEmbedder import BaseEmbedder


class BERTEmbedder(BaseEmbedder):
    def __init__(self, cache_dir: str = Constants.BASE_PATH + "/model/textEmbedding/bertEmbeddingCache",
                 model_name: str = 'bert-base-uncased'):
        """
        初始化BERT嵌入器

        :param cache_dir: 缓存文件夹路径
        :param model_name: BERT模型名称
        """
        # 使用父类构造函数，它会自动处理模型特定的缓存路径
        super().__init__(cache_dir, model_name)

        # 加载分词器和模型
        self.tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=Constants.BERT_MODEL_CACHE_PATH)
        self.model = BertModel.from_pretrained(model_name, cache_dir=Constants.BERT_MODEL_CACHE_PATH)

        # 设备和模式配置
        self.device = torch.device(f'cuda:{Constants.CUDA}' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()  # 设置模型为评估模式

    def _embed_text(self, text: str, max_length: int = 512) -> torch.Tensor:
        """
        实现文本嵌入的具体方法

        :param text: 待嵌入的文本
        :param max_length: 最大token长度
        :return: 文本嵌入张量，形状为 [num, dim]
        """
        # 设置片段大小为510，预留两个位置给[CLS]和[SEP]
        chunk_size = max_length - 2

        # 确保输入的'text'是字符串列表
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        embeddings = []  # 用于存储每个文本的嵌入表示
        for t in texts:
            # 对文本进行分词，不添加特殊标记（[CLS], [SEP]）
            encoding = self.tokenizer(t, return_tensors='pt', add_special_tokens=False)
            input_ids = encoding['input_ids'][0]  # 形状：(seq_len)

            # 将input_ids分割成大小为'chunk_size'的片段
            input_id_chunks = input_ids.split(chunk_size)

            chunk_embeddings = []  # 用于存储每个片段的[CLS]嵌入
            for chunk in input_id_chunks:
                # 在片段的开头和结尾添加[CLS]和[SEP]标记
                chunk = torch.cat([
                    torch.tensor([self.tokenizer.cls_token_id]),  # [CLS]的token id
                    chunk,
                    torch.tensor([self.tokenizer.sep_token_id])  # [SEP]的token id
                ], dim=0)

                # 创建注意力掩码（真实的token位置为1）
                attention_mask = torch.ones(chunk.shape, dtype=torch.long)

                # 添加batch维度并移动到正确的设备上
                chunk = chunk.unsqueeze(0).to(self.device)  # 形状：(1, chunk_seq_len)
                attention_mask = attention_mask.unsqueeze(0).to(self.device)  # 形状：(1, chunk_seq_len)

                with torch.no_grad():
                    outputs = self.model(input_ids=chunk, attention_mask=attention_mask)

                # 提取[CLS]嵌入（第一个token的输出）
                cls_embedding = outputs.last_hidden_state[:, 0, :]  # 形状：(1, hidden_size)
                chunk_embeddings.append(cls_embedding)

            # 将所有片段的[CLS]嵌入堆叠起来：形状 (num_chunks, hidden_size)
            chunk_embeddings = torch.cat(chunk_embeddings, dim=0)

            # 对片段的[CLS]嵌入进行max pooling，得到单一的文本表示
            text_embedding, _ = torch.max(chunk_embeddings, dim=0, keepdim=True)  # 形状：(1, hidden_size)

            # 将池化后的嵌入添加到列表中
            embeddings.append(text_embedding)

        # 将所有文本的嵌入在batch维度上连接起来：形状 (batch_size, hidden_size)
        embeddings = torch.cat(embeddings, dim=0)

        return embeddings
