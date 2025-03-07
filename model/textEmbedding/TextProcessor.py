import re

import numpy as np
import torch

from util import load_glove_embedding


class TextProcessor:
    def __init__(self,emb_file):
        # 加载预训练的词向量和词表
        self.wcnt, self.emb_size, self.words, self.embs = load_glove_embedding(emb_file)
        # wcnt: 词表大小
        # emb_size: 词向量维度（如300）
        # words: 词表列表 ["the", "a", "is", ...]
        # embs: 词向量矩阵 [wcnt, emb_size]

        # 构建词到索引的映射字典
        self.word2idx = {word: idx for idx, word in enumerate(self.words)}
        # 例如: {"the": 0, "a": 1, "is": 2, ...}
        self.unk_token = '<unk>'  # 用于未知词
        # 为UNK创建词向量（可以选择不同的初始化策略）
        if self.unk_token not in self.word2idx:
            self.word2idx[self.unk_token] = len(self.word2idx)
            # 方法1：使用所有词向量的平均值
            unk_vector = np.mean(self.embs, axis=0)
            # 或者 方法2：随机初始化
            # unk_vector = np.random.normal(0, 0.1, self.emb_size)
            # 将UNK向量添加到词向量矩阵
            self.embs = np.vstack([self.embs, unk_vector])
            self.wcnt += 1  # 更新词表大小
        self.cache = {}
    def process_text(self, text):
        """
        处理文本：分词并转换为索引序列，并输出未找到的单词
        """
        # 分词
        if text in self.cache:
            return self.cache[text]
        text = re.sub(r'([.,!?:;])', r' \1 ', text)
        text = re.sub(r'\${3}.*?\${3}', ' ', text)
        words = text.lower().split()

        # # 记录未找到的单词
        # unknown_words = [word for word in words if word not in self.word2idx]
        # if unknown_words:
        #     print(f"未找到的单词: {unknown_words}")

        # 转换为索引
        indices = [self.word2idx.get(word, self.word2idx[self.unk_token])
                   for word in words]

        result = torch.tensor(indices)

        # Cache the result
        self.cache[text] = result

        return result