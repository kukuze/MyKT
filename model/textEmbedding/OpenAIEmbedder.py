import os
from datetime import time

import torch
from constant import Constants
from model.textEmbedding.BaseEmbedder import BaseEmbedder
import http.client
import json


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, cache_dir: str = Constants.BASE_PATH + "/model/textEmbedding/openaiEmbeddingCache",
                 model_name: str = "text-embedding-3-large-768"):
        """
        初始化OpenAI嵌入器

        :param cache_dir: 缓存文件夹路径
        :param model_name: OpenAI嵌入模型名称text-embedding-3-small、text-embedding-3-large-3072
        """
        super().__init__(cache_dir, model_name)
        self.model_name = model_name
        self.device = torch.device(f'cuda:{Constants.CUDA}' if torch.cuda.is_available() else 'cpu')
        parts = model_name.split('-')  # 按 '-' 分割
        self.dimensions = int(parts[-1]) if parts[-1].isdigit() else None  # 提取 dimensions
        self.base_model_name = '-'.join(parts[:-1]) if parts[-1].isdigit() else model_name  # 去掉最后的数字部分
    def _embed_text(self, text: str) -> torch.Tensor:
        """
        使用OpenAI API计算文本的嵌入

        :param text: 待嵌入的文本
        :return: 文本嵌入张量，形状为 [num, dim]
        """
        # 确保输入的是字符串列表
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        ##因为最长为8191 且cf文本最长为10000，平均为1500，所以直接截断影响不大
        texts = [t[:8191] for t in texts]
        embeddings = []
        for t in texts:
            conn = http.client.HTTPSConnection(Constants.openai_url)
            payload = json.dumps({
                "model": self.base_model_name,
                "input": t,
                "dimensions":self.dimensions
            })
            headers = {
                'Authorization': 'Bearer ' + Constants.openai_api_key,
                'Content-Type': 'application/json'
            }
            conn.request("POST", "/v1/embeddings", payload, headers)
            res = conn.getresponse()
            response = json.loads(res.read().decode("utf-8"))
            # 提取嵌入向量
            embedding = response['data'][0]['embedding']

            # 转换为PyTorch张量，增加batch维度，并移动到指定设备
            embedding_tensor = torch.tensor(embedding, device=self.device).unsqueeze(0)
            embeddings.append(embedding_tensor)

        # 将所有文本的嵌入在batch维度上连接起来
        embeddings = torch.cat(embeddings, dim=0)

        return embeddings


import pandas as pd
from model.textEmbedding.OpenAIEmbedder import OpenAIEmbedder
import time
from tqdm import tqdm


def process_embeddings():
    # 初始化OpenAI embedder
    embedder = OpenAIEmbedder()

    # 读取Excel文件
    df = pd.read_excel("D:/MyKT/data/codeforces_problem_detail.xlsx")

    # 遍历每一行
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing embeddings"):
        try:
            cid = str(row['cid'])
            qindex = row['qindex']
            content = row['content']

            # 获取embedding
            embedder.get_embeddings(cid, qindex, content)

        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue


if __name__ == "__main__":
    Constants.CUDA = 0
    process_embeddings()

