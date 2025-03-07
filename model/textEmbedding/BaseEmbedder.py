import os
import pickle
from abc import ABC, abstractmethod
from typing import Optional
import torch

from constant import Constants


class BaseEmbedder(ABC):
    def __init__(self, cache_dir: str, model_name: str):
        """
        Initialize the base embedder with caching and model-specific storage

        :param cache_dir: Base directory for caching embeddings
        :param model_name: Name of the embedding model
        """
        self.model_name = model_name
        self.base_cache_dir = os.path.normpath(cache_dir)
        self.cache_dir = os.path.join(self.base_cache_dir, model_name)
        os.makedirs(self.cache_dir, exist_ok=True)
        # 设备配置
        self.device = torch.device(f'cuda:{Constants.CUDA}' if torch.cuda.is_available() else 'cpu')
    def _get_cache_path(self, cid: str, q_index: int) -> str:
        """
        Generate cache file path with model-specific subdirectory

        :param cid: Unique content identifier
        :param q_index: Question index
        :return: Full path to cache file
        """
        filename = f"{cid}_{q_index}.pkl"

        return os.path.join(self.cache_dir, filename)

    def get_embeddings(self, cid: str, q_index: int, text: str) -> torch.Tensor:
        """
        Retrieve text embeddings with model-specific caching

        :param cid: Unique content identifier
        :param q_index: Question index
        :param text: Text to embed
        :return: Embedding tensor
        """
        # Check cache
        cache_path = self._get_cache_path(cid, q_index)

        # If cache exists, load from file
        if os.path.exists(cache_path):
            try:
                # 使用 torch.load 并明确指定设备
                with open(cache_path, 'rb') as f:
                    embedding = torch.load(f, map_location=self.device)
                return embedding
            except Exception as e:
                print(f"Cache read error: {e}")

        # Generate embedding if not cached
        embedding = self._embed_text(text).to(self.device)

        # Save to cache if embedding successful
        if embedding is not None:
            try:
                # 使用 torch.save 并保存到特定设备
                with open(cache_path, 'wb') as f:
                    torch.save(embedding.cpu(), f)
            except Exception as e:
                print(f"Cache write error: {e}")

        return embedding

    @abstractmethod
    def _embed_text(self, text: str) -> torch.Tensor:
        """
        Abstract method to be implemented by subclasses for text embedding

        :param text: Text to embed
        :return: Embedding tensor
        """
        pass


if __name__ == '__main__':
    import torch

    # 定义 .pkl 文件路径
    cache_path = r"D:\MyKT\model\textEmbedding\bertEmbeddingCache\bert-base-uncased\1_A.pkl"

    try:
        # 加载 .pkl 文件
        with open(cache_path, 'rb') as f:
            embedding = torch.load(f)

        # 检查 embedding 的类型和形状
        print(f"Type of embedding: {type(embedding)}")
        print(f"Shape of embedding: {embedding.shape}")
    except Exception as e:
        print(f"Error loading file: {e}")