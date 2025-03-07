# 创建 EERNNM 模型实例
import logging
import os
import sys

############################################################ 必须在最上边
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
#############################################################
from constant import Constants
from log import setup_logger
from model.EERNNA.EERNNA import EERNNA
from evaluation.train import train

# # 选择最佳 GPU
if __name__ == "__main__":
    arg = {
        'embedding_model': Constants.EMBEDDING_MODEL,
        'rnn_model': "EERNNA",
        'seq_model': "original"
    }
    Constants.CUDA = 0
    model = EERNNA()
    setup_logger(model, Constants.LOG_PATH, arg, additional_info='experiment1')
    logging.info(f"运行在cuda:{Constants.CUDA}")
    train(model, arg)
