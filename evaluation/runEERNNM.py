# 创建 EERNNM 模型实例
import os
import sys

############################################################ 必须在最上边
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
#############################################################
from util import *
from constant import Constants
from log import setup_logger
import logging
from model.EERNNM.EERNNM import EERNNM
from evaluation.train import train

if __name__ == "__main__":
    arg = {
        'embedding_model': Constants.EMBEDDING_MODEL,
        'rnn_model': "EERNNM",
        'seq_model': "original"
    }
    # # 选择最佳 GPU
    Constants.CUDA = 0
    model = EERNNM()
    setup_logger(model, Constants.LOG_PATH, arg, additional_info='experiment1')
    logging.info(f"运行在cuda:{Constants.CUDA}")
    train(model, arg)
