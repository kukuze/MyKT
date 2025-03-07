# 创建 EERNNM 模型实例
import os
import sys

############################################################ 必须在最上边
from model.EKTM.EKTM import EKTM

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
#############################################################
from constant import Constants
from log import setup_logger
import logging
from evaluation.train import train

if __name__ == "__main__":
    arg = {
        'embedding_model': "RNNText",
        'rnn_model': "EKTM",
        'seq_model': "original"
    }
    # # 选择最佳 GPU
    Constants.CUDA = 0
    model = EKTM()
    setup_logger(model, Constants.LOG_PATH, arg, additional_info='experiment1')
    logging.info(f"运行在cuda:{Constants.CUDA}")
    train(model, arg)
