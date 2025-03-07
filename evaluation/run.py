import os
import sys

############################################################ 必须在最上边
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
#############################################################
from model.EKTA_D.EKTA_D import EKTA_D
from evaluation.trainEKTA_D import trainEKTA_D
from evaluation.train_batch import train_batch
from model.EKTM_D.EKTM_D import EKTM_D
from constant import Constants
from log import setup_logger
import logging
from evaluation.train import train

if __name__ == "__main__":
    arg = {
        'embedding_model': Constants.EMBEDDING_MODEL,
        'rnn_model': "EKTM_D",
        'seq_model': "original"
    }
    # # 选择最佳 GPU
    Constants.CUDA = 0
    model = EKTM_D()
    setup_logger(model, Constants.LOG_PATH, arg)
    logging.info(f"运行在cuda:{Constants.CUDA}")
    train(model, arg)
