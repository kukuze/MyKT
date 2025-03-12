import random
import time

import torch.nn.functional as F

from evaluation.test_new_user import test_new_users
from util import *
from data.pretreatment import split_user
from constant import Constants


def train(model, arg):
    logging.info('model: %s' % (type(model).__name__))
    logging.info('loading dataset')
    # 获取训练集和测试集
    data, _ = split_user()
    users = list(data)
    optimizer = torch.optim.Adam(model.parameters())
    start_epoch = load_latest_model(model, Constants.MODEL_SNAPSHOT_PATH, arg)
    for epoch in range(start_epoch, Constants.EPOCH):
        logging.info(('Starting epoch {}:'.format(epoch + 1)))
        then = time.time()

        total_loss = 0
        total_mae = 0
        total_acc = 0
        total_seq_cnt = 0
        random.shuffle(users)
        seq_cnt = len(users)

        MAE = torch.nn.L1Loss()

        for user in users:
            total_seq_cnt += 1
            seq = data[user]
            length = len(seq)
            optimizer.zero_grad()
            loss = 0
            mae = 0
            acc = 0
            h = None
            for i, item in enumerate(seq):
                res = torch.FloatTensor([1.0 if item.status == "OK" else 0.0]).to(model.device)
                pred, h = model(item, res, h)  # 自己要记录的什么不同模型自己取
                loss += F.binary_cross_entropy_with_logits(pred, res)
                m = MAE(torch.sigmoid(pred), res).item()
                mae += m
                acc += m < 0.5

            loss /= length
            mae /= length
            acc /= length
            if torch.isnan(loss) or torch.isnan(torch.tensor(mae)) or torch.isnan(torch.tensor(acc)):
                logger.warning(f'Skipping user {user} due to NaN in loss/MAE/accuracy')
                continue
            total_loss += loss.item()
            total_mae += mae
            total_acc += acc

            loss.backward()
            optimizer.step()
            now = time.time()
            duration = (now - then) / 60
            then = now
            train_epoch_info = {
                'epoch': epoch + 1,
                'total_seq_cnt': total_seq_cnt,
                'seq_cnt': seq_cnt,
                'loss': total_loss / total_seq_cnt,
                'mae': total_mae / total_seq_cnt,
                'acc': total_acc / total_seq_cnt,
                'seqs_per_min': ((total_seq_cnt - 1) % 10 + 1) / duration
            }
            # logging.info the epoch's information

            logging.info('[%d:%d/%d] (%.2f seqs/min) '
                         'loss %.6f, mae %.6f, acc %.6f '
                         'user: %s, seq_length: %d' %
                         (epoch + 1, total_seq_cnt, seq_cnt,
                          train_epoch_info['seqs_per_min'],
                          train_epoch_info['loss'],
                          train_epoch_info['mae'],
                          train_epoch_info['acc'],
                          user,  # 添加用户名
                          length))  # 添加序列长度
        test_epoch_info = test_new_users(model)
        # Save the snapshot with the current epoch's information
        save_model(model, Constants.MODEL_SNAPSHOT_PATH, train_epoch_info, test_epoch_info,arg)
