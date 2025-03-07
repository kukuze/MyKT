import random
import time

import torch.nn.functional as F

from evaluation.test_new_user import test_new_users
from util import *
from data.pretreatment import split_user
from constant import Constants

logger = logging.getLogger()


def train_batch(model,arg):
    logger.info('model: %s' % (type(model).__name__))
    logger.info('loading dataset')

    # 获取训练集和测试集
    data, _ = split_user()
    optimizer = torch.optim.Adam(model.parameters())

    start_epoch = load_latest_model(model, Constants.MODEL_SNAPSHOT_PATH,arg)

    batch_size = 5  # 每 200 个记录反向传播一次 过长的梯度累计 可能效果不好+占用内存

    for epoch in range(start_epoch, Constants.EPOCH):
        logger.info(('Training epoch {}:'.format(epoch + 1)))
        then = time.time()

        total_loss = 0
        total_mae = 0
        total_acc = 0
        total_seq_cnt = 0

        users = list(data)
        random.shuffle(users)
        seq_cnt = len(users)

        MAE = torch.nn.L1Loss()

        for user in users:
            total_seq_cnt += 1
            seq = data[user]
            length = len(seq)
            optimizer.zero_grad()
            user_loss = 0
            step_loss = 0  # 当前时间步的损失
            batch_loss = 0  # 本批次的累积的损失
            user_mae = 0
            user_acc = 0
            h = None
            model.init_history()
            for i, item in enumerate(seq):
                res = torch.FloatTensor([1.0 if item.status == "OK" else 0.0]).to(model.device)
                pred, h = model(item, res, h)  # 自己要记录的什么不同模型自己取
                step_loss = F.binary_cross_entropy_with_logits(pred, res)
                # 累积当前记录的损失
                batch_loss += step_loss
                user_loss += step_loss
                # 计算预测的概率值
                prob = torch.sigmoid(pred)  # 结果范围为 [0, 1]
                # 计算 MAE（绝对误差）
                user_mae += MAE(prob, res).item()
                # 计算 Accuracy（是否预测正确）
                predicted = prob >= 0.5  # 概率大于等于 0.5 则预测为 1
                correct = (predicted == (res == 1.0))  # 比较预测值和真实标签
                user_acc += correct.item()  # 累积正确样本数

                # 每 batch_size 或用户序列结束时反向传播
                if (i + 1) % batch_size == 0 or i == length - 1:
                    batch_loss.backward()  # 反向传播
                    optimizer.step()  # 参数更新
                    optimizer.zero_grad()  # 清空梯度
                    batch_loss = 0  # 重置累积损失
            # 归一化用户的统计数据
            user_mae /= length
            user_acc /= length
            user_loss /= length
            if torch.isnan(user_loss) or torch.isnan(torch.tensor(user_mae)) or torch.isnan(torch.tensor(user_acc)):
                logger.warning(f'Skipping user {user} due to NaN in loss/MAE/accuracy')
                continue

            # 累积到总损失
            total_loss += user_loss
            total_mae += user_mae
            total_acc += user_acc

            # 输出当前用户的日志信息
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
        # 每个epoch后做一个测试
        test_epoch_info = test_new_users(model)
        # Save the snapshot with the current epoch's information
        save_model(model, Constants.MODEL_SNAPSHOT_PATH, train_epoch_info, test_epoch_info)
