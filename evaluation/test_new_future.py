import logging
import random
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from data.pretreatment import split_future
from util import *

logger = logging.getLogger()


def test_new_future(model):
    """
    Refactored function to test the future by:
    - Splitting the dataset into training and testing.
    - Using the training data to update the student state.
    - Iteratively updating the student state using real scores during testing.
    """
    # Split the dataset
    torch.set_grad_enabled(False)
    train_data, test_data = split_future()

    if not train_data or not test_data:
        raise ValueError("Train or test data is empty after splitting.")

    # Initialize variables
    mse_loss = torch.nn.MSELoss()
    mae_loss = torch.nn.L1Loss()

    logger.info("Starting testing process")
    total_mse, total_mae, total_acc, total_seq_cnt = 0, 0, 0, 0

    for user, test_seq in test_data.items():
        train_seq = train_data.get(user, [])
        combined_seq = train_seq + test_seq
        total_seq_cnt += 1

        h = None  # Student state
        mse, mae, acc, length = 0, 0, 0, len(test_seq)

        # Process each item in the combined sequence
        for i, item in enumerate(combined_seq):
            res = torch.FloatTensor([1.0 if item.status == "OK" else 0.0]).to(model.device)
            #item_time = torch.FloatTensor([item.time]).to(model.device)

            if i < len(train_seq):
                # Update student state using training data
                _, h = model(item, res, h)
            else:
                # Predict using test data
                pred, h = model(item, res, h)
                pred = torch.sigmoid(pred)

                # Calculate losses
                mse += mse_loss(pred, res).item()
                mae += mae_loss(pred, res).item()
                # 计算 Accuracy（是否预测正确）
                predicted = pred >= 0.5  # 概率大于等于 0.5 则预测为 1
                correct = (predicted == (res == 1.0))  # 比较预测值和真实标签
                acc += correct.item()  # 累积正确样本数

        # Normalize metrics over test sequence length
        mse /= length
        mae /= length
        acc /= length

        total_mse += mse
        total_mae += mae
        total_acc += acc

        logger.info(f"Testing User {user}: MSE={mse:.4f}, MAE={mae:.4f}, Accuracy={acc:.4f}")

    # Aggregate results over all users
    avg_mse = total_mse / total_seq_cnt
    avg_mae = total_mae / total_seq_cnt
    avg_acc = total_acc / total_seq_cnt

    logger.info(f"Testing Final Results: MSE={avg_mse:.4f}, MAE={avg_mae:.4f}, Accuracy={avg_acc:.4f}")
