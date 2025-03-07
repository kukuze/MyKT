import logging

import torch
import torch.nn.functional as F

import constant
from data.pretreatment import split_user
from model.EKTM.EKTM import EKTM
from model.EKTM_D.EKTM_D import EKTM_D
from statistic.step_statistic import visualize_step_statistics
from util import *
import matplotlib.pyplot as plt
import numpy as np



def test_new_users(model):
    """
    Tests the model's ability to predict for new users (cold start).
    These users were not seen during training.
    """
    torch.set_grad_enabled(False)
    _, new_users_data = split_user()

    if not new_users_data:
        raise ValueError("New users data is empty.")
        # Initialize tracking dictionaries for per-step statistics
    step_counts = {}  # Number of users at each step
    step_correct = {}  # Number of correct predictions at each step
    # Initialize loss functions
    mae_loss = torch.nn.L1Loss()

    logging.info("Starting testing process for new users")
    total_bce, total_mae, total_acc, total_seq_cnt = 0, 0, 0, 0
    total_len=len(new_users_data)
    for user, test_seq in new_users_data.items():
        total_seq_cnt += 1

        h = None  # Initialize hidden state for new user
        bce, mae, acc, length = 0, 0, 0, len(test_seq)

        # Process each item in the test sequence
        for step_idx, item in enumerate(test_seq):
            res = torch.FloatTensor([1.0 if item.status == "OK" else 0.0]).to(model.device)

            # Predict using the model without prior user history
            pred, h = model(item, res, h)  # Pass None for previous score

            # Calculate losses
            step_loss = F.binary_cross_entropy_with_logits(pred, res)  # BCE with logits
            mae += mae_loss(torch.sigmoid(pred), res).item()  # MAE with sigmoid applied to pred
            bce += step_loss.item()

            step_counts[step_idx] = step_counts.get(step_idx, 0) + 1
            predicted = torch.sigmoid(pred) >= 0.5
            correct = (predicted == (res > 0.5))
            step_correct[step_idx] = step_correct.get(step_idx, 0) + correct.item()
            acc += correct.item()  # Accumulate correct samples count

        # Normalize metrics over the sequence length
        bce /= length
        mae /= length
        acc /= length

        total_bce += bce
        total_mae += mae
        total_acc += acc

        logging.info(f"Testing New User [{total_seq_cnt}/{total_len}]{user} seq_len:{length} ,  BCE={bce:.4f}, MAE={mae:.4f}, Accuracy={acc:.4f}")

    # Compute average metrics over all new users
    avg_bce = total_bce / total_seq_cnt
    avg_mae = total_mae / total_seq_cnt
    avg_acc = total_acc / total_seq_cnt
    logging.info(f"Testing Final Results for New Users: BCE={avg_bce:.4f}, MAE={avg_mae:.4f}, Accuracy={avg_acc:.4f}")
    torch.set_grad_enabled(True)
    visualize_step_statistics(step_counts,step_correct)
    return {"loss": avg_bce, "mae": avg_mae, "acc": avg_acc}


if __name__ == '__main__':
    arg = {
        'embedding_model': Constants.EMBEDDING_MODEL,
        'rnn_model': "EKTM_D",
        'seq_model': "original"
    }
    constant.Constants.CUDA=0
    model = EKTM_D()
    start_epoch = load_latest_model(model, Constants.MODEL_SNAPSHOT_PATH, arg)
    test_new_users(model)