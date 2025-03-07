import logging
import os
import re

import numpy as np
import pandas as pd
import pynvml
import torch

from constant import Constants
from spider.UserSubmissionSpider import sanitize_filename

logger = logging.getLogger()


def save_model(model, save_dir, train_epoch_info, test_epoch_info, arg):
    """
    保存模型，包含参数配置信息
    """
    save_dir = os.path.normpath(save_dir)
    config_str = get_config_string(arg)
    model_folder = os.path.join(save_dir, type(model).__name__, config_str)

    # 创建文件夹
    os.makedirs(model_folder, exist_ok=True)

    file_name = (
        f"model_train_epoch_{train_epoch_info['epoch']}_"
        f"train_loss_{train_epoch_info['loss']:.4f}_"
        f"train_mae_{train_epoch_info['mae']:.4f}_"
        f"train_acc_{train_epoch_info['acc']:.4f}_"
        f"test_loss_{test_epoch_info['loss']:.4f}_"
        f"test_mae_{test_epoch_info['mae']:.4f}_"
        f"test_acc_{test_epoch_info['acc']:.4f}.pth"
    )
    file_path = os.path.join(model_folder, file_name)
    # 保存前先将模型状态转移到CPU
    state_dict = model.state_dict()
    cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
    torch.save(cpu_state_dict, file_path)

    logger.info(f"模型已保存到: {file_path}")


def load_latest_model(model, save_dir, arg):
    """
    加载模型，考虑参数配置信息
    """
    save_dir = os.path.normpath(save_dir)
    config_str = get_config_string(arg)
    model_folder = os.path.join(save_dir, type(model).__name__, config_str)
    device = torch.device(f'cuda:{Constants.CUDA}' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(model_folder):
        logger.info(f"模型文件夹 {model_folder} 不存在。")
        return 0

    epoch_pattern = re.compile(r'model_.*_epoch_(\d+)_.*\.pth')
    model_files = [
        f for f in os.listdir(model_folder)
        if os.path.isfile(os.path.join(model_folder, f)) and epoch_pattern.match(f)
    ]

    if not model_files:
        logger.info(f"没有找到模型文件在 {model_folder} 中。")
        return 0

    model_files.sort(key=lambda x: int(epoch_pattern.search(x).group(1)), reverse=True)
    latest_model_path = os.path.join(model_folder, model_files[0])

    logger.info(f"加载最新模型: {latest_model_path}")
    latest_epoch = int(epoch_pattern.search(model_files[0]).group(1))
    # 加载时指定目标设备
    model_state = torch.load(latest_model_path, map_location=device)
    model.load_state_dict(model_state)
    return latest_epoch
def get_config_string(arg):
    """
    将配置参数转换为字符串，用于文件夹命名
    """
    config_str = []
    for key, value in sorted(arg.items()):
        config_str.append(f"{key}-{value}")
    return "_".join(config_str)
def load_embedding(filename):
    f = open(filename, encoding='utf-8')
    wcnt, emb_size = next(f).strip().split(' ')
    wcnt = int(wcnt)
    emb_size = int(emb_size)

    words = []
    embs = []
    for line in f:
        fields = line.strip().split(' ')
        word = fields[0]
        emb = np.array([float(x) for x in fields[1:]])
        words.append(word)
        embs.append(emb)

    embs = np.asarray(embs)
    return wcnt, emb_size, words, embs


def load_glove_embedding(filename):
    words = []
    embs = []

    # 读取第一行来确定向量维度
    with open(filename, encoding='utf-8') as f:
        first_line = f.readline().strip().split(' ')
        emb_size = len(first_line) - 1  # 减去词本身，得到向量维度

    # 重新读取文件
    with open(filename, encoding='utf-8') as f:
        for line in f:
            fields = line.strip().split(' ')
            word = fields[0]
            emb = np.array([float(x) for x in fields[1:]])
            words.append(word)
            embs.append(emb)

    # 转换为numpy数组并获取词表大小
    embs = np.asarray(embs)
    wcnt = len(words)

    return wcnt, emb_size, words, embs
import shutil
import os

def delete_folder(folder_path):
    try:
        # 检查路径是否存在且是一个文件夹
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            # 删除文件夹及其所有内容
            shutil.rmtree(folder_path)
            print(f"文件夹 '{folder_path}' 已成功删除。")
        else:
            print(f"'{folder_path}' 不是一个有效的文件夹路径。")
    except Exception as e:
        print(f"删除文件夹时发生错误: {e}")

def extract_file_paths(error_file_path):
    file_paths = []
    with open(error_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(' : ')
            if len(parts) > 1:
                file_paths.append(parts[1])
    return file_paths

def delete_files(base_directory, file_paths):
    for relative_path in file_paths:
        full_path = os.path.join(base_directory, relative_path)
        try:
            if os.path.exists(full_path):
                os.remove(full_path)
                print(f"已删除文件: {full_path}")
            else:
                print(f"文件不存在: {full_path}")
        except Exception as e:
            print(f"删除文件 {full_path} 时发生错误: {e}")

def get_folder_names(directory):
    return set(folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder)))

def extract_users_from_ratings(ratings_directory):
    users = set()
    for file in os.listdir(ratings_directory):
        if file.endswith(".xlsx"):
            file_path = os.path.join(ratings_directory, file)
            df = pd.read_excel(file_path)
            users.update(df['User Name'].dropna())
    return users

def reverse_sanitize_filename(filename):
    if filename.endswith('_dot'):
        return filename[:-4] + '.'
    return filename


import torch
import os
import glob


def convert_model_to_cpu(model_path):
    """
    将模型从GPU转换到CPU并重新保存
    """
    try:
        # 加载模型状态
        state_dict = torch.load(model_path, map_location='cpu')

        # 确保所有张量都在CPU上
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}

        # 创建备份
        backup_path = model_path + '.backup'
        if not os.path.exists(backup_path):
            os.rename(model_path, backup_path)

        # 保存CPU版本
        torch.save(cpu_state_dict, model_path)

        print(f"成功转换模型: {model_path}")
        return True

    except Exception as e:
        print(f"转换模型时出错 {model_path}: {str(e)}")
        return False


def batch_convert_models(base_dir):
    """
    批量转换目录下的所有模型文件
    """
    # 查找所有.pth文件
    model_files = glob.glob(os.path.join(base_dir, "**/*.pth"), recursive=True)

    success_count = 0
    fail_count = 0

    for model_path in model_files:
        print(f"处理模型: {model_path}")
        if convert_model_to_cpu(model_path):
            success_count += 1
        else:
            fail_count += 1

    print(f"\n转换完成:")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"总计: {len(model_files)}")


def verify_model_device(model_path):
    """
    验证模型中的所有张量是否都在CPU上
    """
    state_dict = torch.load(model_path)
    all_on_cpu = True

    for key, tensor in state_dict.items():
        if tensor.device.type != 'cpu':
            print(f"警告: {key} 在 {tensor.device}")
            all_on_cpu = False

    return all_on_cpu

def check_memory(description=""):
    import torch
    import gc

    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()

    # Get memory stats
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # Convert to MB

    print(f"\n=== Memory Status {description} ===")
    print(f"Allocated: {allocated:.2f} MB")
    print(f"Reserved:  {reserved:.2f} MB")


# import pandas as pd
#
# # Load the first Excel file
# df1 = pd.read_excel(r'D:\MyKT\data\new.xlsx')
#
# # Load the second Excel file
# df2 = pd.read_excel(r'D:\MyKT\data\codeforces_problem_detail.xlsx')
#
# # Merge the two dataframes based on 'cid' and 'qindex'
# merged_df = pd.merge(df1, df2, on=['cid', 'qindex'], how='left')
#
# # Add the 'content' column from df2 to df1
# df1['content'] = merged_df['content']
#
# # Save the updated dataframe back to the first Excel file
# df1.to_excel(r'D:\MyKT\data\new.xlsx', index=False)


