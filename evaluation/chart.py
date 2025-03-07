import matplotlib
matplotlib.use('Agg')
import os
import re
import matplotlib.pyplot as plt
import numpy as np

# 设置文件夹路径
folder_path = 'D:\linuxDown\snapshots'

# 模型文件夹名称
model_folders = ['EERNNA', 'EERNNM', 'EKTA', 'EKTM']

# 指标
metrics = ['train_loss', 'train_mae', 'train_acc', 'test_loss', 'test_mae', 'test_acc']

# 用于保存每个模型的数据
model_data = {model: {metric: [] for metric in metrics} for model in model_folders}


# 读取数据并提取指标
def extract_metrics_from_filename(filename):
    """
    从文件名中提取出epoch和各项指标
    """
    # 正则表达式提取指标和对应的值
    pattern = r"model_train_epoch_(\d+)_train_loss_(\d+\.\d+)_train_mae_(\d+\.\d+)_train_acc_(\d+\.\d+)_test_loss_(\d+\.\d+)_test_mae_(\d+\.\d+)_test_acc_(\d+\.\d+).pth"
    match = re.match(pattern, filename)
    if match:
        epoch = int(match.group(1))
        train_loss = float(match.group(2))
        train_mae = float(match.group(3))
        train_acc = float(match.group(4))
        test_loss = float(match.group(5))
        test_mae = float(match.group(6))
        test_acc = float(match.group(7))

        return epoch, train_loss, train_mae, train_acc, test_loss, test_mae, test_acc
    return None


# 遍历所有文件夹和文件，提取数据
for model in model_folders:
    model_folder = os.path.join(folder_path, model)
    if os.path.exists(model_folder):
        for filename in os.listdir(model_folder):
            if filename.endswith('.pth'):
                result = extract_metrics_from_filename(filename)
                if result:
                    epoch, train_loss, train_mae, train_acc, test_loss, test_mae, test_acc = result
                    model_data[model]['train_loss'].append((epoch, train_loss))
                    model_data[model]['train_mae'].append((epoch, train_mae))
                    model_data[model]['train_acc'].append((epoch, train_acc))
                    model_data[model]['test_loss'].append((epoch, test_loss))
                    model_data[model]['test_mae'].append((epoch, test_mae))
                    model_data[model]['test_acc'].append((epoch, test_acc))


# 绘图
def plot_metric(metric_name):
    plt.figure(figsize=(10, 6))

    for model in model_folders:
        # 提取每个模型的训练和测试数据
        train_data = model_data[model][f'train_{metric_name}']
        test_data = model_data[model][f'test_{metric_name}']

        # 提取epoch和metric值
        train_epochs, train_values = zip(*train_data)
        test_epochs, test_values = zip(*test_data)

        # 绘制训练数据的散点
        plt.scatter(train_epochs, train_values, label=f'{model} Train {metric_name}', marker='o')

        # 绘制测试数据的散点
        plt.scatter(test_epochs, test_values, label=f'{model} Test {metric_name}', marker='x')

    # 设置图表标题和标签
    plt.title(f'Model Comparison: {metric_name.capitalize()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.capitalize())
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{metric_name}_comparison.png')  # 保存图像到文件
    plt.close()


# 绘制每个指标的图表
for metric in ['loss', 'mae', 'acc']:
    plot_metric(metric)
