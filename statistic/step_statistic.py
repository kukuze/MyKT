import matplotlib
matplotlib.use('Agg')  # 在导入 pyplot 之前设置后端
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from datetime import datetime
def visualize_step_statistics(step_counts, step_correct):
    # 准备数据
    steps = sorted(step_counts.keys())[:120]
    accuracies = [step_correct[step] / step_counts[step] for step in steps]
    total_predictions = [step_counts[step] for step in steps]

    # 创建图形
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制柱状图（预测总数）
    ax1.bar(steps, total_predictions, alpha=0.3, color='blue', label='预测总数')

    # 创建第二个y轴
    ax2 = ax1.twinx()

    # 绘制折线图（准确率）
    ax2.plot(steps, accuracies, color='red', marker='o', label='准确率')

    # 设置标签和标题
    ax1.set_xlabel('步骤')
    ax1.set_ylabel('预测数量')
    ax2.set_ylabel('准确率')
    plt.title('各步骤预测数量和准确率对比')

    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 构建文件名
    filename = f'step_statistics_{current_time}.png'

    # 保存图形
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    import pickle

    # 方法1：使用 pickle.load()
    with open('D:\MyKT\statistic\step_statistics_20241224_232924.pkl', 'rb') as f:
        data = pickle.load(f)

    step_counts = data['step_counts']
    step_correct = data['step_correct']
    # 调用可视化函数
    visualize_step_statistics(step_counts, step_correct)
