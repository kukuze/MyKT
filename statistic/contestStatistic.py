import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate():
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # AtCoder 的比赛类型数据
    atcoder_data = {
        'abc': 345,
        'arc': 132,
        'agc': 69
    }
    # Codeforces 的比赛类型数据
    codeforces_data = {
        'ICPC(vp)': 1844,
        'CF': 1434,
        'ICPC': 476,
        'IOI(vp)': 288,
        'IOI': 41
    }
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    # 设置位置（x轴）和宽度
    bar_width = 0.35
    # AtCoder 的柱状图位置
    atcoder_positions = range(len(atcoder_data))
    # Codeforces 的柱状图位置，需要确保与 AtCoder 的位置相对偏移
    codeforces_positions = [x + bar_width for x in range(len(atcoder_data))]
    # 绘制 AtCoder 的柱状图
    ax.bar(atcoder_positions, atcoder_data.values(), bar_width, label='AtCoder', color='blue')
    # 绘制 Codeforces 的柱状图
    ax.bar(codeforces_positions, codeforces_data.values(), bar_width, label='Codeforces', color='orange')
    # 设置 x 轴标签和标题
    ax.set_xlabel('比赛类型', fontsize=12)
    ax.set_ylabel('比赛数量', fontsize=12)
    ax.set_title('AtCoder 和 Codeforces 比赛类型分布', fontsize=14)
    # 设置 x 轴的 ticks 和标签
    # 注意：两个平台的比赛类型合并后的标签长度为 8，所以需要调整
    ax.set_xticks([x + bar_width / 2 for x in range(max(len(atcoder_data), len(codeforces_data)))])
    # 合并两者的比赛类型作为 x 轴标签
    ax.set_xticklabels(list(atcoder_data.keys()) + list(codeforces_data.keys()))
    # 添加图例
    ax.legend()
    # 显示图形
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# generate()
