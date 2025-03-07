import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'，取决于你的环境
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 数据
tag_counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]  # 横坐标：标签数量
problem_counts = [1666, 2777, 2364, 1395, 662, 315, 115, 34, 10, 2]  # 纵坐标：题目数量

# 绘制图形
plt.figure(figsize=(10, 6))
plt.bar(tag_counts, problem_counts, color='b', edgecolor='black')

# 设置标题和标签
plt.title("题目标签数量分布图")
plt.xlabel("类型数量")
plt.ylabel("题目数量")

# 设置横坐标刻度
plt.xticks(tag_counts)  # 显示所有标签数量

# 添加数据标签
for i, count in enumerate(problem_counts):
    plt.text(tag_counts[i], count + 50, str(count), ha='center', fontsize=10)

# 显示网格
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 展示图形
plt.tight_layout()  # 自动调整布局
# plt.show()
# 保存图片到文件
plt.savefig("题目标签分布统计.png", dpi=300, bbox_inches='tight')  # 保存为高分辨率PNG图片