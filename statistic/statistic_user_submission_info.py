import os
import json
import matplotlib
import numpy as np
from scipy import stats
from tqdm import tqdm
from pathlib import Path
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def calculate_user_statistics(base_directory):
    user_statistics = {}
    base_path = Path(base_directory)
    total_files = sum(1 for _ in base_path.rglob('*.json'))

    with tqdm(total=total_files, desc="Processing files") as pbar:
        for user_folder in base_path.iterdir():
            if user_folder.is_dir():
                for file_path in user_folder.glob("submissions_*.json"):
                    try:
                        with file_path.open('r') as f:
                            submissions = json.load(f)
                            user_name = user_folder.name
                            if user_name not in user_statistics:
                                user_statistics[user_name] = {"total": 0, "success": 0}
                            for submission in submissions:
                                user_statistics[user_name]["total"] += 1
                                if submission["verdict"] == "OK":
                                    user_statistics[user_name]["success"] += 1
                    except json.JSONDecodeError as e:
                        print(f"JSON解析错误，文件: {file_path}, 错误: {e}")
                    except Exception as e:
                        print(f"处理文件 {file_path} 时发生一般错误: {e}")
                    pbar.update(1)  # 更新进度条

    return user_statistics
def plot_statistics(user_statistics):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    users = list(user_statistics.keys())
    total_submissions = [user_statistics[user]["total"] for user in users]
    success_rate = [
        user_statistics[user]["success"] / user_statistics[user]["total"]
        if user_statistics[user]["total"] > 0 else 0 for user in users
    ]

    plt.figure()
    # 计算95%分位数作为x轴上限
    x_limit = np.percentile(total_submissions, 95)
    plt.hist(total_submissions, bins=range(0, int(x_limit) + 1, 10), color='tab:blue', alpha=0.7)
    plt.xlabel('提交次数区间')
    plt.ylabel('用户数量')
    plt.title('用户提交次数分布 (95%分位数以内)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cleaned_user_submission_distribution.png", dpi=300)
    plt.close()

    # 创建成功率分布图
    plt.figure(figsize=(12, 6))

    # 方法1：使用非常高的bins数量的直方图
    plt.hist(success_rate, bins=200, density=True, color='tab:orange', alpha=0.7)

    # 方法2：添加核密度估计曲线
    kernel = stats.gaussian_kde(success_rate)
    x_range = np.linspace(0, 1, 1000)
    plt.plot(x_range, kernel(x_range), 'r-', lw=2)

    plt.xlabel('成功率')
    plt.ylabel('密度')
    plt.title('用户成功率分布')
    plt.xticks([i/10 for i in range(11)])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cleaned_user_success_rate_distribution.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    base_directory = r'D:\MyKT\problem_info\cleaned_user_submissions'
    user_statistics = calculate_user_statistics(base_directory)
    plot_statistics(user_statistics)