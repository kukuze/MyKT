from tqdm import tqdm
import shutil
from pathlib import Path

from statistic.statistic_user_submission_info import calculate_user_statistics


def filter_users_by_success_rate(user_statistics,
                                 min_success_rate=0.2,
                                 max_success_rate=0.7,
                                 min_submissions=100,
                                 max_submissions=800):
    """
    根据成功率和提交次数范围过滤用户

    :param user_statistics: 用户统计信息字典
    :param min_success_rate: 最小成功率阈值
    :param max_success_rate: 最大成功率阈值
    :param min_submissions: 最小提交次数
    :param max_submissions: 最大提交次数
    :return: 过滤后的用户统计信息
    """
    filtered_users = {}

    for user, data in user_statistics.items():
        # 计算成功率
        total_submissions = data["total"]
        success_submissions = data["success"]

        # 检查条件：
        # 1. 总提交次数在指定范围内
        # 2. 成功率在指定范围内
        if (min_submissions <= total_submissions <= max_submissions and
                min_success_rate <= (success_submissions / total_submissions) <= max_success_rate):
            filtered_users[user] = data

    # 打印过滤信息
    print(f"原始用户数: {len(user_statistics)}")
    print(f"清洗后用户数: {len(filtered_users)}")
    print(f"成功率区间: [{min_success_rate}, {max_success_rate}]")
    print(f"提交次数区间: [{min_submissions}, {max_submissions}]")

    return filtered_users


def clean_user_submissions(input_dir, output_dir, filtered_users):
    """
    根据过滤后的用户列表清理提交数据，并显示进度条

    :param input_dir: 输入目录
    :param output_dir: 输出目录
    :param filtered_users: 过滤后的用户字典
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 获取所有需要处理的用户文件夹
    user_folders = [f for f in input_path.iterdir() if f.name in filtered_users]

    for user_folder in tqdm(user_folders, desc="处理用户文件夹", unit="folder"):
        # 创建对应的输出用户文件夹
        user_output_folder = output_path / user_folder.name
        user_output_folder.mkdir(exist_ok=True)

        # 获取提交文件（只有一个）
        submission_file = next(user_folder.glob("submissions_*.json"), None)

        if submission_file:
            try:
                shutil.copy(
                    submission_file,
                    user_output_folder / submission_file.name
                )
            except Exception as e:
                print(f"复制文件 {submission_file} 时出错: {e}")
        else:
            print(f"用户 {user_folder.name} 没有找到提交文件")


def main():
    input_dir = r'C:\Users\CharmingZe\Desktop\毕设\codeforces爬取数据\problem_info\user_submissions'
    output_dir = r'C:\Users\CharmingZe\Desktop\毕设\codeforces爬取数据\problem_info\cleaned_user_submissions'

    # 计算用户统计信息
    print("计算用户统计信息...")
    user_statistics = calculate_user_statistics(input_dir)

    # 过滤用户
    print("过滤用户...")
    filtered_users = filter_users_by_success_rate(
        user_statistics,
        min_success_rate=0.2,  # 最小成功率
        max_success_rate=0.7,  # 最大成功率
        min_submissions=100,  # 最小提交次数
        max_submissions=800,  # 最大提交次数
    )

    # 清理并复制数据
    print("开始清理并复制数据...")
    clean_user_submissions(input_dir, output_dir, filtered_users)
    print("数据清理完成！")


if __name__ == "__main__":
    main()
