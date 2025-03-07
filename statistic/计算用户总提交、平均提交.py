import os
import json
from tqdm import tqdm  # 导入 tqdm

if __name__ == '__main__':
    # 定义文件夹路径
    base_folder = r"C:\Users\CharmingZe\Desktop\毕设\codeforces爬取数据\problem_info\cleaned_user_submissions"

    # 初始化变量
    total_submissions = 0  # 总提交数量
    user_count = 0  # 用户数量

    # 获取所有用户文件夹
    user_folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]

    # 使用 tqdm 添加进度条
    for user_folder in tqdm(user_folders, desc="处理用户", unit="用户"):
        user_folder_path = os.path.join(base_folder, user_folder)

        # 构造 submissions 文件路径
        submissions_file = os.path.join(user_folder_path, f"submissions_{user_folder}.json")

        # 检查文件是否存在
        if os.path.exists(submissions_file):
            # 读取 JSON 文件
            with open(submissions_file, 'r', encoding='utf-8') as f:
                try:
                    submissions_data = json.load(f)

                    # 假设 JSON 数据是一个数组，计算其长度
                    submission_count = len(submissions_data)

                    # 更新统计数据
                    total_submissions += submission_count
                    user_count += 1

                except json.JSONDecodeError:
                    print(f"文件 {submissions_file} 格式错误，无法解析为 JSON。")
        else:
            print(f"文件 {submissions_file} 不存在。")

    # 计算平均提交数量
    if user_count > 0:
        avg_submissions = total_submissions / user_count
    else:
        avg_submissions = 0

    # 输出统计结果
    print("\n统计结果：")
    print(f"总提交数量: {total_submissions}")
    print(f"用户数量: {user_count}")
    print(f"平均提交数量: {avg_submissions:.2f}")