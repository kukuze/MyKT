import shutil
import concurrent.futures
import pandas as pd
import requests
import json
import os

# 读取 codeforces_ratings 目录中的所有 Excel 文件，提取用户名字
from statistic.statistic_user_submission_info import calculate_user_statistics, plot_statistics
def sanitize_filename(handle):
    # 如果用户名以点号结尾，将最后一个点替换为 '_dot'
    if handle.endswith('.'):
        return handle[:-1] + '_dot'
    return handle

def extract_users_from_ratings(ratings_directory):
    users = set()  # 使用集合去重
    for file in os.listdir(ratings_directory):
        if file.endswith(".xlsx"):  # 确保是xlsx文件
            file_path = os.path.join(ratings_directory, file)
            df = pd.read_excel(file_path)  # 读取excel文件
            # 假设 'User Name' 是用户名字列
            users.update(df['User Name'].dropna())  # 提取用户名字，去除空值
            print(f"{users.__len__()}个目标用户了")
    return users


# 检查用户提交数据是否已经存在
def is_submission_exist(handle):
    user_submissions_dir = r"C:\Users\CharmingZe\Desktop\毕设\codeforces爬取数据\problem_info\user_submissions"
    sanitized_handle = sanitize_filename(handle)
    user_path = os.path.join(user_submissions_dir, sanitized_handle)
    return os.path.exists(user_path) and os.path.exists(os.path.join(user_path, f"submissions_{sanitized_handle}.json"))


def get_user_submissions(handle, count=100000):
    sanitized_handle = sanitize_filename(handle)
    directory = os.path.join(r"C:\Users\CharmingZe\Desktop\毕设\codeforces爬取数据\problem_info\user_submissions", sanitized_handle)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, f"submissions_{sanitized_handle}.json")

    # 检查用户的提交信息文件是否已经存在
    if os.path.exists(filename):
        print(f"User {sanitized_handle} already has submissions data. Skipping...")
        return  # 如果数据已存在，跳过请求

    # 如果缓存文件不存在，进行网络请求
    url = f"https://codeforces.com/api/user.status?handle={handle}&count={count}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'OK':
            submissions = data['result']
            try:
                # 保存结果到缓存文件
                with open(filename, 'w') as file:
                    json.dump(submissions, file)
                print(f"Saved submissions for user {sanitized_handle}")
            except Exception as e:
                print(f"Error saving submissions for user {sanitized_handle}: {e}")
        else:
            print(f"Error fetching submissions for user {sanitized_handle}: {data.get('comment', 'Unknown error')}")
    else:
        print(f"Failed to fetch submissions for {sanitized_handle}, status code: {response.status_code}")

    # 使用示例：删除空文件夹


##base_directory = r'D:\MyKT\problem_info\user_submissions'
##remove_empty_user_folders(base_directory)
def remove_empty_user_folders(base_directory):
    # 遍历 base_directory 下的所有文件夹
    for user_folder in os.listdir(base_directory):
        user_path = os.path.join(base_directory, user_folder)

        # 检查是否为文件夹，且是否为空
        if os.path.isdir(user_path) and not os.listdir(user_path):
            print(f"正在删除空文件夹: {user_path}")
            # 删除空文件夹
            shutil.rmtree(user_path)


def main():
    ratings_directory = r"C:\Users\CharmingZe\Desktop\毕设\codeforces爬取数据\problem_info\codeforces_ratings"  # 指定你的ratings目录
    # 每次循环开始时重新提取用户和未处理用户
    all_users = extract_users_from_ratings(ratings_directory)  # 提取所有用户
    print(f"Total users extracted from ratings: {len(all_users)}")
    while True:
        unprocessed_users = []
        # 过滤掉已经在 user_submissions 目录中的用户
        for user in all_users:
            if not is_submission_exist(user):
                unprocessed_users.append(user)
                print(f"未处理用户数量{unprocessed_users.__len__()} ")

        # 如果没有未处理的用户，则退出循环
        if not unprocessed_users:
            print("All users' submissions have been processed.")
            break

        print(f"Total users to process: {len(unprocessed_users)}")

        # 使用线程池处理未处理的用户
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_handle = {executor.submit(get_user_submissions, handle): handle for handle in unprocessed_users}

            for future in concurrent.futures.as_completed(future_to_handle):
                handle = future_to_handle[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"用户 {handle} 的提交获取时发生异常: {exc}")
                finally:
                    # 手动释放内存
                    del future  # 删除已完成的任务对象
if __name__ == "__main__":
    main()

