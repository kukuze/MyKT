
from bs4 import BeautifulSoup
import cloudscraper
import re
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def statistic():
    # 设置中文字体（SimHei 是黑体，可替换为其他字体，如 Microsoft YaHei）
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 保存文件的文件夹路径
    output_dir = "codeforces_ratings"

    # 初始化一个列表来存储所有的ratings
    all_ratings = []

    # 遍历文件夹中所有的Excel文件
    for file_name in os.listdir(output_dir):
        if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            file_path = os.path.join(output_dir, file_name)
            try:
                # 读取Excel文件
                data = pd.read_excel(file_path)
                # 提取Rating列并追加到all_ratings列表
                all_ratings.extend(data['Rating'].dropna().tolist())
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

    # 将ratings数据转换为DataFrame便于统计和绘图
    ratings_df = pd.DataFrame(all_ratings, columns=['Rating'])

    # 过滤800分以上的数据
    ratings_df = ratings_df[ratings_df['Rating'] >= 800]

    # 定义范围、颜色和级别名称
    ranges = [
        (800, 1200, (128 / 255, 128 / 255, 128 / 255), "新手"),
        (1200, 1400, (0, 128 / 255, 0), "学徒"),
        (1400, 1600, (3 / 255, 168 / 255, 158 / 255), "专家"),
        (1600, 1900, (0, 0, 1), "高手"),
        (1900, 2100, (170 / 255, 0, 170 / 255), "候选大师"),
        (2100, 2300, (1, 140 / 255, 0), "大师"),
        (2300, 2400, (1, 140 / 255, 0), "国际大师"),
        (2400, 2600, (1, 0, 0), "特级大师"),
        (2600, 3000, (1, 0, 0), "国际特级大师")
    ]

    # 绘制直方图
    plt.figure(figsize=(12, 6))

    # 横坐标粒度为1，统计每个rating的频数
    bins = range(800, 3001)  # 粒度为1
    ratings_df['binned'] = pd.cut(ratings_df['Rating'], bins=bins, right=False)  # 创建分箱
    hist = ratings_df['binned'].value_counts(sort=False)  # 统计每个分箱的数量
    bin_edges = hist.index.categories.left  # 获取每个分箱的左边界

    # 绘制不同范围内的频数，使用对应的颜色填充
    for start, end, color, label in ranges:
        mask = (bin_edges >= start) & (bin_edges < end)
        plt.bar(bin_edges[mask], hist.values[mask], width=1, color=color, label=f'{label} ({start}-{end})')

    xticks = [800, 1200, 1400, 1600, 1900, 2100, 2300, 2400, 2600, 3000]  # 关键的分界点
    plt.xticks(xticks, labels=[str(x) for x in xticks])  # 转换为字符串

    # 图表美化
    plt.xlabel('评分')
    plt.ylabel('人数')
    plt.title('Codeforces 评分分布图')
    plt.legend(title='等级')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 保存为图片
    plt.savefig("all_ratings_distribution_correct_colors_with_names.png", dpi=300)
    print("Plot with names in legend saved as 'all_ratings_distribution_correct_colors_with_names.png'.")

def statistic_cleaned():
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 保存文件的文件夹路径
    ratings_dir = "codeforces_ratings"
    submissions_dir = "cleaned_user_submissions"

    # 获取需要统计的用户列表
    users_to_analyze = set()
    for folder_name in os.listdir(submissions_dir):
        if os.path.isdir(os.path.join(submissions_dir, folder_name)):
            if folder_name.endswith('_dot'):
                username = folder_name[:-4] + '.'  # 移除 '_dot' 并添加 '.'
            else:
                username = folder_name
            users_to_analyze.add(username)

    # 初始化一个列表来存储所有的ratings
    all_ratings = []
    # 遍历ratings文件夹中所有的Excel文件
    for file_name in os.listdir(ratings_dir):
        if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            file_path = os.path.join(ratings_dir, file_name)
            try:
                # 读取Excel文件
                data = pd.read_excel(file_path)
                # 只保留需要分析的用户的rating
                filtered_data = data[data['User Name'].isin(users_to_analyze)]
                # 提取Rating列并追加到all_ratings列表
                all_ratings.extend(filtered_data['Rating'].dropna().tolist())
            except Exception as e:
                print(f"Error reading {file_name}: {e}")


    # 将ratings数据转换为DataFrame便于统计和绘图
    ratings_df = pd.DataFrame(all_ratings, columns=['Rating'])

    # 过滤800分以上的数据
    ratings_df = ratings_df[ratings_df['Rating'] >= 800]

    # 定义范围、颜色和级别名称
    ranges = [
        (800, 1200, (128 / 255, 128 / 255, 128 / 255), "新手"),
        (1200, 1400, (0, 128 / 255, 0), "学徒"),
        (1400, 1600, (3 / 255, 168 / 255, 158 / 255), "专家"),
        (1600, 1900, (0, 0, 1), "高手"),
        (1900, 2100, (170 / 255, 0, 170 / 255), "候选大师"),
        (2100, 2300, (1, 140 / 255, 0), "大师"),
        (2300, 2400, (1, 140 / 255, 0), "国际大师"),
        (2400, 2600, (1, 0, 0), "特级大师"),
        (2600, 3000, (1, 0, 0), "国际特级大师")
    ]

    # 绘制直方图
    plt.figure(figsize=(12, 6))

    # 横坐标粒度为1，统计每个rating的频数
    bins = range(800, 3001)  # 粒度为1
    ratings_df['binned'] = pd.cut(ratings_df['Rating'], bins=bins, right=False)  # 创建分箱
    hist = ratings_df['binned'].value_counts(sort=False)  # 统计每个分箱的数量
    bin_edges = hist.index.categories.left  # 获取每个分箱的左边界

    # 绘制不同范围内的频数，使用对应的颜色填充
    for start, end, color, label in ranges:
        mask = (bin_edges >= start) & (bin_edges < end)
        plt.bar(bin_edges[mask], hist.values[mask], width=1, color=color, label=f'{label} ({start}-{end})')

    xticks = [800, 1200, 1400, 1600, 1900, 2100, 2300, 2400, 2600, 3000]  # 关键的分界点
    plt.xticks(xticks, labels=[str(x) for x in xticks])  # 转换为字符串

    # 图表美化
    plt.xlabel('评分')
    plt.ylabel('人数')
    plt.title('Codeforces 评分分布图')
    plt.legend(title='等级')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 保存为图片
    plt.savefig("cleaned_ratings_distribution_correct_colors_with_names.png", dpi=300)
    print("Plot with names in legend saved as 'cleaned_ratings_distribution_correct_colors_with_names.png'.")


def statistic_avg_rating():
    import os
    import pandas as pd
    from tqdm import tqdm  # 导入 tqdm

    # 定义路径
    cleaned_users_folder = r"C:\Users\CharmingZe\Desktop\毕设\codeforces爬取数据\problem_info\cleaned_user_submissions"
    ratings_folder = r"C:\Users\CharmingZe\Desktop\毕设\codeforces爬取数据\problem_info\codeforces_ratings"

    # 获取清理后的用户名，并将结尾的 _dot 替换为 .
    cleaned_users = []
    for user_folder in os.listdir(cleaned_users_folder):
        if os.path.isdir(os.path.join(cleaned_users_folder, user_folder)):
            # 检查是否以 _dot 结尾，如果是则替换为 .
            if user_folder.endswith("_dot"):
                cleaned_username = user_folder[:-4] + "."  # 将结尾的 _dot 替换为 .
            else:
                cleaned_username = user_folder  # 不符合规则的保持原样
            cleaned_users.append(cleaned_username)

    print(f"清理后的用户名数量: {len(cleaned_users)}")

    # 存储所有用户的 Rating 数据
    all_ratings_data = []

    # 记录未匹配的用户名
    unmatched_users = set(cleaned_users)  # 使用集合存储未匹配的用户名

    # 遍历 ratings 文件夹中的所有 xlsx 文件
    xlsx_files = [f for f in os.listdir(ratings_folder) if f.endswith(".xlsx")]

    # 使用 tqdm 添加进度条
    for file_name in tqdm(xlsx_files, desc="处理 XLSX 文件", unit="文件"):
        file_path = os.path.join(ratings_folder, file_name)
        try:
            # 读取 xlsx 文件
            df = pd.read_excel(file_path)

            # 确保列名正确（大小写可能不同）
            df.columns = df.columns.str.strip()  # 去掉列名中的空格
            df.rename(columns={"User Name": "User_Name"}, inplace=True)  # 统一列名

            # 过滤出清理后的用户名对应的 Rating 数据
            matched_data = df[df["User_Name"].isin(cleaned_users)]
            all_ratings_data.append(matched_data)

            # 更新未匹配的用户名集合
            matched_users_in_file = set(matched_data["User_Name"])
            unmatched_users -= matched_users_in_file
        except Exception as e:
            print(f"读取文件 {file_name} 出错: {e}")

    # 合并所有匹配的数据
    if all_ratings_data:
        combined_data = pd.concat(all_ratings_data, ignore_index=True)

        # 去重（可能存在重复的用户名）
        combined_data.drop_duplicates(subset=["User_Name"], inplace=True)

        # 计算平均 Rating
        average_rating = combined_data["Rating"].mean()
        print("\n统计结果：")
        print(f"清理后的用户数量: {len(combined_data)}")
        print(f"平均 Rating 分数: {average_rating:.2f}")
    else:
        print("未找到任何匹配的用户数据。")

    # 输出未匹配的用户名
    if unmatched_users:
        print("\n以下用户名未在 XLSX 文件中找到：")
        for user in sorted(unmatched_users):
            print(user)
    else:
        print("\n所有用户名均已匹配。")
if __name__ == '__main__':
    statistic_avg_rating()
