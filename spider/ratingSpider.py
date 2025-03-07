from bs4 import BeautifulSoup
import cloudscraper
import re
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 获取已爬取的最大页码
def get_last_page(output_dir):
    # 列出目录中的所有文件
    files = os.listdir(output_dir)
    # 使用正则匹配文件名中的页码
    page_numbers = []
    for file in files:
        match = re.match(r"ratings_page_(\d+)\.xlsx", file)
        if match:
            page_numbers.append(int(match.group(1)))
    # 返回最大页码
    return max(page_numbers) if page_numbers else 0
def getRating():
    # 初始化 cloudscraper
    scraper = cloudscraper.create_scraper()

    # 保存文件的文件夹路径
    output_dir = "codeforces_ratings"
    os.makedirs(output_dir, exist_ok=True)

    # 基础 URL
    base_url = "https://codeforces.com/ratings/page/"
    start_page = get_last_page(output_dir) + 1
    # 遍历页面
    for page in range(start_page, 855):
        print(f"Processing page {page}...")
        url = f"{base_url}{page}"

        # 使用 cloudscraper 发送请求
        response = scraper.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch page {page}, status code: {response.status_code}")
            break

        # 使用 BeautifulSoup 解析 HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        trs = soup.find_all("tr")
        page_results = []
        # 提取表格中的数据
        for tr in trs:
            # 查找所有 <td> 标签
            tds = tr.find_all("td")
            if len(tds) >= 4:  # 确保有足够的数据
                # 提取名字部分
                name_tag = tds[1].find("a")
                if name_tag:
                    user_name = name_tag.text.strip()
                else:
                    continue
                # 提取分数部分
                user_rating = tds[3].text.strip()
                # 添加到结果列表
                page_results.append({
                    "User Name": user_name,
                    "Rating": user_rating
                })
                print(page_results.__len__())

            # 保存本页的数据到 Excel
        if page_results:
            df = pd.DataFrame(page_results)
            output_file = os.path.join(output_dir, f"ratings_page_{page}.xlsx")
            df.to_excel(output_file, index=False)
            print(f"Page {page} data saved to {output_file}")

    print("All pages processed.")