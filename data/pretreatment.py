import json
import pickle
from random import sample, seed

import pandas as pd
from tqdm import tqdm

from data.SubmitDetail import SubmitDetail
from util import *


def load_problem_details(excel_path=Constants.PROBLEM_EXCEL_PATH):
    """
    从Excel文件加载题目详细信息

    :param excel_path: Excel文件路径
    :return: 以(cid, qindex)为键的题目详细信息字典
    """
    # 读取Excel文件
    df = pd.read_excel(excel_path)

    # 创建一个字典，使用 'cid_qindex' 作为键
    problem_details = {}
    for _, row in df.iterrows():
        # 构造键
        key = f"{row['cid']}_{row['qindex']}"
        # 检查content是否为空
        # 将整行转换为字典并存储
        problem_details[key] = row.to_dict()
    return problem_details


def convert_submission_from_original(submission, problem_details):
    """
    转换提交信息，并添加题目内容

    :param submission: 原始提交信息
    :param problem_details: 题目详细信息字典
    :return: SubmitDetail 对象
    """
    cid = submission.get("contestId")
    qindex = submission["problem"]["index"] if submission.get("problem") else None

    # 尝试获取题目内容
    content = problem_details.get((cid, qindex), "")
    # 直接创建并返回 SubmitDetail 对象
    return SubmitDetail(
        cid=cid,
        cfid=submission["author"]["members"][0]["handle"] if submission.get("author") and submission["author"].get("members") else None,
        qindex=qindex,
        difficulty=submission["problem"].get("rating") if submission.get("problem") else None,
        name=submission["problem"].get("name") if submission.get("problem") else None,
        tags=",".join(submission["problem"].get("tags", [])) if submission.get("problem") else "",
        status=submission.get("verdict"),
        submitTime=submission.get("creationTimeSeconds"),
        content=content
    )


def convert_submission_from_front_end(submission, problem_details):
    """
    Convert a frontend submission to SubmitDetail object using problem_details.

    Args:
        submission (dict): Frontend submission format e.g. {'problem_id': '1419_A', 'status': 'COMPILATION_ERROR'}
        problem_details (pandas.DataFrame): DataFrame containing problem information

    Returns:
        SubmitDetail: Converted submission object with all required fields

    Raises:
        KeyError: If required fields are missing
        ValueError: If problem_id format is invalid or problem not found
    """
    try:
        # Extract problem_id and status from submission
        problem_id = submission.get('problem_id')
        status = submission.get('status')

        if not problem_id or not status:
            raise KeyError("Missing required fields: problem_id or status")

        # Split problem_id into cid and qindex
        # Assuming problem_id format is like '1419_A' where 1419 is cid and A is qindex
        try:
            cid, qindex = problem_id.split('_')
        except ValueError:
            raise ValueError(f"Invalid problem_id format: {problem_id}. Expected format: 'cid_qindex'")

        # Find matching problem in problem_details
        # Assuming problem_details has columns: 'cid', 'qindex', 'difficulty', 'tags', 'name', 'content', 'cfid'
        problem_row = problem_details[
            (problem_details['cid'] == cid) &
            (problem_details['qindex'] == qindex)
            ]

        if problem_row.empty:
            raise ValueError(f"Problem not found in problem_details: {problem_id}")

        # Get the first matching row (should be unique)
        problem = problem_row.iloc[0]

        # Create SubmitDetail object with all required fields
        submit_detail = SubmitDetail(
            cid=cid,
            cfid=problem.get('cfid', ''),  # Use empty string if cfid not available
            qindex=qindex,
            submitTime=submission.get('submitTime', None),  # Use None if not provided
            difficulty=problem['difficulty'],
            tags=problem['tags'],
            name=problem.get('name', ''),
            content=problem['content'],
            status=status
        )

        return submit_detail

    except KeyError as e:
        raise KeyError(f"Submission conversion failed: {str(e)}")
    except ValueError as e:
        raise ValueError(f"Submission conversion failed: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error during submission conversion: {str(e)}")


def getSubmissionsByCfid(submissions_dir=Constants.DATA_PATH, use_cache=True, cache_file=Constants.TRAIN_DATA_PKL_PATH):
    """
    获取所有用户的提交记录，支持缓存功能

    Args:
        submissions_dir: 提交记录所在目录
        use_cache: 是否使用缓存
        cache_file: 缓存文件路径

    Returns:
        dict: 用户ID到提交记录的映射
    """
    # 如果使用缓存且缓存文件存在，直接返回缓存的结果
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                print(f"Loading cached submissions from {cache_file}")
                cached_submissions = pickle.load(f)
                # 添加过滤逻辑
                filtered_submissions = {
                    cfid: [sub for sub in subs if sub.content and sub.difficulty and sub.tags]
                    for cfid, subs in cached_submissions.items()
                }
                # 再次过滤：移除空列表的用户
                filtered_submissions = {
                    cfid: subs for cfid, subs in filtered_submissions.items()
                    if len(subs) > 0
                }
                return filtered_submissions
        except Exception as e:
            print(f"Error loading cache: {e}")
            # 如果加载缓存失败，继续执行常规加载流程

    submissions_by_cfid = {}
    # 首先加载题目详细信息
    problem_details = load_problem_details()
    user_dirs = [d for d in os.listdir(submissions_dir) if os.path.isdir(os.path.join(submissions_dir, d))]

    for user_dir in tqdm(user_dirs, desc="Loading user submissions"):
        user_path = os.path.join(submissions_dir, user_dir)
        submission_file = f"submissions_{user_dir}.json"
        submission_path = os.path.join(user_path, submission_file)

        if os.path.exists(submission_path):
            try:
                with open(submission_path, 'r', encoding='utf-8') as f:
                    submissions = json.load(f)
                    # 反转submissions列表，使其变为正序
                    submissions.reverse()
                    converted_submissions = [
                        sub for sub in [convert_submission(s, problem_details) for s in submissions]
                        if sub.content and sub.difficulty and sub.tags # 只保留content、diff、tags不为空的提交
                    ]
                    # 计算过滤比例
                    original_count = len(submissions)
                    filtered_count = len(converted_submissions)
                    if original_count > 0:  # 避免除以0
                        removed_percentage = (original_count - filtered_count) / original_count * 100
                    else:
                        removed_percentage = 0

                    # 如果过滤后还有提交，且过滤比例不超过5%，则添加到字典中
                    if filtered_count > 0 and removed_percentage <= 5:
                        submissions_by_cfid[user_dir] = converted_submissions

            except Exception as e:
                print(f"Error reading or converting file for user {user_dir}: {e}")
    # 保存缓存
    if use_cache:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(submissions_by_cfid, f)
                print(f"Saved submissions cache to {cache_file}")
        except Exception as e:
            print(f"Error saving cache: {e}")

    return submissions_by_cfid


# 评估模型在面对完全未见过的用户时的泛化能力。
def split_user(frac=0.8, rand_seed=101):
    # 按照用户组织提交记录
    submissions_by_cfid = getSubmissionsByCfid()
    # 随机划分用户
    seed(rand_seed)
    all_users = list(submissions_by_cfid.keys())
    train_size = int(len(all_users) * frac)
    train_users = set(sample(all_users, train_size))

    # 划分数据集
    train_data = {}
    test_data = {}
    for cfid, submissions in submissions_by_cfid.items():
        if cfid in train_users:
            train_data[cfid] = submissions
        else:
            test_data[cfid] = submissions

    # 计算总提交数
    train_submission_count = sum(len(submissions) for submissions in train_data.values())
    test_submission_count = sum(len(submissions) for submissions in test_data.values())

    print(f"总用户数: {len(submissions_by_cfid)}")
    print(f"训练集用户数: {len(train_users)}")
    print(f"测试集用户数: {len(test_data)}")  # 使用 submissions_by_cfid 来确保一致性
    print(f"训练集提交数: {train_submission_count}")
    print(f"测试集提交数: {test_submission_count}")
    return train_data, test_data


# split_future 方法：基于时间序列，评估模型根据用户过去的行为预测未来表现的能力。
# 这个在测试的时候应该先获取训练集的h
def split_future(frac=0.8):
    # 按照用户组织提交记录
    submissions_by_cfid = getSubmissionsByCfid()
    # 划分数据集
    train_data = {}
    test_data = {}
    for cfid, submissions in submissions_by_cfid.items():
        num_submissions = len(submissions)
        train_len = int(num_submissions * frac)
        train_submissions = submissions[:train_len]
        test_submissions = submissions[train_len:]
        train_data[cfid] = train_submissions
        test_data[cfid] = test_submissions

    total_submissions = sum(len(submissions) for submissions in submissions_by_cfid.values())
    train_submission_count = sum(len(submissions) for submissions in train_data.values())
    test_submission_count = sum(len(submissions) for submissions in test_data.values())
    print(f"总提交数: {total_submissions}")
    print(f"训练集提交数: {train_submission_count}")
    print(f"测试集提交数: {test_submission_count}")
    return train_data, test_data



def get_submissions_without_content(problem_details, submissions_dir=Constants.DATA_PATH):
    """
    统计提交记录中没有 content 的题目，并基于 cid 和 qindex 去重

    Args:
        submissions_dir (str): 提交记录所在目录
        problem_details (dict): 题目详细信息

    Returns:
        set: 没有 content 的题目的 (cid, qindex) 集合
    """
    no_content_set = set()

    # 获取用户目录列表
    user_dirs = [d for d in os.listdir(submissions_dir) if os.path.isdir(os.path.join(submissions_dir, d))]

    for user_dir in tqdm(user_dirs, desc="Checking submissions for missing content"):
        user_path = os.path.join(submissions_dir, user_dir)
        submission_file = f"submissions_{user_dir}.json"
        submission_path = os.path.join(user_path, submission_file)

        if os.path.exists(submission_path):
            try:
                with open(submission_path, 'r', encoding='utf-8') as f:
                    submissions = json.load(f)
                    for s in submissions:
                        converted = convert_submission(s, problem_details)
                        if not converted.content:
                            # 如果没有 content，记录 (cid, qindex)
                            cid = s.get("contestId",10000)
                            qindex = s.get("problem").get("index")
                            if cid <5000:
                                no_content_set.add((cid, qindex))

            except Exception as e:
                print(f"Error processing {submission_path}: {s}")

    return no_content_set

    return no_content_stats
if __name__ == '__main__':
    getSubmissionsByCfid(use_cache=False)
