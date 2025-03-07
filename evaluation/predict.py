import os
import pickle

import torch
from flask import Flask, request, jsonify
from torch import sigmoid

from constant import Constants
from data.SubmitDetail import SubmitDetail
from data.pretreatment import load_problem_details, convert_submission_from_front_end
from model.EKTM_D.EKTM_D import EKTM_D

app = Flask(__name__)

# 全局变量（在应用启动时初始化）
MODEL_PATH = "your_model.pth"  # 替换为你的模型路径
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = None


def initialize_model():
    """初始化模型并加载权重"""
    global model
    try:
        # 初始化模型（根据你的实际参数调整）
        Constants.CUDA = 0
        model = EKTM_D()
        # model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"模型初始化失败: {str(e)}")


def create_response(data=None, code=200, message="Success"):
    response = {"code": code, "message": message}
    if data is not None:
        response.update(data)
    return jsonify(response), code


@app.route('/predict_score', methods=['POST'])
def predict_score_api():
    data = request.get_json()
    if not data:
        return create_response(code=400, message="请求数据为空")
    refSeq = data.get('refSeq', [])
    pred_seq = data.get('pred_seq', [])

    h = None
    details = load_problem_details()  # 确保线程安全
    # 提前转换所有 refSeq 和 pred_seq 元素
    processed_ref_items = [convert_submission(item, details) for item in refSeq]
    processed_pred_items = [convert_submission(item, details) for item in pred_seq]

    # 过滤掉不符合条件的转换结果
    valid_ref_items = [
        item for item in processed_ref_items
        if all(hasattr(item, attr) for attr in ['content', 'difficulty', 'tags'])
    ]
    valid_pred_items = [
        item for item in processed_pred_items
        if all(hasattr(item, attr) for attr in ['content', 'difficulty', 'tags'])
    ]

    if not valid_ref_items:
        return create_response(code=400, message="没有有效的参考项目可用于参考")
    if not valid_pred_items:
        return create_response(code=400, message="没有有效的预测项目可用于预测")

    with torch.no_grad():
        # 处理参考序列
        for processed_item in valid_ref_items:
            res = torch.tensor(
                [1.0 if processed_item.status == "OK" else 0.0],
                device=DEVICE
            )
            _, h = model(processed_item, res, h)
        # 处理预测序列并计算分数
        scores = {}
        for processed_item in valid_pred_items:
            key = f"{processed_item.cid}_{processed_item.qindex}"
            res = torch.tensor(
                [1.0 if processed_item.status == "OK" else 0.0],
                device=DEVICE
            )
            s, h = model(processed_item, res, h)
            scores[key] = sigmoid(s).item()
    return jsonify({"scores": scores})


@app.route('/predict_status', methods=['POST'])
def predict_status_api():
    """Predict student status based on reference sequence"""
    data = request.get_json()
    if not data:
        return create_response(code=400, message="请求数据为空")

    refSeq = data.get('refSeq', [])
    if not refSeq:
        return create_response(code=400, message="没有可参考的数据")

    # Calculate student state using the same logic as pareto_recommend
    h, current_student_status = calculate_student_h_and_status(refSeq)
    if isinstance(h, tuple):  # Check if error response was returned
        return h

    # Return the predicted status
    return create_response({"recommendations": current_student_status})


import pandas as pd


def filter_problems_by_rating(ref_seq, rating, lower_bound=300, upper_bound=300,
                              problem_file='D:\MyKT\data\cf_problem_info.xlsx'):
    """
    根据用户评分和历史记录筛选挑战性题目，区分已解决和未解决题目。

    参数:
        ref_seq (list): 用户的历史记录，例如 [{'problem_id': '123_1', 'status': 'OK'}]
        rating (int): 用户当前评分
        problem_file (str): XLSX 文件路径

    返回:
        tuple: (solved_problems, unsolved_problems)，分别是已解决和未解决的挑战性题目列表
    """
    try:
        # 加载题目数据
        problem_df = pd.read_excel(problem_file)

        # 定义挑战性题目的评分范围
        min_rating = max(0, rating - lower_bound)  # 下限
        max_rating = rating + upper_bound  # 上限

        # 筛选挑战性题目
        filtered_df = problem_df[
            (problem_df['difficulty'] >= min_rating) &
            (problem_df['difficulty'] <= max_rating)
            ]

        # 从 ref_seq 中提取已解决的 problem_id
        solved_ids = {item['problem_id'] for item in ref_seq if item.get('status') == 'OK'}

        # 分离已解决和未解决题目
        solved_problems = []
        unsolved_problems = []

        for _, row in filtered_df.iterrows():
            if pd.isna(row['content']):
                print(f"{row['cid']}_{row['qindex']} content为空")
                continue

            problem = SubmitDetail(
                cid=str(row['cid']),
                cfid=str(row.get('cfid', '')),
                qindex=str(row['qindex']),
                submitTime=None,
                difficulty=row['difficulty'],
                tags=row['tags'],
                name=row.get('name', ''),
                content=row['content'],
                status='UNKNOWN'  # 默认状态，后续根据 ref_seq 调整
            )
            problem_id = f"{problem.cid}_{problem.qindex}"

            if problem_id in solved_ids:
                problem.status = 'OK'  # 已解决题目
                solved_problems.append(problem)
            else:
                unsolved_problems.append(problem)

        return solved_problems, unsolved_problems

    except Exception as e:
        raise Exception(f"筛选题目时出错: {str(e)}")


@app.route('/pareto_recommend', methods=['POST'])
def pareto_recommend():
    """
    帕累托最优推荐策略
    """

    data = request.get_json()
    refSeq = data.get('refSeq', {})
    if not refSeq:
        return create_response(code=400, message="没有可参考的数据")
    rating = data.get("rating", 800)
    solved_problems, unsolved_problems = filter_problems_by_rating(refSeq, rating)
    problem_set = solved_problems + unsolved_problems  # 合并所有题目
    problem_set = problem_set[0:10]  # 取前 10 个
    selected_types = data.get('selectedTypes') or list(range(0, Constants.TAGS_NUM))
    selected_difficulties = data.get('selectedDifficulties') or list(range(0, Constants.DIFFS_NUM))
    h, current_student_status = calculate_student_h_and_status(refSeq)
    if isinstance(h, tuple):  # Check if error response was returned
        return h

    # 预测每道题的能力提升量和解题成功率
    recommendations = []
    for problem in problem_set:
        if not (problem.content and
                problem.difficulty is not None and  # difficulty might be 0, so check None explicitly
                problem.tags):  # tags could be empty list, adjust based on needs
            continue
        # 继续处理有效的 problem

        # 计算能力提升量
        delta = calculate_ability_gain(h, current_student_status, problem, selected_types,
                                       selected_difficulties)

        # 预测解题成功率
        success_rate = predict_success_rate(h, problem)

        # 添加到推荐列表
        recommendations.append({
            'problem_id': problem.cid + "_" + problem.qindex,
            'delta': delta,
            'success_rate': success_rate
        })

    # 按帕累托最优排序
    recommendations.sort(key=lambda x: (-x['delta'], -x['success_rate']))

    return create_response({"recommendations": recommendations})


@app.route('/similarity_recommend', methods=['POST'])
def similarity_recommend():
    """
    基于特征相似性的学习推荐策略
    """

    data = request.get_json()
    if not data:
        return create_response(code=400, message="请求数据为空")
    rating = data.get('rating', 800)
    refSeq = data.get('refSeq', {})
    if not refSeq:
        return create_response(code=400, message="没有可参考的数据")
    selected_types = data.get('selectedTypes') or list(range(0, Constants.TAGS_NUM))
    selected_difficulties = data.get('selectedDifficulties') or list(range(0, Constants.DIFFS_NUM))
    h, current_student_status = calculate_student_h_and_status(refSeq)
    if isinstance(h, tuple):
        return h
    # 筛选挑战性题目
    # 获取已解决和未解决的挑战性题目
    solved_problems, unsolved_problems = filter_problems_by_rating(refSeq, rating)
    if not solved_problems:
        return create_response(code=400, message="没有符合条件的已解决题目")
    if not unsolved_problems:
        return create_response(code=400, message="没有符合条件的未解决题目")

    # 为每个已做过的题目找到最相似的未解决题目
    candidate_pairs = find_most_similar_matrix(solved_problems, unsolved_problems)
    if not candidate_pairs:
        return create_response(code=400, message="没有找到相似的未解决题目")

    # 评估候选题目的学习价值
    recommendations = []
    for solved_problem, candidate_problem in candidate_pairs:
        if not (candidate_problem.content and candidate_problem.difficulty is not None and candidate_problem.tags):
            continue
        delta = calculate_ability_gain(h, current_student_status, candidate_problem, selected_types,
                                       selected_difficulties)
        success_rate = predict_success_rate(h, candidate_problem)
        recommendations.append({
            'problem_id': f"{candidate_problem.cid}_{candidate_problem.qindex}",  # 未解决题目 ID
            'delta': delta,
            'success_rate': success_rate,
            'similar_to': f"{solved_problem.cid}_{solved_problem.qindex}"  # 对应的已解决题目 ID
        })

        # 按帕累托最优排序
    recommendations.sort(key=lambda x: (-x['delta'], -x['success_rate']))
    return create_response({"recommendations": recommendations})


@app.route('/progressive_recommend', methods=['POST'])
def progressive_recommend():
    """
    渐进式解题推荐策略
    """

    data = request.get_json()
    target_problem = data.get('targetProblemId', {})
    refSeq = data.get('refSeq', {})
    rating = data.get("rating", 800)  # Default rating if not provided
    # 输入验证
    if not refSeq:
        return create_response(code=400, message="没有可参考的数据")
    if not target_problem:
        return create_response(code=400, message="目标题目数据为空")
    target_problem_detail, error_response = validate_and_load_target_problem(target_problem)
    if error_response:
        return error_response
    h, _ = calculate_student_h_and_status(refSeq)
    if isinstance(h, tuple):  # Check if error response was returned
        return h
    # Filter problems using filter_problems_by_rating
    solved_problems, unsolved_problems = filter_problems_by_rating(refSeq, rating, 500, 0)
    problems = solved_problems + unsolved_problems
    # 计算一下原来目标题目解题成功率多少
    target_problem_rating, _ = model(target_problem_detail, torch.FloatTensor([1.0]).to(model.device), h)
    target_problem_rating = sigmoid(target_problem_rating).item()
    # 计算前置题目对目标的成功率提升
    recommendations = []
    for problem in problems[0:10]:
        # 转换题目数据
        delta_r = calculate_rating_gain(h, problem, target_problem_detail, target_problem_rating)
        success_rate_problem = predict_success_rate(h, problem)
        recommendations.append({
            'problem_id': f"{problem.cid}_{problem.qindex}",
            'delta_r': delta_r,
            'success_rate': success_rate_problem
        })

    # 按帕累托最优排序
    recommendations.sort(key=lambda x: (-x['delta_r'], -x['success_rate']))

    return create_response({"recommendations": recommendations})


def validate_and_load_target_problem(target_problem_id, problem_file='D:\\MyKT\\data\\cf_problem_info.xlsx'):
    """
    Validate and load target problem details from an Excel file.

    参数:
        target_problem_id (str): Target problem data from the request, e.g., '456_A'
        problem_file (str): Path to the Excel file containing problem details

    返回:
        tuple: (SubmitDetail object or None, response tuple or None)
            - If valid: (SubmitDetail object, None)
            - If invalid: (None, (json_response, status_code))
    """
    # Extract target problem ID
    if not target_problem_id or '_' not in target_problem_id:
        return None, create_response(code=400, message="目标题目 ID 格式无效，需为 'cid_qindex'")

    target_cid, target_qindex = target_problem_id.split('_')

    # Load problem data from Excel file
    try:
        problem_df = pd.read_excel(problem_file)
    except Exception as e:
        return None, create_response(code=500, message=f"无法加载题目数据文件: {str(e)}")

    # Check if target_problem exists in the Excel file
    target_exists = problem_df[
        (problem_df['cid'].astype(str) == target_cid) &
        (problem_df['qindex'].astype(str) == target_qindex)
        ]
    if target_exists.empty:
        return None, create_response(code=404, message=f"目标题目 {target_problem_id} 在数据文件中不存在")

    # Convert to SubmitDetail object
    target_row = target_exists.iloc[0]
    target_problem_detail = SubmitDetail(
        cid=str(target_row['cid']),
        cfid=str(target_row.get('cfid', '')),
        qindex=str(target_row['qindex']),
        submitTime=None,
        difficulty=target_row['difficulty'],
        tags=target_row['tags'],
        name=target_row.get('name', ''),
        content=target_row['content'],
        status='UNKNOWN'
    )

    return target_problem_detail, None


def calculate_rating_gain(h, pre_problem, target_problem, target_problem_rating):
    with torch.no_grad():
        # 我们先假设做对了这道题 然后获取到了一个新的状态
        _, h = model(pre_problem, torch.FloatTensor([1.0]).to(model.device), h)
        # 用这个状态去预测目标题目的解题成功率
        s, _ = model(target_problem, torch.FloatTensor([1.0]).to(model.device), h)
        return sigmoid(s).item() - target_problem_rating


def calculate_ability_gain(h, current_student_state, problem, selected_types, selected_difficulties):
    """
    计算完成某道题目后的能力提升量
    """
    # 示例计算逻辑（需根据具体公式实现）
    l_t = current_student_state
    # 我们假设做对了这道题后的状态，然后用这个状态h 再用这个向量去预测数值类型的状态
    with torch.no_grad():
        _, h = model(problem, torch.FloatTensor([1.0 if problem.status == "OK" else 0.0]).to(model.device), h)
        s, _ = model(problem, problem.status, h, True)
        l_t_plus_1 = sigmoid(s).squeeze().cpu().numpy().tolist()
    delta = sum(
        l_t_plus_1[i][j] - l_t[i][j]
        for i in (selected_types)
        for j in (selected_difficulties)
    )
    return delta


def predict_success_rate(h, problem):
    """
    预测学生对某道题目的解题成功率
    """
    with torch.no_grad():
        res = torch.tensor([1.0 if problem.status == "OK" else 0.0], device=DEVICE)
        s, _ = model(problem, res, h)
        return sigmoid(s).item()


def find_most_similar_matrix(solved_problems, unsolved_problems):
    """
    使用矩阵运算为每个已解决题目在未解决题目中找到最相似题目
    """
    # 确定缓存路径和维度
    if Constants.EMBEDDING_MODEL == 'bert':
        cache_dir = r"D:\MyKT\model\textEmbedding\bertEmbeddingCache\bert-base-uncased"
        feature_dim = 768  # BERT 默认维度
    elif Constants.EMBEDDING_MODEL == 'openai':
        feature_dim = Constants.TEXT_FEATURE_DIM  # 根据配置动态选择
        cache_dir = r"D:\MyKT\model\textEmbedding\openaiEmbeddingCache\text-embedding-3-large-" + feature_dim

    else:
        raise ValueError("未知的 EMBEDDING_MODEL 配置")

    # 设备配置，与 BaseEmbedder 一致
    device = torch.device(f'cuda:{Constants.CUDA}' if torch.cuda.is_available() else 'cpu')
    # 加载已解决题目的特征向量
    solved_embeddings = []
    valid_solved_problems = []
    for problem in solved_problems:
        problem_id = f"{problem.cid}_{problem.qindex}"
        cache_path = os.path.join(cache_dir, f"{problem_id}.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    embedding = torch.load(f, map_location=device)  # 使用 torch.load
                embedding = embedding.squeeze(0)  # 转换为 [768]
                if embedding.shape[0] == feature_dim:  # 检查维度
                    solved_embeddings.append(embedding)
                    valid_solved_problems.append(problem)
            except Exception as e:
                print(f"加载 {problem_id} 的缓存文件失败: {e}")
        else:
            print(f"未找到 {problem_id} 的缓存文件，跳过")
    # 加载未解决题目的特征向量
    unsolved_embeddings = []
    valid_unsolved_problems = []
    for problem in unsolved_problems:
        problem_id = f"{problem.cid}_{problem.qindex}"
        cache_path = os.path.join(cache_dir, f"{problem_id}.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    embedding = torch.load(f, map_location=device)  # 使用 torch.load
                embedding = embedding.squeeze(0)  # 转换为 [768]
                if embedding.shape[0] == feature_dim:  # 检查维度
                    unsolved_embeddings.append(embedding)
                    valid_unsolved_problems.append(problem)
            except Exception as e:
                print(f"加载 {problem_id} 的缓存文件失败: {e}")
        else:
            print(f"未找到 {problem_id} 的缓存文件，跳过")
    if not solved_embeddings or not unsolved_embeddings:
        return []
    # 转换为张量
    solved_embeddings = torch.stack(solved_embeddings)  # Shape: [S, D]
    unsolved_embeddings = torch.stack(unsolved_embeddings)  # Shape: [U, D]
    # 计算余弦相似度矩阵
    solved_norms = torch.norm(solved_embeddings, dim=1, keepdim=True)  # Shape: [S, 1]
    unsolved_norms = torch.norm(unsolved_embeddings, dim=1, keepdim=True)  # Shape: [U, 1]
    normalized_solved = solved_embeddings / solved_norms  # Shape: [S, D]
    normalized_unsolved = unsolved_embeddings / unsolved_norms  # Shape: [U, D]
    similarity_matrix = torch.matmul(normalized_solved, normalized_unsolved.T)  # Shape: [S, U]
    # 找到每个已解决题目的最相似未解决题目索引
    max_sim_indices = torch.argmax(similarity_matrix, dim=1)  # Shape: [S]
    # 提取候选对 (solved_problem, unsolved_problem)
    candidate_pairs = []
    for i, idx in enumerate(max_sim_indices):
        candidate_pairs.append((valid_solved_problems[i], valid_unsolved_problems[idx.item()]))
    return candidate_pairs


def cosine_similarity(vec1, vec2):
    """
    计算余弦相似度
    """
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_vec1 = sum(a ** 2 for a in vec1) ** 0.5
    norm_vec2 = sum(b ** 2 for b in vec2) ** 0.5
    return dot_product / (norm_vec1 * norm_vec2)


def calculate_student_h_and_status(ref_seq):
    try:
        h = None
        details = load_problem_details()
        processed_items = []
        for item in ref_seq:
            if 'problem_id' not in item or 'status' not in item:
                continue
            problem_id = item['problem_id']
            if problem_id not in details:
                continue
            # Get problem details and combine with submission status
            problem_detail = details[problem_id]
            cid, qindex = problem_id.split('_')
            processed_item = SubmitDetail(
                cid=cid,
                cfid=problem_detail.get('cfid', ''),
                qindex=qindex,
                submitTime=item.get('submitTime', None),
                difficulty=problem_detail['difficulty'],
                tags=problem_detail['tags'],
                name=problem_detail.get('name', ''),
                content=problem_detail['content'],
                status=item['status']
            )
            processed_items.append(processed_item)

        with torch.no_grad():
            for processed_item in processed_items:
                res = torch.tensor([1.0 if processed_item.status == "OK" else 0.0], device=DEVICE)
                _, h = model(processed_item, res, h)
            s, _ = model(processed_item, res, h, True)
            status = sigmoid(s).squeeze().cpu().numpy().tolist()

        return h, status
    except Exception as e:
        return create_response(code=500, message=f"计算学生状态失败: {str(e)}")


if __name__ == '__main__':
    # 初始化模型
    initialize_model()
    # 启动服务（生产环境建议使用 gunicorn）
    app.run(host='0.0.0.0', port=5000)
