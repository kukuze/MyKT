LR = 0.005
EPOCH = 10000
TAGS = [
    "brute force",
    "meet-in-the-middle",
    "constructive algorithms",
    "graphs",
    "shortest paths",
    "bitmasks",
    "dp",
    "trees",
    "greedy",
    "interactive",
    "sortings",
    "data structures",
    "dsu",
    "math",
    "strings",
    "combinatorics",
    "binary search",
    "implementation",
    "dfs and similar",
    "two pointers",
    "games",
    "fft",
    "probabilities",
    "flows",
    "graph matchings",
    "divide and conquer",
    "hashing",
    "number theory",
    "chinese remainder theorem",
    "*special",
    "geometry",
    "matrices",
    "ternary search",
    "string suffix structures",
    "schedules",
    "2-sat",
    "expression parsing"
]
DIFFS = [
    800,
    900,
    1000,
    1100,
    1200,
    1300,
    1400,
    1500,
    1600,
    1700,
    1800,
    1900,
    2000,
    2100,
    2200,
    2300,
    2400,
    2500,
    2600,
    2700,
    2800,
    2900,
    3000,
    3100,
    3200,
    3300,
    3400,
    3500,
]
DIFFS_MIN = min(DIFFS)  # 800
DIFFS_MAX = max(DIFFS)  # 3500
DIFFICULTY_EMB_DIM = 25
# input dimension
TEXT_FEATURE_DIM = 768  # 题目文本的维度
# hidden layer dimension
HIDDEN_DIM = 150  # RNN隐藏维度
TAGS_NUM = len(TAGS)
TAG_EMB_DIM = 25
DIFFS_NUM = len(DIFFS)
# BASE_PATH = "/workspace/MyKt"
BASE_PATH = "D:/MyKT"
BERT_MODEL_CACHE_PATH = BASE_PATH + "/model/bert"
MODEL_SNAPSHOT_PATH = BASE_PATH + "/snapshots"
LOG_PATH = BASE_PATH + "/log"
JSON_PATH = BASE_PATH + "/data"
DATA_PATH = r"C:\Users\CharmingZe\Desktop\毕设\codeforces爬取数据\problem_info\test_user_submissions"  # 所有用户的提交记录文件夹
TRAIN_DATA_PKL_PATH = BASE_PATH + "/data/submissions_cache.pkl"
PROBLEM_EXCEL_PATH = BASE_PATH + "/data/cf_problem_info.xlsx"
openai_api_key = ""
openai_url = "yunwu.ai"
EMBEDDING_MODEL = "bert"  # bert、openai
