import os
import logging
from datetime import datetime
from util import get_config_string


def setup_logger(model, log_base_path, arg, additional_info=''):
    """
    配置日志记录器，包含参数配置信息
    """
    # 获取开始时间和模型名称
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = type(model).__name__
    config_str = get_config_string(arg)

    # 创建包含配置信息的子文件夹
    log_base_path = os.path.normpath(log_base_path)
    log_dir = os.path.join(log_base_path, config_str)
    if additional_info:
        additional_info = f"_{additional_info}"
    log_file = os.path.join(log_dir, f"{model_name}_{start_time}{additional_info}.log")

    # 创建日志文件夹
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 配置日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)

    # 重置之前的处理器
    logging.getLogger().handlers = []

    # 创建并配置文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    # 创建并配置标准输出处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 获取根日志记录器并添加处理器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # 记录配置信息
    logging.info(f"Model Configuration: {arg}")
    logging.info(f"Logger initialized for {model_name}. Log file: {log_file}")

    return log_file
