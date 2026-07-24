# -*-coding : utf-8 -*-
# @Author   : zhangtao
# @FileName : config.py
# @Desc     : 
# @Time     : 2025/11/8 17:14
# @Software : PyCharm
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# 获取当前文件的绝对路径
current_file_path = Path(__file__).resolve()
root_path = current_file_path.parent.parent

class Settings(BaseSettings):
    # 启动配置
    app_name: str = "ai_analyse"
    host: str = "0.0.0.0"
    port: int = 8000
    # NPU训练进程不应与Uvicorn热重载并用，生产环境默认关闭。
    uvicorn_reload: bool = False

    # 模型保存路径
    save_model_path: str = f"{root_path}/models/"
    # 模型信息文件路径
    models_info_table: str = f"{root_path}/models/model_info.json"

    # 训练集保存路径
    save_dataset_path: str = f"{root_path}/output/datasets/"

    # 单机训练进度文件目录，用于FastAPI父进程与NPU训练子进程交换百分比。
    training_progress_dir: str = f"{root_path}/output/training_progress/"

    # 模型训练完成后通知业务系统；支持通过同名环境变量覆盖。
    training_callback_url: str = (
        "http://192.168.10.40:8080/product-archives/model.updateTrainingState.erren"
    )
    training_callback_timeout_seconds: float = 10.0
    training_callback_max_attempts: int = 3
    training_callback_retry_delay_seconds: float = 2.0
    training_callback_pending_dir: str = f"{root_path}/output/training_callbacks/"
    training_callback_retry_interval_seconds: int = 60

    # 训练数据集
    train_dataset_path: str = f"{root_path}/core/datasets/train_datasets.jsonl"

    # 基础模型信息
    base_model_info: dict = {
        "model_id": "ErrenBasicModel",
        "model_name": "DeepSeek-R1-Distill-Qwen-7B",
        "model_path": f"{root_path}/base_model/DeepSeek-R1-Distill-Qwen-7B"
    }

    # 模型类别
    model_type_list: list[str] = [
        "jd",
        "hk"
    ]

    # 规则文件地址
    rules:dict = {
        "jd": {
            "file_name": "jd_rules.jsonl",
            "datasets_path": f"{root_path}/core/datasets/jd_rules.jsonl",
            "model_path": f"{root_path}/core/models/deepseek-r1-JD",
        },
        "hk": {
            "file_name": "hk_rules.jsonl",
            "datasets_path": f"{root_path}/core/datasets/hk_rules.jsonl",
            "model_path": f"{root_path}/core/models/deepseek-r1-HK",
        }
    }

    # 获取训练集数据地址
    datasets_url: dict = {
        "method": "GET",                    # 请求方式
        "baseUrl": "http://127.0.0.1:8080/product-archives/model.getTrainingSetData.erren",   # 训练集地址
        "parameters": {
            "trainingSetID": None,          # 训练集的ID
            "containFile": False            # 是否包含附件信息
        }
    }

    # OCR 模型路径
    ocr_config: dict = {
        "det_path": f"{root_path}/ocrModel/PP-OCRv5_mobile_det_infer",
        "rec_path": f"{root_path}/ocrModel/PP-OCRv5_mobile_rec_infer",
    }

    # OCR优先使用NPU；Paddle Custom Device或算子不可用时自动回退CPU。
    ocr_device: str = "npu:0"
    ocr_fallback_device: str = "cpu"
    ocr_cache_dir: str = f"{root_path}/output/ocr_cache/"
    ocr_max_file_bytes: int = 50 * 1024 * 1024
    ocr_max_pdf_pages: int = 200
    ocr_pdf_text_min_chars: int = 20
    ocr_pdf_dpi: int = 200

    # 输入还需容纳题名、元数据、候选规范条款和答案，正文应先选取高价值片段。
    training_content_max_chars: int = 2200
    # 低于该数量无法建立可信训练/验证集；HK还必须同时覆盖开放和控制。
    training_min_samples: int = 10
    training_min_samples_per_label: int = 2
    training_min_label_types: int = 2

    # 业务系统附件下载地址。附件 url 为相对路径时拼接该地址；可由环境变量覆盖。
    file_download_base_url: str = "http://192.168.10.40:8080"
    file_download_timeout: int = 60

    # 日志文件配置信息
    log_info:dict = {
        "log_name": "app",
        "log_level": "INFO",
        "log_dir": f"{root_path}/logs",
        "log_size": 50,
        "log_retention": 7,
    }


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )



settings = Settings()
