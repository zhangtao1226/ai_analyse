# -*-coding : utf-8 -*-
# @Author   : zhangtao
# @FileName : main.py
# @Desc     :
# @Time     : 2025/10/31 10:33
# @Software : PyCharm
import sys
import os
import socket

# Paddle/OpenBLAS与多进程训练并用时默认单CPU线程；现场可用AI_OMP_NUM_THREADS覆盖。
os.environ["OMP_NUM_THREADS"] = os.environ.get("AI_OMP_NUM_THREADS", "1")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import gc
import re
import json
import httpx
import time
import shutil
import asyncio
import requests
import threading
import multiprocessing
from typing import Optional, Any
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from fastapi import BackgroundTasks
from collections import deque

import torch
import torch_npu
import fitz
from PIL import Image
import torch.multiprocessing as mp
from dotenv import load_dotenv
from peft import PeftModel
from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Request
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from schemas.ResponseModel import BaseResponse, ErrorResponse
from schemas.ReviewRequest import ReviewRequest

from core.config import settings
from core.LoggerDetector import logger
from core.DatasetsUtil import DatasetsUtil
from core.TrainingProgress import TrainingProgressStore
from utils.ResponseUtil import ResponseUtil
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from services.NewLoRATrainModels import LoRATrainModelNPU
from services.ArchiveReviewer import ArchiveReviewer, normalize_keyword_review_signal
from services.ArchiveContentExtractor import ArchiveContentExtractor
from services.PaddleOCRServices import PaddleOCRServices
from services.ReviewKnowledgeService import RuleKnowledgeRetriever
from services.TrainingCallbackService import TrainingCallbackService

load_dotenv(verbose=True)

# 全局进程管理器
process_manager: dict[Any, Any] = {}
_process_manager_lock = threading.Lock()
_preparing_training_ids: set[str] = set()
_reserved_master_ports: set[int] = set()

process_executor = ProcessPoolExecutor(max_workers=1)

_active_tasks: set[str] = set()
_active_tasks_lock = threading.Lock()

_cached_reviewer: Optional[ArchiveReviewer] = None
_cached_model_id: Optional[str] = None
_cached_lora_model_path: Optional[str] = None

_queue: deque[str] = deque()
_queue_lock = threading.Lock()

_content_extractor: Optional[ArchiveContentExtractor] = None
_content_extractor_lock = threading.Lock()

_training_progress_store = TrainingProgressStore(settings.training_progress_dir)
_training_callback_service = TrainingCallbackService(
    callback_url=settings.training_callback_url,
    timeout_seconds=settings.training_callback_timeout_seconds,
    max_attempts=settings.training_callback_max_attempts,
    retry_delay_seconds=settings.training_callback_retry_delay_seconds,
    pending_dir=settings.training_callback_pending_dir,
)


def retry_pending_training_callbacks() -> None:
    succeeded = _training_callback_service.retry_pending()
    if succeeded:
        logger.info(f"本轮成功补发模型训练回调数量: {succeeded}")


def clean_process(data=None):
    """
    清理完成的进行信息
    :param process_manager:
    :return:
    """
    global process_manager
    print(f"process_manager = {process_manager}")

    with _process_manager_lock:
        _cleanup_finished_training_ports_locked()
        process_manager = {
            k: v for k, v in process_manager.items()
            if v.get("process") is not None and v["process"].exitcode is None
        }


def _is_training_process_alive(train_model_info: Optional[dict]) -> bool:
    process = train_model_info.get("process") if train_model_info else None
    return bool(process and process.is_alive())


def _release_training_port(train_model_info: Optional[dict]) -> None:
    if not train_model_info or train_model_info.get("masterPortReleased"):
        return

    master_port = train_model_info.get("masterPort") or train_model_info.get("master_port")
    if master_port is None:
        return

    try:
        _reserved_master_ports.discard(int(master_port))
        train_model_info["masterPortReleased"] = True
    except (TypeError, ValueError):
        logger.warning(f"训练端口格式异常，无法释放: {master_port}")


def _cleanup_finished_training_ports_locked() -> None:
    for train_model_info in process_manager.values():
        process = train_model_info.get("process")
        if process is not None and not process.is_alive():
            _release_training_port(train_model_info)


def _is_master_port_available(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", port))
            return True
    except OSError:
        return False


def _read_env_port(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        logger.warning(f"{name}端口配置无效，使用默认值{default}")
        return default


def _reserve_master_port_locked() -> str:
    preferred_port = _read_env_port("MASTER_PORT", 29500)
    port_start = _read_env_port("LORA_MASTER_PORT_START", min(preferred_port, 29500))
    port_end = _read_env_port("LORA_MASTER_PORT_END", max(preferred_port, 29999))
    port_start = max(1, min(65535, port_start))
    port_end = max(port_start, min(65535, port_end))

    checked_ports = set()
    candidate_ports = [preferred_port, *range(port_start, port_end + 1)]
    for port in candidate_ports:
        if port in checked_ports:
            continue
        checked_ports.add(port)
        if port < 1 or port > 65535 or port in _reserved_master_ports:
            continue
        if _is_master_port_available(port):
            _reserved_master_ports.add(port)
            return str(port)

    raise RuntimeError(f"未找到可用的分布式训练端口，已检查范围: {port_start}-{port_end}")


def _release_reserved_master_port(master_port: Optional[str]) -> None:
    if master_port is None:
        return
    try:
        _reserved_master_ports.discard(int(master_port))
    except (TypeError, ValueError):
        logger.warning(f"训练端口格式异常，无法释放: {master_port}")


def _abort_training_start(model_id: str, master_port: Optional[str]) -> None:
    with _process_manager_lock:
        _preparing_training_ids.discard(model_id)
        _release_reserved_master_port(master_port)


async def _training_start_failure_response(model_id: str, message: str):
    """训练进程创建前失败时也通知业务系统，避免客户端状态停留在训练中。"""
    logger.error(f"模型训练启动失败: modelID={model_id}, reason={message}")
    try:
        await asyncio.to_thread(_training_callback_service.notify, model_id, "异常")
    except Exception as callback_error:
        # 回调服务已将失败事件持久化，定时任务会继续补发。
        logger.error(
            f"模型训练启动失败回调最终失败: modelID={model_id}, error={callback_error}"
        )
    return ResponseUtil.error(message=message)


def _parse_training_int(value, default: int, min_value: int, max_value: int, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        logger.warning(f"训练参数{name}无效: {value}, 使用默认值{default}")
        return default

    if parsed < min_value or parsed > max_value:
        logger.warning(f"训练参数{name}超出范围: {parsed}, 使用默认值{default}")
        return default
    return parsed


def _parse_training_float(value, default: float, min_value: float, max_value: float, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        logger.warning(f"训练参数{name}无效: {value}, 使用默认值{default}")
        return default

    if parsed < min_value or parsed > max_value:
        logger.warning(f"训练参数{name}超出范围: {parsed}, 使用默认值{default}")
        return default
    return parsed


def _count_jsonl_samples(dataset_path: str) -> int:
    if not dataset_path or not os.path.exists(dataset_path):
        return 0
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception as e:
        logger.warning(f"统计训练样本数失败: {dataset_path}, {e}")
        return 0


def _resolve_train_world_size(requested_world_size: int, sample_count: int) -> int:
    try:
        npu_count = torch_npu.npu.device_count()
    except Exception:
        npu_count = requested_world_size

    resolved = max(1, min(requested_world_size, npu_count if npu_count else requested_world_size))
    if sample_count > 0 and sample_count < 8:
        logger.warning(f"训练样本数较少({sample_count}条)，自动使用单卡训练，避免多rank稀释样本")
        return 1
    if sample_count > 0:
        resolved = min(resolved, sample_count)
    return max(1, resolved)


def _is_npu_out_of_memory(error: BaseException) -> bool:
    """识别子进程透传到主训练进程的NPU显存异常。"""
    message = str(error).lower()
    return "npu out of memory" in message or "acl_error_rt_memory_allocation" in message


def _build_oom_retry_configs(config: dict) -> list[dict]:
    """生成由保真到节省显存的训练配置，重复配置只保留一次。"""
    attempts = []
    seen = set()

    def append_attempt(candidate: dict) -> None:
        signature = (
            int(candidate["batch_size"]),
            int(candidate["gradient_accumulation_steps"]),
            int(candidate["max_length"]),
        )
        if signature not in seen:
            seen.add(signature)
            attempts.append(candidate.copy())

    current = config.copy()
    append_attempt(current)

    # 首先降低单卡瞬时批量，并用梯度累积尽量保持原有效批量。
    if int(current["batch_size"]) > 1:
        original_batch_size = int(current["batch_size"])
        current["batch_size"] = 1
        current["gradient_accumulation_steps"] = min(
            32,
            int(current["gradient_accumulation_steps"]) * original_batch_size,
        )
        append_attempt(current)

    # batch_size已经无法继续降低时，才逐级收紧最大上下文长度。
    for length_limit in (2048, 1536, 1024):
        if int(current["max_length"]) > length_limit:
            current["max_length"] = length_limit
            append_attempt(current)

    return attempts


# 定义 lifespan 事件处理器
async def lifespan(app: FastAPI):
    # 启动时创建并启动调度器
    scheduler = AsyncIOScheduler(timezone='Asia/Shanghai')
    # 每天凌晨2点清空temp文件夹
    # scheduler.add_job(clean_dir, "cron", hour=2, args=[f"{BASE_DIR}/temp"])
    # 每 40 分钟清空文件夹
    scheduler.add_job(clean_process, "interval", minutes=20, args=[process_manager])
    scheduler.add_job(
        retry_pending_training_callbacks,
        "interval",
        seconds=settings.training_callback_retry_interval_seconds,
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()

    # 将调度器存储在app.state中
    app.state.scheduler = scheduler

    yield  # 程序运行期间

    # 关闭时停止调度器
    scheduler.shutdown()


app = FastAPI(lifespan=lifespan)


@app.get("/getModelInfo")
async def get_model_info(modelId: str = None):
    if modelId is None or modelId == "":
        return ResponseUtil.error(message=f"缺少参数")

    try:
        with open(settings.models_info_table, 'r', encoding='utf-8') as file:
            model_list = json.load(file)

        print(model_list)

        if modelId not in model_list.keys():
            return ResponseUtil.error(message=f"modelID 不存在！{modelId}")

        model_info = model_list[modelId]
        return ResponseUtil.success(data=model_info)
    except Exception as e:
        return ResponseUtil.error(message=str(e))


@app.get("/getAvailableModels")
async def get_models(modelType: Optional[str] = None):
    """
    模型列表接口
    :return:
    """
    # 模型信息文件路径
    model_info_path = settings.models_info_table

    try:
        with open(model_info_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            file.close()

        models_list = []
        for key, value in data.items():
            if value["model_type"] == modelType:
                models_list.append(value)
        result_data = dict()
        result_data["models"] = models_list
        result_data["models_count"] = len(models_list)
        return ResponseUtil.success(data=result_data)
    except Exception as e:
        return ResponseUtil.error(message=str(e))


@app.delete("/modelDelete/")
async def delete_model(model_id: Optional[str] = None):
    """
    删除模型
    :param model_id:
    :return:
    """
    if model_id is None or model_id == "":
        return ResponseUtil.error(message=f"缺少参数！model_id")

    try:
        with open(settings.models_info_table, 'r', encoding='utf-8') as file:
            model_list = json.load(file)
        model_info = model_list[model_id]

        shutil.rmtree(model_info["model_path"])

        new_model_list = dict()
        for key, value in model_list.items():
            if key != model_id:
                new_model_list[key] = value

        with open(settings.models_info_table, 'w', encoding='utf-8') as file:
            json.dump(new_model_list, file, ensure_ascii=False, indent=4)

        return ResponseUtil.success(message=f"模型删除成功！")
    except Exception as e:
        return ResponseUtil.error(message=f"模型删除失败; modelID={model_id}, {str(e)}")


@app.post("/startTraining")
async def train_models(request: Request):
    """
    模型训练
    :param modelId:         模型ID
    :param baseModelID      基本模型ID
    :param trainingSetIDs   数据集
    :param modelType:       模型类型
    :param learningRate     学习率
    :param learningRounds   训练轮数
    :return:
    """
    json_data = await request.json()
    print(f"接收训练参数: {json_data}")

    modelID = json_data["modelID"]
    basicModelID = json_data["baseModelID"]
    trainingSetIDs = json_data["trainingSetIDs"]
    modelType = json_data["modelType"]
    learningRate = json_data.get("learningRate", 1e-4)
    learningRounds = json_data.get("learningRounds", 4)
    training_options = {
        "learning_rate": _parse_training_float(learningRate, 1e-4, 1e-6, 1e-3, "learningRate"),
        "num_epochs": _parse_training_int(learningRounds, 4, 1, 30, "learningRounds"),
        "batch_size": _parse_training_int(json_data.get("batchSize", 1), 1, 1, 16, "batchSize"),
        "gradient_accumulation_steps": _parse_training_int(
            json_data.get("gradientAccumulationSteps", 2),
            2,
            1,
            32,
            "gradientAccumulationSteps"
        ),
        "max_length": _parse_training_int(json_data.get("maxLength", 2048), 2048, 512, 4096, "maxLength"),
        "lora_r": _parse_training_int(json_data.get("loraR", 8), 8, 2, 64, "loraR"),
        "lora_alpha": _parse_training_int(json_data.get("loraAlpha", 16), 16, 4, 128, "loraAlpha"),
        "validation_ratio": _parse_training_float(
            json_data.get("validationRatio", 0.15), 0.15, 0.1, 0.3, "validationRatio"
        ),
        "requested_world_size": _parse_training_int(
            json_data.get("worldSize", os.environ.get("LORA_WORLD_SIZE", 2)),
            2,
            1,
            8,
            "worldSize"
        ),
    }

    if modelID is None or modelID == "":
        return ResponseUtil.error(message="缺少模型ID！")

    if modelType is None or modelType == "":
        return ResponseUtil.error(message="缺少模型类别参数")

    print(f"modelID:{modelID}, basicModelID:{basicModelID}, trainingSetIDs:{trainingSetIDs}, "
          f"modelType:{modelType}, learningRate:{learningRate}, learningRounds:{learningRounds}")

    if modelType not in settings.model_type_list:
        return ResponseUtil.error(message=f"模型类别不存在; modelType={modelType}")

    try:
        with _process_manager_lock:
            _cleanup_finished_training_ports_locked()
            train_model_info = process_manager.get(modelID)
            if modelID in _preparing_training_ids or _is_training_process_alive(train_model_info):
                return ResponseUtil.error(message=f"模型正在训练中，请勿重复启动; modelID={modelID}")
            master_port = _reserve_master_port_locked()
            _preparing_training_ids.add(modelID)
    except RuntimeError as e:
        return await _training_start_failure_response(modelID, str(e))

    # 请求一旦被当前节点接收，立即重置进度，避免同一modelID重训启动失败时仍显示旧的100%。
    try:
        _training_progress_store.reset(modelID)
    except Exception as e:
        _abort_training_start(modelID, master_port)
        logger.error(f"初始化训练进度失败; modelID={modelID}, error={e}")
        return await _training_start_failure_response(modelID, f"初始化训练进度失败: {str(e)}")

    # 加载已保存的模型信息文件
    try:
        if os.path.exists(settings.models_info_table):
            with open(settings.models_info_table, 'r', encoding='utf-8') as file:
                model_list = json.load(file)
                file.close()
        else:
            model_list = dict()
    except Exception as e:
        _abort_training_start(modelID, master_port)
        logger.error(f"加载模型信息失败; {str(e)}")
        return await _training_start_failure_response(modelID, "加载模型信息失败")

    # 第一阶段：完整整理训练集。Base64解码、文本提取/OCR和JSONL校验均在启动LoRA前完成。
    datasets_path = None
    try:
        logger.info(f"开始整理训练集; modelID={modelID}, modelType={modelType}")
        datasets_util = DatasetsUtil(trainingSetIDs=trainingSetIDs, modelType=modelType)
        datasets_path = datasets_util.create_train_data(modelID=modelID, modelType=modelType)
        sample_count = _count_jsonl_samples(datasets_path)
        if sample_count <= 0:
            raise ValueError("整理后的训练数据集没有有效样本")
        logger.info(
            f"训练集整理完成; modelID={modelID}, samples={sample_count}, path={datasets_path}"
        )
    except Exception as e:
        _abort_training_start(modelID, master_port)
        logger.error(f"数据集获取失败; {str(e)}; datasets_path={datasets_path}")
        return await _training_start_failure_response(modelID, f"训练集整理失败: {str(e)}")

    # 测试数据集(临时)
    # datasets_path = settings.train_dataset_path

    if modelID not in model_list.keys():
        print("创建新模型")
        logger.info(f"创建新模型; modelID={modelID}, basicModelID={basicModelID}")
        if basicModelID:
            if basicModelID not in model_list:
                _abort_training_start(modelID, master_port)
                return await _training_start_failure_response(
                    modelID, f"基础模型ID不存在; basicModelID={basicModelID}"
                )
            previous_model_path = model_list[basicModelID]['model_path']
            enable_continue_training = True
        else:
            previous_model_path = None
            enable_continue_training = False
    else:
        print("继续训练模型")
        logger.info(f"继续训练模型; modelID={modelID}, basicModelID={basicModelID}")
        if basicModelID == "":
            previous_model_path = model_list[modelID]['model_path']
        else:
            if basicModelID not in model_list:
                _abort_training_start(modelID, master_port)
                return await _training_start_failure_response(
                    modelID, f"基础模型ID不存在; basicModelID={basicModelID}"
                )
            previous_model_path = model_list[basicModelID]['model_path']
        enable_continue_training = True

    training_options["master_port"] = master_port
    logger.info(f"训练分布式通信端口: MASTER_PORT={master_port}, modelID={modelID}")

    # 第二阶段：只有最终JSONL生成并通过有效样本校验后，才创建LoRA训练进程。
    process = multiprocessing.Process(
        target=start_train,
        args=(modelID, modelType, datasets_path, enable_continue_training, previous_model_path, training_options)
    )
    try:
        process.start()
    except Exception as e:
        _abort_training_start(modelID, master_port)
        logger.error(f"启动训练进程失败; {str(e)}")
        return await _training_start_failure_response(modelID, f"启动训练进程失败: {str(e)}")

    with _process_manager_lock:
        _preparing_training_ids.discard(modelID)
        process_manager[modelID] = {
            "model_id": modelID,
            "base_model_id": basicModelID,
            "trainingSetIDs": trainingSetIDs,
            "learningRate": training_options["learning_rate"],
            "learningRounds": training_options["num_epochs"],
            "batchSize": training_options["batch_size"],
            "gradientAccumulationSteps": training_options["gradient_accumulation_steps"],
            "maxLength": training_options["max_length"],
            "masterPort": int(master_port),
            "masterPortReleased": False,
            "process_id": process.pid,
            "process": process,
            "status": process.is_alive()
        }
    return ResponseUtil.success(message=f"模型已经开始训练; modelID={modelID}, modelType={modelType}")


def run_training(rank, world_size, config):
    """单进程训练逻辑 - 适配动态数据均分模式"""

    # 读取json中模型信息
    if os.path.exists(settings.models_info_table):
        try:
            with open(settings.models_info_table, 'r', encoding='utf-8') as file:
                model_table = json.load(file)
        except Exception as e:
            logger.error(f"读取模型JSON文件中失败: {str(e)}")
            raise RuntimeError(f"读取模型JSON文件失败: {e}") from e
    else:
        model_table = {}

    # 解析配置参数
    model_id = config["model_id"]
    model_type = config["model_type"]
    base_model_path = config["base_model_path"]
    output_dir = config["output_dir"]
    previous_model_path = config["previous_model_path"]
    datasets_path = config["datasets_path"]
    enable_continue_training = config["enable_continue_training"]
    num_epochs = config.get("num_epochs", 4)
    learning_rate = config.get("learning_rate", 1e-4)
    max_length = config.get("max_length", 2048)
    batch_size = config.get("batch_size", 1)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 2)
    master_port = config.get("master_port", os.environ.get("MASTER_PORT", "29500"))

    # 仅主进程创建输出目录
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    # 初始化训练器（NPU设备ID与rank绑定，实现一一对应）
    lora_r = config.get("lora_r", 8)
    lora_alpha = config.get("lora_alpha", 16)
    validation_ratio = config.get("validation_ratio", 0.15)
    min_samples = config.get("min_samples", settings.training_min_samples)
    progress_callback = None
    if rank == 0:
        progress_callback = lambda percentage: _training_progress_store.update(model_id, percentage)
    trainer = None
    try:
        trainer = LoRATrainModelNPU(
            base_model_path=base_model_path,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            npu_device_id=rank,
            seed=42,
            world_size=world_size,
            rank=rank,
            master_port=master_port
        )
        # 简化启动日志
        if rank == 0 and enable_continue_training:
            logger.info("=" * 60)
            logger.info(f"启动{world_size}卡NPU LoRA增量训练（动态数据均分模式）")
            logger.info("=" * 60)
        elif rank == 0:
            logger.info("=" * 60)
            logger.info(f"启动{world_size}卡NPU LoRA训练")
            logger.info("=" * 60)

        # 执行纯手动训练
        result = trainer.train_manual_npu(
            dataset_path=datasets_path,
            output_dir=output_dir,
            num_epochs=num_epochs,
            max_samples=None,
            continue_training=enable_continue_training,
            previous_model_path=previous_model_path,
            max_length=max_length,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation_steps,
            validation_ratio=validation_ratio,
            min_samples=min_samples,
            progress_callback=progress_callback,
        )

        if rank == 0:
            # 保存模型信息
            model_info = {
                "model_id": model_id,
                "model_name": model_id,
                "model_path": output_dir,
                "model_type": model_type,
                "model_status": True,
                "model_epochs": num_epochs,
                "model_rate": learning_rate,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "max_length": max_length,
                "world_size": world_size,
                "datasets_path": datasets_path,
                "sample_count": config.get("sample_count", 0),
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "validation_ratio": validation_ratio,
                "best_epoch": result.get("best_epoch") if isinstance(result, dict) else None,
                "best_validation_loss": result.get("best_validation_loss") if isinstance(result, dict) else None,
                "train_sample_count": result.get("train_sample_count") if isinstance(result, dict) else None,
                "validation_sample_count": result.get("validation_sample_count") if isinstance(result, dict) else None,
                "F1": None,
                "accuracy_rate": None,
                "datasets_name": ""
            }
            logger.info(f'当前模型信息: {model_info}')

            # 模型信息写入json中
            target_path = settings.models_info_table

            try:
                dir_path = os.path.dirname(target_path)
                if not os.path.exists(target_path):
                    os.makedirs(dir_path, exist_ok=True)

                # 保存模型信息
                model_table[model_id] = model_info

                with open(target_path, 'w', encoding='utf-8') as f:
                    json.dump(model_table, f, ensure_ascii=False, indent=2)
                    logger.info(f"模型信息已成功写入: {target_path}")
            except Exception as e:
                logger.error(f"模型信息保存到JSON文件中失败: {str(e)}")
                raise RuntimeError(f"模型信息保存失败: {e}") from e

    except Exception as e:
        logger.error(f"执行失败（NPU:{rank}）: {e}", exc_info=True)
        raise
    finally:
        if trainer is not None:
            trainer.cleanup()


def start_train(
        model_id,
        model_type,
        datasets_path,
        enable_continue_training=False,
        previous_model_path=None,
        training_options=None
):
    """
    训练模型
    :param model_id:
    :param model_type:
    :param base_model_info:
    :param train_model:
    :return:
    """
    base_model_path = settings.base_model_info['model_path']
    output_dir = f"{settings.save_model_path}Deepseek-R1-{model_id}_{int(time.time())}"
    training_options = training_options or {}
    sample_count = _count_jsonl_samples(datasets_path)
    requested_world_size = training_options.get("requested_world_size", 2)
    world_size = _resolve_train_world_size(requested_world_size, sample_count)

    config = {
        "model_id": model_id,
        "model_type": model_type,
        "base_model_path": base_model_path,
        "output_dir": output_dir,
        "previous_model_path": previous_model_path,
        "datasets_path": datasets_path,
        "enable_continue_training": enable_continue_training,
        "batch_size": training_options.get("batch_size", 1),
        "gradient_accumulation_steps": training_options.get("gradient_accumulation_steps", 2),
        "max_length": training_options.get("max_length", 2048),
        "learning_rate": training_options.get("learning_rate", 1e-4),
        "num_epochs": training_options.get("num_epochs", 4),
        "lora_r": training_options.get("lora_r", 8),
        "lora_alpha": training_options.get("lora_alpha", 16),
        "validation_ratio": training_options.get("validation_ratio", 0.15),
        "min_samples": int(getattr(settings, "training_min_samples", 10)),
        "sample_count": sample_count,
        "world_size": world_size,
        "master_port": str(training_options.get("master_port", os.environ.get("MASTER_PORT", "29500"))),
    }

    print(f"训练模型配置信息: {config}")
    logger.info(f"训练模型配置信息: {config}")

    training_error = None
    try:
        min_samples = max(2, int(config["min_samples"]))
        if sample_count < min_samples:
            raise RuntimeError(
                f"有效训练样本仅{sample_count}条，至少需要{min_samples}条，已在加载NPU模型前终止"
            )
        if enable_continue_training and (not previous_model_path or not os.path.exists(previous_model_path)):
            raise RuntimeError(f"增量训练所需的已训练模型路径不存在: {previous_model_path}")
        if not os.path.exists(base_model_path):
            raise RuntimeError(f"基础模型路径不存在: {base_model_path}")
        if not os.path.exists(config["datasets_path"]):
            raise RuntimeError(f"数据集路径不存在: {config['datasets_path']}")

        _training_progress_store.update(model_id, 1.0)
        attempt_configs = _build_oom_retry_configs(config)
        for attempt_index, attempt_config in enumerate(attempt_configs, 1):
            logger.info(
                f"开始训练尝试 {attempt_index}/{len(attempt_configs)}: "
                f"modelID={model_id}, batchSize={attempt_config['batch_size']}, "
                f"gradientAccumulationSteps={attempt_config['gradient_accumulation_steps']}, "
                f"maxLength={attempt_config['max_length']}"
            )
            try:
                mp.spawn(
                    run_training,
                    args=(world_size, attempt_config),
                    nprocs=world_size,
                    join=True,
                    daemon=False
                )
                config = attempt_config
                break
            except Exception as attempt_error:
                is_last_attempt = attempt_index == len(attempt_configs)
                if not _is_npu_out_of_memory(attempt_error) or is_last_attempt:
                    raise
                next_config = attempt_configs[attempt_index]
                logger.warning(
                    f"检测到NPU显存溢出，将完整重启训练任务: modelID={model_id}, "
                    f"下一次batchSize={next_config['batch_size']}, "
                    f"gradientAccumulationSteps={next_config['gradient_accumulation_steps']}, "
                    f"maxLength={next_config['max_length']}"
                )
    except Exception as error:
        training_error = error
        logger.error(f"模型训练任务异常结束: modelID={model_id}, error={error}", exc_info=True)

    training_result = "异常" if training_error else "成功"
    try:
        updateTrainingState(model_id, training_result)
    except Exception as callback_error:
        # 回调失败不能改变模型产物是否训练成功的事实；错误会完整记录，便于业务侧补偿。
        logger.error(
            f"模型训练完成回调最终失败: modelID={model_id}, "
            f"result={training_result}, error={callback_error}"
        )

    if training_error is None:
        _training_progress_store.update(model_id, 100.0)
        logger.info(f"模型训练任务完成: modelID={model_id}, completedPercentage=100.0")
        return

    raise training_error


def updateTrainingState(model_id, result):
    """
    推送模型训练结果
    :param model_id:
    :param result:
    :return:
    """
    _training_callback_service.notify(model_id=model_id, result=result)

@app.get("/trainingProcess/{modelID}")
async def get_model_state(modelID: str):
    """返回模型训练已完成的百分比。"""
    train_model_info = process_manager.get(modelID)
    if train_model_info:
        process = train_model_info.get("process")
        is_running = bool(process and process.is_alive())
        train_model_info["status"] = is_running
        if not is_running:
            with _process_manager_lock:
                _release_training_port(train_model_info)

    completed_percentage = _training_progress_store.read(modelID)
    if completed_percentage is None:
        if not train_model_info:
            return ResponseUtil.error(message=f"模型ID不存在; modelId={modelID}")
        completed_percentage = 0.0

    return ResponseUtil.success(data={
        "completedPercentage": completed_percentage,
    })


@app.post("/interruptTraining")
async def interrupt_training(modelId: str = Form(...)):
    """
    中断训练
    :param modelId:
    :return:
    """
    if modelId not in process_manager.keys():
        return ResponseUtil.error(message=f"模型ID不存在; modelId={modelId}")
    else:
        print(process_manager[modelId])
        try:
            train_model_info = process_manager[modelId]
            process = train_model_info["process"]
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
            with _process_manager_lock:
                _release_training_port(train_model_info)
                del process_manager[modelId]
            try:
                await asyncio.to_thread(updateTrainingState, modelId, "异常")
            except Exception as callback_error:
                logger.error(
                    f"模型训练中断回调最终失败: modelID={modelId}, error={callback_error}"
                )
            return ResponseUtil.success(message=f"模型训练已经中断, modelID={modelId}")
        except Exception as e:
            return ResponseUtil.error(message=f"中断训练失败: {str(e)}, modelID={modelId}")


def _add_task(aiAuditId: str) ->bool:
    with _active_tasks_lock:
        if aiAuditId in _active_tasks:
            return False
        _active_tasks.add(aiAuditId)
        return True


def _remove_task(aiAuditId: str) ->bool:
    with _active_tasks_lock:
        _active_tasks.remove(aiAuditId)

def _enqueue(aiAuditId: str) -> int:
    with _queue_lock:
        _queue.append(aiAuditId)
        pos = len(_queue)
        return pos

def _dequeue(aiAuditId: str):
    with _queue_lock:
        try:
            _queue.remove(aiAuditId)
        except ValueError:
            pass

def _queue_position(aiAuditId: str) -> int:
    with _queue_lock:
        try:
            return list(_queue).index(aiAuditId) + 1
        except ValueError:
            return 0

@app.post("/review")
async def review(review_request: ReviewRequest, background_tasks: BackgroundTasks):
    """
    ai审核
    :param modelId:
    :param data:
    :return:
    """
    logger.info("开始 AI 进行审核·······")
    modelId = review_request.model_id
    modelType = review_request.model_type
    data = review_request.data
    aiAuditId = review_request.ai_audit_id

    logger.info(
        f"接收到审核请求: aiAuditId={aiAuditId}, modelId={modelId}, "
        f"modelType={modelType}, records={len(data)}, "
        f"keywordRecords={sum(1 for item in data if item.keywords)}, "
        f"attachments={sum(len(item.files) for item in data)}"
    )

    callback_url = settings.review_callback_url


    try:
        logger.info("获取模型信息······")
        with open(settings.models_info_table, 'r', encoding='utf-8') as file:
            model_table = json.load(file)
    except Exception as e:
        logger.error(f"获取模型信息失败; {str(e)}")
        return ResponseUtil.error(message="获取模型信息失败")

    if modelId not in model_table.keys():
        return ResponseUtil.error(message="模型不存在！")

    model_info = model_table[modelId]

    if modelType != model_info["model_type"]:
        logger.info(f"模型类型与模型不匹配; AI审核类型:{modelType}; 模型类型: {model_info['model_type']}")
        return ResponseUtil.error(message="模型类型与模型不匹配！")

    instructions = []
    for item in data:
        item_dict = {
            "arid": item.arid,
            "title": item.title,
            "date_time": item.date_time,
            # 附件正文在后台进程中解析，避免请求接口被下载/OCR阻塞。
            "files": item.files,
            "content": item.content,
            "archive_category": item.archive_category,
            "organization_problem": item.organization_problem,
            "fonds_no": item.fonds_no,
            "document_no": item.document_no,
            "responsible_org": item.responsible_org,
            "retention_period": item.retention_period,
            "archive_year": item.archive_year,
            "keywords": item.keywords,
            "audit_result": item.audit_result,
        }
        instructions.append(item_dict)

    if not _add_task(aiAuditId):
        logger.warning(f"[任务 {aiAuditId}] 重复提交, 当前任务仍在审核中······")
        return ResponseUtil.error(message=f"[任务 {aiAuditId}] 重复提交, 当前任务仍在审核中······")

    pos = _enqueue(aiAuditId)

    background_tasks.add_task(
        submit_review_task,
        model_id=modelId,
        base_model_path=settings.base_model_info["model_path"],
        lora_model_path=model_info["model_path"],
        model_type=modelType,
        instructions=instructions,
        aiAuditId=aiAuditId,
        callback_url=callback_url,
    )

    logger.info(f"[任务 {aiAuditId}] 已提交后台，立即返回")
    return ResponseUtil.success(data={
        "code": 200,
        "message": "AI审核中",
        "aiAuditId": aiAuditId,
        "queuePosition": pos,
    })

def _load_or_reuse_reviewer(
    model_id: str,
    base_model_path: str,
    lora_model_path: str,
    model_type: str,
) -> ArchiveReviewer:
    global _cached_reviewer, _cached_model_id, _cached_lora_model_path

    if (
            _cached_reviewer is not None
            and _cached_model_id == model_id
            and _cached_lora_model_path == lora_model_path
    ):
        logger.info(f"[Worker] 命中模型缓存，复用 modelId={model_id}, path={lora_model_path}")
        return _cached_reviewer

    if _cached_reviewer is not None:
        logger.info(
            f"[Worker] 模型切换: {_cached_model_id}/{_cached_lora_model_path} "
            f"→ {model_id}/{lora_model_path}，释放旧模型"
        )
        try:
            _cached_reviewer.cleanup()
        except Exception as e:
            logger.warning(f"[Worker] 旧模型释放异常（忽略）: {e}")
        del _cached_reviewer
        torch_npu.npu.empty_cache()

    # 加载新模型
    logger.info(f"[Worker] 加载新模型 modelId={model_id}")
    reviewer = ArchiveReviewer(
        base_model_path=base_model_path,
        lora_model_path=lora_model_path,
        batch_size=max(1, int(os.environ.get("REVIEW_BATCH_SIZE", "1"))),
        model_type=model_type,
        npu_device_ids=[0],
    )
    reviewer.load_model()

    # 写入缓存
    _cached_reviewer = reviewer
    _cached_model_id = model_id
    _cached_lora_model_path = lora_model_path

    return _cached_reviewer

async def submit_review_task(
    model_id: str,
    base_model_path: str,
    lora_model_path: str,
    model_type: str,
    instructions: list,
    aiAuditId: str,
    callback_url: str,
):

    pos = _queue_position(aiAuditId)

    logger.info(
        f"[任务 {aiAuditId}] 进入排队，当前位置: 第 {pos -1} 位，"
        f"前方等待任务数: {pos - 2}"
    )

    loop = asyncio.get_event_loop()
    start_wait_time = time.time()

    try:
        await loop.run_in_executor(
            process_executor,
            run_review_in_process,
            model_id,
            base_model_path,
            lora_model_path,
            model_type,
            instructions,
            aiAuditId,
            callback_url,
        )
    except Exception as e:
        logger.error(f"[任务 {aiAuditId}] 进程执行异常: {str(e)}")
    finally:
        wait_seconds = time.time() - start_wait_time
        _dequeue(aiAuditId)
        _remove_task(aiAuditId)

        # 出队后打印剩余队列状态
        with _queue_lock:
            remaining = list(_queue)

        logger.info(
            f"[任务 {aiAuditId}] 执行完毕，"
            f"总耗时: {wait_seconds:.1f}s，"
            f"剩余队列长度: {len(remaining)}"
        )
        if remaining:
            logger.info(f"[队列状态] 剩余任务: {remaining}")

        logger.info(f"[任务 {aiAuditId}] 已从审核队列移除")

def run_review_in_process(
        model_id: str,
    base_model_path: str,
    lora_model_path: str,
    model_type: str,
    instructions: list,
    aiAuditId: str,
    callback_url: str,
):
    result_payload = {
        "aiAuditId": aiAuditId,
        "code": 200,
        "message": "审核完成",
        "data": []
    }

    try:
        # 缓存复用 or 切换加载
        reviewer = _load_or_reuse_reviewer(
            model_id=model_id,
            base_model_path=base_model_path,
            lora_model_path=lora_model_path,
            model_type=model_type,
        )

        prepared_instructions = prepare_archive_contents(instructions, model_type)
        review_results = reviewer.batch_review(prepared_instructions)
        logger.info(f"[{aiAuditId}] 审核完成，结果条数: {len(review_results)}")
        result_payload["data"] = review_results

    except Exception as e:
        import traceback
        traceback.print_exc()
        result_payload["code"] = 500
        result_payload["message"] = f"审核失败: {str(e)}"
        logger.error(f"[{aiAuditId}] 审核失败: {e}")

    finally:
        _notify_review_callback(callback_url, result_payload)


def _notify_review_callback(callback_url: str, result_payload: dict) -> bool:
    """发送审核结果；网络、HTTP和明确的业务失败均按指数退避重试。"""
    ai_audit_id = str(result_payload.get("aiAuditId") or "")
    max_attempts = max(1, int(settings.review_callback_max_attempts))
    retry_delay = max(0.0, float(settings.review_callback_retry_delay_seconds))
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                callback_url,
                data=json.dumps(result_payload, ensure_ascii=False),
                headers={"Content-Type": "application/json; charset=utf-8"},
                timeout=float(settings.review_callback_timeout_seconds),
            )
            if not 200 <= response.status_code < 300:
                raise RuntimeError(f"HTTP状态异常: {response.status_code}")
            try:
                response_payload = response.json()
            except Exception:
                response_payload = None
            if isinstance(response_payload, dict) and "code" in response_payload:
                business_code = str(response_payload["code"])
                if business_code not in {"0", "200"}:
                    raise RuntimeError(f"业务状态异常: code={business_code}")
            logger.info(
                f"[{ai_audit_id}] 审核回调成功: "
                f"attempt={attempt}/{max_attempts}, status={response.status_code}"
            )
            return True
        except Exception as error:
            last_error = error
            logger.warning(
                f"[{ai_audit_id}] 审核回调失败: "
                f"attempt={attempt}/{max_attempts}, error={error}"
            )
            if attempt < max_attempts and retry_delay > 0:
                time.sleep(retry_delay * (2 ** (attempt - 1)))

    logger.error(f"[{ai_audit_id}] 审核回调最终失败: {last_error}")
    return False


def _get_content_extractor() -> ArchiveContentExtractor:
    global _content_extractor
    if _content_extractor is None:
        with _content_extractor_lock:
            if _content_extractor is None:
                _content_extractor = ArchiveContentExtractor.from_settings(settings)
    return _content_extractor


def _attachment_download_url(url: str) -> str:
    return _get_content_extractor().resolve_download_url(url)


def _download_attachment(url: str) -> tuple[bytes, str]:
    payload, content_type, _ = _get_content_extractor().download_attachment(url)
    return payload, content_type


def _ocr_attachment(data: bytes, content_type: str, source_url: str) -> str:
    """兼容旧调用：优先提取PDF文本层，无文本页和图片再执行OCR。"""
    text, _ = _get_content_extractor().extract_binary(data, content_type, source_url)
    return text


def extract_archive_content(archive_item: dict) -> str:
    """按正文和附件顺序提取内容，供在线审核使用。"""
    return _get_content_extractor().extract_archive_content(archive_item)


def prepare_archive_contents(instructions: list, model_type: str) -> list:
    rule_paths = {
        model_type: settings.rules[model_type]["datasets_path"]
    }
    retriever = RuleKnowledgeRetriever(
        rule_paths=rule_paths,
        top_k=int(os.environ.get("REVIEW_RAG_TOP_K", "3")),
    )
    prepared = []
    for item in instructions:
        enriched = dict(item)
        enriched["content"] = extract_archive_content(enriched)
        enriched["retrieved_rules"] = retriever.search(model_type, enriched)
        logger.info(
            f"档案{enriched.get('arid', '')}正文准备完成，字符数={len(enriched['content'])}，"
            f"检索条款数={len(enriched['retrieved_rules'])}"
        )
        prepared.append(enriched)
    return prepared

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    """
    OCR 识别
    :param file:
    :return:
    """
    name, extension = os.path.splitext(file.filename)
    lower_extension = extension.lower()

    if lower_extension == ".pdf":
        try:
            # 保存PDF文件
            pdf_path = f"temp/temp_{int(time.time())}.pdf"
            with open(pdf_path, "wb") as f:
                f.write(await file.read())

            if not os.path.exists(pdf_path):
                return HTTPException(status_code=500, detail=f"文件接收失败")

            images_list = pdf_process(pdf_path)

            ocr_result = ocr_process(images_list)

            # 删除临时文件
            os.remove(pdf_path)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OCR识别PDF过程报错: {str(e)}")
    elif file.content_type.startswith("image/"):
        print(f"上传图片")
        try:
            # 读取上传的图片数据
            image = await file.read()
            # 将二进制数据转换为PIL图像对象
            image_obj = Image.open(io.BytesIO(image))
            image_path = f"temp/temp_image_{int(time.time())}_1.png"
            image_obj.save(image_path)

            # 执行OCR识别
            ocr_result = ocr_process([image_path])

            os.remove(image_path)

        except Exception as e:
            # 处理可能出现的异常
            raise HTTPException(status_code=500, detail=f"OCR 识别报错: {str(e)}")
    logger.info(f"OCR识别结果: {{'results': ocr_result}}")
    return {"results": ocr_result}


def pdf_process(pdf_path):
    """
    pdf文件处理
    :param pdf_path:
    :return:
    """
    image_list = []
    try:
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)

        for page_num in range(total_pages):
            try:
                page = pdf_document.load_page(page_num)

                # 设置缩放比例以控制 DPI
                zoom = 200 / 72
                matrix = fitz.Matrix(zoom, zoom)

                # 渲染页面为像素图
                pix = page.get_pixmap(matrix=matrix)

                # 创建输出文件名
                output_filename = f"{pdf_path.split('.')[0]}_page_{page_num + 1}.png"

                # 保存图片
                pix.save(output_filename)

                image_list.append(output_filename)

                print(f"转换成功: {pdf_path} 第 {page_num + 1} 页 -> {output_filename}")

            except Exception as e:
                print(f"转换 PDF 页失败: {pdf_path} 第 {page_num + 1} 页 - {str(e)}")

        pdf_document.close()

    except Exception as e:
        print(f"转换 PDF 失败: {pdf_path} - {str(e)}")

    return image_list


def ocr_process(image_list):
    """
    OCR 初始化并识别
    :param image_list:
    :return:
    """
    # 初始化OCR
    rec_path = settings.ocr_config["rec_path"]
    det_path = settings.ocr_config["det_path"]
    ocr_detector = PaddleOCRServices(
        det_path=det_path,
        rec_path=rec_path,
        device=settings.ocr_device,
        fallback_device=settings.ocr_fallback_device,
    )

    texts = []

    for image_path in image_list:
        text = ocr_detector.ocr_predict(image_path)

        page = image_path.split("_")[3].split('.')[0]
        texts.append({"page": page, "text": text})

        logger.info(f"OCR识别结果: 第 {page} 完成; ")

    return ResponseUtil.success(data=texts)


@app.get("/")
async def root():
    return {"message": "档案规范审核接口系统", "status": "running"}


@app.get("/test_datasetUtil")
async def health_check():
    modelID = '202601051631057129cpKTaXeI7EJiiF'
    trainingSetIDs = ['202601061333464485gaMna~ccGuzI9E']
    util = DatasetsUtil(trainingSetIDs=trainingSetIDs, modelType="hk")
    datasets_jsonl_path = util.create_train_data(modelID=modelID, modelType="hk")
    return ResponseUtil.success(data=datasets_jsonl_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.uvicorn_reload,
        workers=1,
    )
