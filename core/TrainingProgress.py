# -*- coding: utf-8 -*-
"""单机训练进度存储。

训练进程与FastAPI进程不共享普通Python对象，因此使用同一文件系统目录
交换进度。每个模型一个JSON文件，并通过临时文件 + os.replace原子更新。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

from core.LoggerDetector import logger


class TrainingProgressStore:
    def __init__(self, progress_dir: str):
        self.progress_dir = Path(progress_dir).resolve()
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def _normalize_percentage(value: float) -> float:
        return round(max(0.0, min(100.0, float(value))), 1)

    def _path(self, model_id: str) -> Path:
        model_id = str(model_id or "").strip()
        if not model_id:
            raise ValueError("modelID不能为空")
        digest = hashlib.sha256(model_id.encode("utf-8")).hexdigest()
        return self.progress_dir / f"{digest}.json"

    def reset(self, model_id: str) -> float:
        """开始新训练任务时强制重置为0%。"""
        self._write(model_id, 0.0)
        return 0.0

    def update(self, model_id: str, percentage: float) -> float:
        """单调更新百分比，避免多进程迟到消息导致进度倒退。"""
        normalized = self._normalize_percentage(percentage)
        with self._lock:
            current = self._read_unlocked(model_id)
            if current is not None:
                normalized = max(current, normalized)
                if normalized == current:
                    return current
            self._write_unlocked(model_id, normalized)
        return normalized

    def read(self, model_id: str) -> Optional[float]:
        with self._lock:
            return self._read_unlocked(model_id)

    def _read_unlocked(self, model_id: str) -> Optional[float]:
        path = self._path(model_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return self._normalize_percentage(payload["completedPercentage"])
        except Exception as error:
            logger.warning(f"读取模型训练进度失败: modelID={model_id}, path={path}, error={error}")
            return None

    def _write(self, model_id: str, percentage: float) -> None:
        with self._lock:
            self._write_unlocked(model_id, self._normalize_percentage(percentage))

    def _write_unlocked(self, model_id: str, percentage: float) -> None:
        path = self._path(model_id)
        temp_path = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        payload = {
            "modelID": str(model_id),
            "completedPercentage": self._normalize_percentage(percentage),
            "updatedAt": int(time.time()),
        }
        try:
            temp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(temp_path, path)
        except Exception:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise
