# -*- coding: utf-8 -*-
"""模型训练结果回调服务。"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import requests

from core.LoggerDetector import logger


class TrainingCallbackError(RuntimeError):
    """训练结果在限定重试次数内仍未成功送达。"""


class TrainingCallbackService:
    def __init__(
            self,
            callback_url: str,
            timeout_seconds: float = 10.0,
            max_attempts: int = 3,
            retry_delay_seconds: float = 2.0,
            pending_dir: Optional[str] = None,
            post: Optional[Callable] = None,
            sleeper: Optional[Callable[[float], None]] = None,
    ):
        self.callback_url = str(callback_url or "").strip()
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.max_attempts = max(1, int(max_attempts))
        self.retry_delay_seconds = max(0.0, float(retry_delay_seconds))
        self.pending_dir = Path(pending_dir).resolve() if pending_dir else None
        if self.pending_dir:
            self.pending_dir.mkdir(parents=True, exist_ok=True)
        self._post = post or requests.post
        self._sleep = sleeper or time.sleep
        self._lock = threading.Lock()

    def _pending_path(self, model_id: str) -> Optional[Path]:
        if self.pending_dir is None:
            return None
        digest = hashlib.sha256(model_id.encode("utf-8")).hexdigest()
        return self.pending_dir / f"{digest}.json"

    def _save_pending(self, model_id: str, result: str, last_error: str = "") -> None:
        path = self._pending_path(model_id)
        if path is None:
            return
        payload = {
            "modelID": model_id,
            "result": result,
            "lastError": last_error,
            "updatedAt": int(time.time()),
        }
        temp_path = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        with self._lock:
            try:
                temp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                os.replace(temp_path, path)
            finally:
                temp_path.unlink(missing_ok=True)

    def _clear_pending(self, model_id: str) -> None:
        path = self._pending_path(model_id)
        if path is None:
            return
        with self._lock:
            path.unlink(missing_ok=True)

    def notify(self, model_id: str, result: str) -> None:
        model_id = str(model_id or "").strip()
        result = str(result or "").strip()
        if not self.callback_url:
            raise TrainingCallbackError("模型训练完成回调地址未配置")
        if not model_id:
            raise TrainingCallbackError("模型训练完成回调缺少modelID")
        if not result:
            raise TrainingCallbackError("模型训练完成回调缺少result")

        # 先持久化再发送，进程在网络请求期间异常退出时，主服务仍可继续补发。
        self._save_pending(model_id, result)
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                logger.info(
                    f"模型训练完成回调: modelID={model_id}, result={result}, "
                    f"attempt={attempt}/{self.max_attempts}, url={self.callback_url}"
                )
                response = self._post(
                    self.callback_url,
                    params={"modelID": model_id, "result": result},
                    timeout=self.timeout_seconds,
                )
                if not 200 <= response.status_code < 300:
                    raise TrainingCallbackError(
                        f"模型训练完成回调HTTP状态异常: status={response.status_code}"
                    )
                try:
                    response_payload = response.json()
                except Exception:
                    response_payload = None
                if isinstance(response_payload, dict) and "code" in response_payload:
                    business_code = str(response_payload["code"])
                    if business_code not in {"0", "200"}:
                        raise TrainingCallbackError(
                            f"模型训练完成回调业务状态异常: code={business_code}"
                        )
                logger.info(
                    f"模型训练完成回调成功: modelID={model_id}, result={result}, "
                    f"status={response.status_code}"
                )
                self._clear_pending(model_id)
                return
            except Exception as error:
                last_error = error
                logger.warning(
                    f"模型训练完成回调失败: modelID={model_id}, result={result}, "
                    f"attempt={attempt}/{self.max_attempts}, error={error}"
                )
                if attempt < self.max_attempts and self.retry_delay_seconds > 0:
                    self._sleep(self.retry_delay_seconds * (2 ** (attempt - 1)))

        self._save_pending(model_id, result, str(last_error))
        raise TrainingCallbackError(
            f"模型训练完成回调在{self.max_attempts}次尝试后仍失败: {last_error}"
        )

    def retry_pending(self) -> int:
        """重试本机待发送回调，返回本轮成功发送数量。"""
        if self.pending_dir is None:
            return 0

        succeeded = 0
        for path in list(self.pending_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.notify(payload["modelID"], payload["result"])
                succeeded += 1
            except Exception as error:
                logger.warning(f"待发送模型训练回调补发失败: path={path}, error={error}")
        return succeeded
