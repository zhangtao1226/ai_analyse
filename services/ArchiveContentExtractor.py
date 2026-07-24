# -*- coding: utf-8 -*-
"""档案附件正文提取。

训练集生成和在线审核共用该模块：优先使用已有文本，其次提取电子
PDF 的文本层，最后才对扫描页和图片执行 OCR。OCR 结果按文件内容缓存，
避免每次生成训练集或每轮训练时重复识别。
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from PIL import Image

from core.LoggerDetector import logger


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_TEXT_SUFFIXES = {".txt", ".text", ".csv", ".json", ".xml", ".html", ".htm"}
_TEXT_CONTENT_TYPES = {
    "application/json",
    "application/xml",
    "application/csv",
}
_BASE64_FIELDS = ("base64", "contentBase64", "fileBase64", "fileData", "fileContent")
_DATA_URI_RE = re.compile(
    r"^data:(?P<mime>[-\w.+/]+)?(?:;charset=[^;,]+)?;base64,(?P<data>.+)$",
    re.IGNORECASE | re.DOTALL,
)


class ArchiveContentExtractor:
    """从档案记录中提取可供模型使用的正文文本。"""

    CACHE_VERSION = "archive-content-v3"

    def __init__(
        self,
        *,
        file_download_base_url: str,
        file_download_timeout: int = 60,
        ocr_config: Optional[Dict[str, str]] = None,
        ocr_device: str = "npu:0",
        ocr_fallback_device: str = "cpu",
        cache_dir: Optional[str] = None,
        max_file_bytes: int = 50 * 1024 * 1024,
        max_pdf_pages: int = 200,
        pdf_text_min_chars: int = 20,
        pdf_dpi: int = 200,
        session: Optional[requests.Session] = None,
        ocr_factory: Optional[Callable[[], Any]] = None,
    ):
        self.file_download_base_url = str(file_download_base_url or "").strip()
        self.file_download_timeout = max(1, int(file_download_timeout))
        self.ocr_config = dict(ocr_config or {})
        self.ocr_device = str(ocr_device or "npu:0").strip()
        self.ocr_fallback_device = str(ocr_fallback_device or "cpu").strip()
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else None
        self.max_file_bytes = max(1, int(max_file_bytes))
        self.max_pdf_pages = max(1, int(max_pdf_pages))
        self.pdf_text_min_chars = max(1, int(pdf_text_min_chars))
        self.pdf_dpi = max(72, int(pdf_dpi))
        self.session = session or requests.Session()
        self._ocr_factory = ocr_factory
        self._ocr_detector = None
        self._ocr_lock = threading.Lock()

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_settings(cls, settings: Any) -> "ArchiveContentExtractor":
        return cls(
            file_download_base_url=settings.file_download_base_url,
            file_download_timeout=settings.file_download_timeout,
            ocr_config=settings.ocr_config,
            ocr_device=getattr(settings, "ocr_device", "npu:0"),
            ocr_fallback_device=getattr(settings, "ocr_fallback_device", "cpu"),
            cache_dir=getattr(settings, "ocr_cache_dir", None),
            max_file_bytes=getattr(settings, "ocr_max_file_bytes", 50 * 1024 * 1024),
            max_pdf_pages=getattr(settings, "ocr_max_pdf_pages", 200),
            pdf_text_min_chars=getattr(settings, "ocr_pdf_text_min_chars", 20),
            pdf_dpi=getattr(settings, "ocr_pdf_dpi", 200),
        )

    @staticmethod
    def _clean_text(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        value = value.replace("\x00", "").strip()
        if value.lower() in {"null", "none", "undefined", "nil", "<null>"}:
            return ""
        return value

    @staticmethod
    def _normalise_files(files: Any) -> Iterable[Any]:
        if files is None:
            return []
        if isinstance(files, (list, tuple)):
            return files
        return [files]

    def extract_archive_content(self, archive_item: Dict[str, Any]) -> str:
        """按正文、附件顺序提取文本；单个附件失败不影响其他附件。"""
        sections = []
        for field in ("content", "正文", "档案内容", "ocrContent"):
            direct = self._clean_text(archive_item.get(field))
            if direct:
                sections.append(direct)
                break

        archive_id = archive_item.get("arid") or archive_item.get("原档案ID") or ""
        for index, attachment in enumerate(self._normalise_files(archive_item.get("files")), 1):
            try:
                text, source = self.extract_attachment(attachment)
                if text:
                    label = f"【附件{index}】" if source == "inline" else f"【附件{index} {source}】"
                    sections.append(f"{label}\n{text}")
                else:
                    logger.warning(f"档案{archive_id}附件{index}未提取到文本")
            except Exception as error:
                logger.error(f"档案{archive_id}附件{index}正文提取失败: {error}")

        return "\n\n".join(section for section in sections if section.strip()).strip()

    def extract_attachment(self, attachment: Any) -> Tuple[str, str]:
        """返回 ``(文本, 来源)``，来源用于审计和训练集标记。"""
        if isinstance(attachment, str):
            data_uri = self._decode_data_uri(attachment)
            if data_uri:
                payload, content_type = data_uri
                return self.extract_binary(payload, content_type, "inline")
            if self._looks_like_url(attachment):
                payload, content_type, source_url = self.download_attachment(attachment)
                return self.extract_binary(payload, content_type, source_url)
            payload = self._try_decode_file_base64(attachment)
            if payload is not None:
                return self.extract_binary(payload, "", "inline")
            return self._clean_text(attachment), "inline"

        if isinstance(attachment, (bytes, bytearray, memoryview)):
            return self.extract_binary(bytes(attachment), "", "inline")

        if not isinstance(attachment, dict):
            raise ValueError(f"不支持的附件数据类型: {type(attachment).__name__}")

        content = attachment.get("content")
        if isinstance(content, str):
            data_uri = self._decode_data_uri(content)
            if data_uri:
                payload, content_type = data_uri
                return self.extract_binary(payload, content_type, "inline")
            # 新训练集接口协议：出现filename字段时，content固定表示文件Base64，
            # 即使filename为空也不能再把content当作普通OCR文本。
            if "filename" in attachment:
                if content.strip().lower() in {"", "base64", "<base64>", "null", "none"}:
                    raise ValueError("files.content是Base64占位符或空值，未提供真实文件内容")
                payload = self._decode_base64(content)
                return self.extract_binary(
                    payload,
                    self._attachment_content_type(attachment),
                    self._attachment_name(attachment),
                )
            encoding = str(attachment.get("encoding") or "").lower()
            if encoding in {"base64", "b64"}:
                payload = self._decode_base64(content)
                return self.extract_binary(
                    payload,
                    self._attachment_content_type(attachment),
                    self._attachment_name(attachment),
                )
            payload = self._try_decode_file_base64(content, attachment)
            if payload is not None:
                return self.extract_binary(
                    payload,
                    self._attachment_content_type(attachment),
                    self._attachment_name(attachment),
                )
            text = self._clean_text(content)
            if text:
                return text, "inline"
        elif isinstance(content, (bytes, bytearray, memoryview)):
            return self.extract_binary(
                bytes(content),
                self._attachment_content_type(attachment),
                self._attachment_name(attachment),
            )

        for field in _BASE64_FIELDS:
            encoded = attachment.get(field)
            if not isinstance(encoded, str) or not encoded.strip():
                continue
            payload = self._decode_base64(encoded)
            return self.extract_binary(
                payload,
                self._attachment_content_type(attachment),
                self._attachment_name(attachment),
            )

        raw_bytes = attachment.get("bytes") or attachment.get("binary")
        if isinstance(raw_bytes, list) and all(isinstance(value, int) and 0 <= value <= 255 for value in raw_bytes):
            raw_bytes = bytes(raw_bytes)
        if isinstance(raw_bytes, (bytes, bytearray, memoryview)):
            return self.extract_binary(
                bytes(raw_bytes),
                self._attachment_content_type(attachment),
                self._attachment_name(attachment),
            )

        source_url = self._clean_text(attachment.get("url"))
        if source_url:
            payload, content_type, resolved_url = self.download_attachment(source_url)
            return self.extract_binary(payload, content_type, resolved_url)

        raise ValueError("附件没有可用的content、二进制内容或url")

    @staticmethod
    def _looks_like_url(value: str) -> bool:
        value = value.strip()
        return value.startswith(("http://", "https://", "/"))

    @staticmethod
    def _attachment_content_type(attachment: Dict[str, Any]) -> str:
        return str(
            attachment.get("contentType")
            or attachment.get("content_type")
            or attachment.get("mimeType")
            or attachment.get("mime_type")
            or ""
        )

    @staticmethod
    def _attachment_name(attachment: Dict[str, Any]) -> str:
        return str(
            attachment.get("name")
            or attachment.get("fileName")
            or attachment.get("filename")
            or attachment.get("originalName")
            or "inline"
        )

    @staticmethod
    def _decode_base64(value: str) -> bytes:
        compact = re.sub(r"\s+", "", value).replace("-", "+").replace("_", "/")
        compact += "=" * (-len(compact) % 4)
        try:
            payload = base64.b64decode(compact, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("附件Base64内容无效") from error
        if not payload:
            raise ValueError("附件Base64解码结果为空")
        return payload

    def _try_decode_file_base64(
        self,
        value: str,
        attachment: Optional[Dict[str, Any]] = None,
    ) -> Optional[bytes]:
        """仅在解码结果具备文件特征时自动认定为Base64，避免误判普通正文。"""
        compact = re.sub(r"\s+", "", str(value or ""))
        if len(compact) < 16 or not re.fullmatch(r"[A-Za-z0-9+/=_-]+", compact):
            return None
        try:
            payload = self._decode_base64(compact)
        except ValueError:
            return None
        if len(payload) > self.max_file_bytes:
            raise ValueError(f"Base64附件超过大小限制: {len(payload)} > {self.max_file_bytes}")

        content_type = self._attachment_content_type(attachment or {}).lower()
        suffix = Path(self._attachment_name(attachment or {})).suffix.lower()
        declared_file = (
            content_type == "application/pdf"
            or content_type.startswith(("image/", "text/"))
            or suffix in _IMAGE_SUFFIXES | _TEXT_SUFFIXES | {".pdf"}
        )
        has_file_signature = payload.startswith(b"%PDF") or self._is_image(payload)
        return payload if declared_file or has_file_signature else None

    @classmethod
    def _decode_data_uri(cls, value: str) -> Optional[Tuple[bytes, str]]:
        match = _DATA_URI_RE.match(value.strip())
        if not match:
            return None
        return cls._decode_base64(match.group("data")), (match.group("mime") or "")

    def resolve_download_url(self, url: str) -> str:
        url = str(url or "").strip()
        if not url:
            raise ValueError("附件url为空")
        parsed = urlparse(url)
        if parsed.scheme:
            if parsed.scheme not in {"http", "https"}:
                raise ValueError(f"不支持的附件URL协议: {parsed.scheme}")
            return url
        if not self.file_download_base_url:
            raise ValueError("相对附件URL缺少file_download_base_url配置")
        return urljoin(self.file_download_base_url.rstrip("/") + "/", url.lstrip("/"))

    def download_attachment(self, url: str) -> Tuple[bytes, str, str]:
        resolved_url = self.resolve_download_url(url)
        response = self.session.get(resolved_url, timeout=self.file_download_timeout, stream=True)
        response.raise_for_status()

        declared_size = response.headers.get("content-length")
        if declared_size and int(declared_size) > self.max_file_bytes:
            raise ValueError(f"附件超过大小限制: {declared_size} > {self.max_file_bytes}")

        chunks = []
        size = 0
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            size += len(chunk)
            if size > self.max_file_bytes:
                raise ValueError(f"附件超过大小限制: {size} > {self.max_file_bytes}")
            chunks.append(chunk)
        payload = b"".join(chunks)
        if not payload:
            raise ValueError(f"附件下载结果为空: {resolved_url}")
        content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
        return payload, content_type, resolved_url

    def extract_binary(self, data: bytes, content_type: str, source_name: str) -> Tuple[str, str]:
        if not data:
            raise ValueError("附件二进制内容为空")
        if len(data) > self.max_file_bytes:
            raise ValueError(f"附件超过大小限制: {len(data)} > {self.max_file_bytes}")

        content_type = str(content_type or "").split(";", 1)[0].lower()
        cached = self._read_cache(data)
        if cached is not None:
            return cached

        suffix = Path(urlparse(str(source_name or "")).path).suffix.lower()
        if content_type.startswith("text/") or content_type in _TEXT_CONTENT_TYPES or suffix in _TEXT_SUFFIXES:
            text = self._decode_text(data)
            method = "文本提取"
        elif content_type == "application/pdf" or suffix == ".pdf" or data.startswith(b"%PDF"):
            text = self._extract_pdf(data)
            method = "PDF文本/OCR"
        elif content_type.startswith("image/") or suffix in _IMAGE_SUFFIXES or self._is_image(data):
            text = self._ocr_image_bytes(data)
            method = "OCR识别内容"
        else:
            raise ValueError(f"暂不支持的附件类型: content-type={content_type}, suffix={suffix}")

        text = text.strip()
        if text:
            self._write_cache(data, text, method)
        return text, method

    @staticmethod
    def _decode_text(data: bytes) -> str:
        for encoding in ("utf-8-sig", "utf-16", "gb18030"):
            try:
                return data.decode(encoding).replace("\x00", "").strip()
            except UnicodeDecodeError:
                continue
        raise ValueError("文本附件编码无法识别")

    @staticmethod
    def _is_image(data: bytes) -> bool:
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            return True
        except Exception:
            return False

    def _extract_pdf(self, data: bytes) -> str:
        try:
            import fitz
        except ImportError as error:
            raise RuntimeError("处理PDF附件需要安装PyMuPDF") from error

        document = fitz.open(stream=data, filetype="pdf")
        try:
            if len(document) > self.max_pdf_pages:
                raise ValueError(f"PDF页数超过限制: {len(document)} > {self.max_pdf_pages}")

            pages = []
            for page_index in range(len(document)):
                page = document.load_page(page_index)
                native_text = page.get_text("text").replace("\x00", "").strip()
                if len(re.sub(r"\s+", "", native_text)) >= self.pdf_text_min_chars:
                    page_text = native_text
                    source = "文本层"
                else:
                    pix = page.get_pixmap(matrix=fitz.Matrix(self.pdf_dpi / 72, self.pdf_dpi / 72), alpha=False)
                    page_text = self._ocr_image_bytes(pix.tobytes("png"))
                    source = "OCR"
                if page_text:
                    pages.append(f"【第{page_index + 1}页 {source}】\n{page_text}")
            return "\n".join(pages).strip()
        finally:
            document.close()

    def _get_ocr_detector(self) -> Any:
        if self._ocr_detector is None:
            if self._ocr_factory:
                self._ocr_detector = self._ocr_factory()
            else:
                from services.PaddleOCRServices import PaddleOCRServices

                self._ocr_detector = PaddleOCRServices(
                    det_path=self.ocr_config.get("det_path"),
                    rec_path=self.ocr_config.get("rec_path"),
                    device=self.ocr_device,
                    fallback_device=self.ocr_fallback_device,
                )
        return self._ocr_detector

    def _ocr_image_bytes(self, data: bytes) -> str:
        temp_path = None
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                with tempfile.NamedTemporaryFile(prefix="archive_ocr_", suffix=".png", delete=False) as handle:
                    temp_path = handle.name
                image.convert("RGB").save(temp_path, format="PNG")
            # PaddleOCR推理对象跨线程并发并不稳定，单实例串行可避免模型重复加载和状态竞争。
            with self._ocr_lock:
                lines = self._get_ocr_detector().ocr_predict(temp_path) or []
            return "\n".join(str(line).strip() for line in lines if str(line).strip())
        finally:
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError as error:
                    logger.warning(f"OCR临时文件清理失败: {temp_path}, {error}")

    def _cache_path(self, data: bytes) -> Optional[Path]:
        if not self.cache_dir:
            return None
        model_signature = json.dumps(
            {
                "version": self.CACHE_VERSION,
                "det": self.ocr_config.get("det_path", ""),
                "rec": self.ocr_config.get("rec_path", ""),
                "dpi": self.pdf_dpi,
                "min_chars": self.pdf_text_min_chars,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.sha256(model_signature + b"\0" + data).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.json"

    def _read_cache(self, data: bytes) -> Optional[Tuple[str, str]]:
        path = self._cache_path(data)
        if not path or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            text = payload.get("text") if isinstance(payload, dict) else ""
            method = payload.get("method", "提取内容") if isinstance(payload, dict) else "提取内容"
            if isinstance(text, str) and text.strip():
                return text.strip(), str(method or "提取内容")
            return None
        except Exception as error:
            logger.warning(f"读取OCR缓存失败，重新识别: {path}, {error}")
            return None

    def _write_cache(self, data: bytes, text: str, method: str) -> None:
        path = self._cache_path(data)
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            temp_path.write_text(
                json.dumps(
                    {"version": self.CACHE_VERSION, "method": method, "text": text},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            os.replace(temp_path, path)
        except Exception as error:
            logger.warning(f"写入OCR缓存失败: {path}, {error}")
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
