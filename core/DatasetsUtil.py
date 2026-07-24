# -*-coding : utf-8 -*-
# @Author   : zhangtao
# @FileName : DatasetsUtil.py
# @Desc     : 请求训练集数据并生成可训练 jsonl 文件
# @Time     : 2025/11/11 15:51
# @Software : PyCharm

import json
import os
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

import requests

from core.config import settings
from core.LoggerDetector import logger
from services.ArchiveContentExtractor import ArchiveContentExtractor
from services.ReviewKnowledgeService import RuleKnowledgeRetriever


TRAINING_SET_URL = "http://192.168.10.40:8080/product-archives/model.getTrainingSetData.erren"


class DatasetsUtil:

    HK_INSTRUCTION = (
        "你是档案划控专家。对下面的档案进行开放/控制审核，请仅使用中文回答。"
        "只输出一行JSON，字段顺序为：审核结果、审核依据、置信度、思考过程。"
    )

    JD_INSTRUCTION = (
        "你是档案鉴定专家。对下面的档案确定保管期限，请仅使用中文回答。"
        "只输出一行JSON，字段顺序为：审核结果、审核依据、置信度、思考过程。"
    )

    def __init__(self, trainingSetIDs, modelType: str):
        self.trainingSetIDs = trainingSetIDs
        if modelType not in settings.model_type_list:
            raise ValueError(f"不支持的模型类型: {modelType}")
        self.modelType = modelType
        rule_paths = {
            modelType: settings.rules[modelType]["datasets_path"]
        }
        self.retriever = RuleKnowledgeRetriever(rule_paths=rule_paths, top_k=3)
        self.content_extractor = ArchiveContentExtractor.from_settings(settings)
        self.training_content_max_chars = max(
            500,
            int(getattr(settings, "training_content_max_chars", 2200)),
        )

    def request_datasets(self) -> Dict[str, Any]:
        logger.info(f"train = {self.trainingSetIDs}")
        params = {
            "trainingSetID": self.trainingSetIDs,
            "containFile": "true",
        }

        try:
            response = requests.get(TRAINING_SET_URL, params=params, timeout=60)
            print(1111,  response.request.url)
            response.raise_for_status()
            datasets = response.json()
        except Exception as e:
            logger.error(f"请求数据集报错; details: {e}")
            raise RuntimeError(f"请求数据集失败: {e}") from e

        if not isinstance(datasets, dict):
            raise ValueError(f"数据集接口返回格式异常: {type(datasets)}")
        return datasets

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str) and value.strip().lower() in {
            "", "null", "none", "undefined", "nil", "<null>"
        }:
            return True
        return False

    @staticmethod
    def _first_present(data: Dict[str, Any], keys: Iterable[str], default: str = "") -> str:
        for key in keys:
            if key in data and not DatasetsUtil._is_missing(data[key]):
                return str(data[key]).strip()
        return default

    @classmethod
    def _get_title(cls, data: Dict[str, Any]) -> str:
        return cls._first_present(data, ("tm", "题名", "title", "archiveTitle"), "无题名")

    @classmethod
    def _get_date(cls, data: Dict[str, Any]) -> str:
        return cls._first_present(
            data,
            ("cwrq", "成文日期", "date_time", "date", "gdnd", "归档年度", "year"),
            "未知",
        )

    @classmethod
    def _get_basis(cls, data: Dict[str, Any]) -> str:
        return cls._first_present(
            data,
            (
                "lyg_audit_basis",
                "audit_basis",
                "审核依据",
                "依据",
                "reason",
            ),
            "",
        )

    @staticmethod
    def _normalize_hk_result(value: Any) -> Optional[str]:
        text = str(value or "").strip()
        if text in {"kf", "开放", "open", "OPEN", "1", "true", "True"}:
            return "开放"
        if text in {"kz", "控制", "不开放", "control", "CONTROL", "0", "false", "False"}:
            return "控制"
        if "开放" in text and "不开放" not in text:
            return "开放"
        logger.warning(f"未知划控结果: {value}")
        return None

    @staticmethod
    def _normalize_jd_result(value: Any) -> Optional[str]:
        text = str(value or "").strip()
        valid = ["永久", "60年", "30年", "15年", "10年"]
        for item in valid:
            if item == text or item in text:
                return item
        logger.warning(f"未知保管期限: {value}")
        return None

    def _get_content(self, data: Dict[str, Any]) -> str:
        return self.content_extractor.extract_archive_content(data)

    @classmethod
    def _has_attachments(cls, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (list, tuple)):
            return any(cls._has_attachments(item) for item in value)
        if isinstance(value, dict):
            return bool(value)
        if isinstance(value, (bytes, bytearray, memoryview)):
            return bool(value)
        return not cls._is_missing(value)

    @classmethod
    def _select_training_content(cls, content: str, model_type: str, max_chars: int) -> str:
        """为有限上下文选择首尾和规则相关证据，避免分词阶段盲目删除正文中段。"""
        content = str(content or "").strip()
        if len(content) <= max_chars:
            return content

        common_keywords = (
            "绝密", "机密", "秘密", "保密", "解密", "身份证", "住址", "电话",
            "工资", "待遇", "任免", "处分", "考察", "干部", "案件", "立案",
            "判刑", "产权", "股权", "房产", "补偿", "信访", "举报", "投诉",
        )
        jd_keywords = (
            "永久", "定期", "保管期限", "重大", "重要", "决策", "历史研究",
            "凭证", "权益", "资产", "合同", "会议记录", "年度总结",
        )
        keywords = common_keywords if model_type == "hk" else common_keywords + jd_keywords

        head_size = min(650, max(200, max_chars // 3))
        tail_size = min(500, max(150, max_chars // 4))
        head = (0, min(len(content), head_size))
        tail = (max(0, len(content) - tail_size), len(content))
        risk_spans = []
        for keyword in keywords:
            start_at = 0
            match_count = 0
            while match_count < 2:
                position = content.find(keyword, start_at)
                if position < 0:
                    break
                risk_spans.append((max(0, position - 130), min(len(content), position + len(keyword) + 220)))
                start_at = position + len(keyword)
                match_count += 1

        merged_risks = []
        for start, end in sorted(risk_spans):
            if merged_risks and start <= merged_risks[-1][1] + 30:
                merged_risks[-1] = (merged_risks[-1][0], max(merged_risks[-1][1], end))
            else:
                merged_risks.append((start, end))

        separator = "\n……中间非关键内容已省略……\n"
        selected = [head]
        reserved = (head[1] - head[0]) + (tail[1] - tail[0]) + len(separator)
        risk_budget = max(0, max_chars - reserved - len(separator))
        for start, end in merged_risks:
            if risk_budget <= 0:
                break
            if end <= head[1] or start >= tail[0]:
                continue
            start = max(start, head[1])
            end = min(end, tail[0], start + risk_budget)
            if end > start:
                selected.append((start, end))
                risk_budget -= end - start + len(separator)

        selected.append(tail)
        merged = []
        for start, end in sorted(selected):
            if merged and start <= merged[-1][1] + 30:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        excerpts = [content[start:end].strip() for start, end in merged if content[start:end].strip()]
        selected = separator.join(excerpts).strip()
        return selected[:max_chars]

    @staticmethod
    def _extract_risk_elements(content: str) -> List[Dict[str, str]]:
        categories = {
            "涉密信息": ("绝密", "机密", "秘密", "保密期限", "解密"),
            "个人信息": ("身份证", "住址", "电话", "工资", "待遇", "个人简历"),
            "人事信息": ("任免", "处分", "考察", "干部", "职工登记"),
            "案件信息": ("立案", "审理", "判刑", "枪决", "犯罪", "案件"),
            "权益信息": ("产权", "债权", "股权", "宅基地", "房产", "补偿标准"),
            "信访信息": ("信访", "来信", "举报", "投诉"),
        }
        elements = []
        for category, keywords in categories.items():
            matched = [keyword for keyword in keywords if keyword in content]
            if not matched:
                continue
            keyword = matched[0]
            position = content.find(keyword)
            start = max(0, position - 60)
            end = min(len(content), position + len(keyword) + 100)
            elements.append({
                "类型": category,
                "关键词": "、".join(matched),
                "证据片段": content[start:end].replace("\n", " "),
            })
        return elements

    @staticmethod
    def _archive_for_retrieval(data: Dict[str, Any], title: str, date: str, content: str) -> Dict[str, str]:
        return {
            "title": title,
            "date_time": date,
            "content": content,
            "archive_category": DatasetsUtil._first_present(data, ("门类", "ml", "arcode"), ""),
            "organization_problem": DatasetsUtil._first_present(
                data, ("机构或问题", "jghwt", "jgwt"), ""
            ),
            "fonds_no": DatasetsUtil._first_present(data, ("全宗号", "qzh"), ""),
            "document_no": DatasetsUtil._first_present(data, ("文号", "wh"), ""),
            "responsible_org": DatasetsUtil._first_present(data, ("责任者", "zrz"), ""),
        }

    def _build_training_input(self, data: Dict[str, Any], model_type: str, title: str, date: str) -> str:
        full_content = self._get_content(data)
        raw_files = data.get("files")
        has_files = self._has_attachments(raw_files)
        if not full_content and has_files:
            raise ValueError("训练样本存在附件，但未能从附件提取正文")
        if not full_content and (title == "无题名" or date == "未知"):
            raise ValueError("无正文训练样本必须同时提供题名和日期")

        archive = self._archive_for_retrieval(data, title, date, full_content)
        rules = self.retriever.search(model_type, archive)
        if full_content:
            training_content = self._select_training_content(
                full_content,
                model_type,
                self.training_content_max_chars,
            )
            evidence_type = "正文或附件"
        else:
            training_content = "（未提供档案正文或附件，本样本仅使用题名、日期及现有元数据）"
            evidence_type = "题名和日期元数据"
            logger.info(f"训练样本无附件正文，按题名和日期生成元数据样本: title={title}, date={date}")

        payload = {
            "题名": title,
            "成文日期": date,
            "门类": archive["archive_category"],
            "机构或问题": archive["organization_problem"],
            "全宗号": archive["fonds_no"],
            "文号": archive["document_no"],
            "责任者": archive["responsible_org"],
            "训练证据类型": evidence_type,
            "正文": training_content,
            "正文是否节选": bool(full_content) and len(training_content) < len(full_content),
            "风险要素": self._extract_risk_elements(full_content) if model_type == "hk" else [],
            "候选规范条款": [
                {"rule_id": rule["rule_id"], "来源": rule["source"], "内容": rule["text"]}
                for rule in rules
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _build_output(result: str, basis_prefix: str, basis: str, confidence: float) -> str:
        clean_basis = basis.strip().rstrip("。.,;；")
        if not clean_basis:
            clean_basis = f"根据档案题名和日期综合判定为{result}"

        output = {
            "审核结果": result,
            "审核依据": f"{basis_prefix}，{clean_basis}",
            "置信度": confidence,
            "思考过程": clean_basis,
        }
        return json.dumps(output, ensure_ascii=False)

    @classmethod
    def _get_confidence(cls, data: Dict[str, Any], default: float = 8.0) -> float:
        raw = cls._first_present(data, ("置信度", "confidence", "audit_confidence"), "")
        try:
            value = float(str(raw).replace("%", ""))
            if value > 10:
                value /= 10
            return round(max(0.0, min(10.0, value)), 1)
        except (TypeError, ValueError):
            return default

    def _format_hk_item(self, data: Dict[str, Any]) -> Dict[str, str]:
        title = self._get_title(data)
        date = self._get_date(data)
        result = self._normalize_hk_result(self._first_present(data, ("kzf", "开放意见", "审核结果", "result")))
        basis = self._get_basis(data)
        if not result:
            raise ValueError("划控结果不是开放或控制")
        if not basis:
            raise ValueError("缺少专家审核依据")

        return {
            "instruction": self.HK_INSTRUCTION,
            "input": self._build_training_input(data, "hk", title, date),
            "output": self._build_output(
                result=result,
                basis_prefix="依据《连云港市档案馆延期开放档案标准及范围（试用）》",
                basis=basis,
                confidence=self._get_confidence(data),
            ),
        }

    def _format_jd_item(self, data: Dict[str, Any]) -> Dict[str, str]:
        title = self._get_title(data)
        date = self._get_date(data)
        result = self._normalize_jd_result(self._first_present(data, ("bgqx", "保管期限", "审核结果", "result")))
        basis = self._get_basis(data)
        if not result:
            raise ValueError("保管期限标签无效")
        if not basis:
            raise ValueError("缺少专家审核依据")

        return {
            "instruction": self.JD_INSTRUCTION,
            "input": self._build_training_input(data, "jd", title, date),
            "output": self._build_output(
                result=result,
                basis_prefix="依据档案保管规范",
                basis=basis,
                confidence=self._get_confidence(data),
            ),
        }

    @staticmethod
    def _validate_formatted_item(item: Dict[str, str]) -> bool:
        required = ("instruction", "input", "output")
        if not all(item.get(key) for key in required):
            return False
        try:
            output = json.loads(item["output"])
        except Exception:
            return False
        return all(key in output for key in ("思考过程", "审核结果", "审核依据", "置信度"))

    @staticmethod
    def _validate_training_distribution(formatted_data: List[Dict[str, str]], model_type: str) -> Counter:
        sample_count = len(formatted_data)
        min_samples = max(2, int(getattr(settings, "training_min_samples", 10)))
        if sample_count < min_samples:
            raise ValueError(
                f"有效训练样本仅{sample_count}条，至少需要{min_samples}条才能建立训练/验证集"
            )

        result_counts = Counter()
        for item in formatted_data:
            try:
                result = str(json.loads(item["output"])["审核结果"]).strip()
            except Exception as error:
                raise ValueError(f"训练样本审核结果无法解析: {error}") from error
            if result:
                result_counts[result] += 1

        min_label_types = max(2, int(getattr(settings, "training_min_label_types", 2)))
        if model_type == "hk":
            missing_labels = {"开放", "控制"} - set(result_counts)
            if missing_labels:
                raise ValueError(
                    f"划控训练集必须同时包含开放和控制样本，当前分布={dict(result_counts)}，"
                    f"缺少={sorted(missing_labels)}"
                )
        elif len(result_counts) < min_label_types:
            raise ValueError(
                f"鉴定训练集至少需要{min_label_types}种保管期限，当前分布={dict(result_counts)}"
            )

        min_per_label = max(1, int(getattr(settings, "training_min_samples_per_label", 2)))
        insufficient = {
            label: count for label, count in result_counts.items() if count < min_per_label
        }
        if insufficient:
            raise ValueError(
                f"每种审核结果至少需要{min_per_label}条样本，数量不足={insufficient}，"
                f"当前分布={dict(result_counts)}"
            )
        return result_counts

    def _format_record(self, record: Dict[str, Any], modelType: str) -> Optional[Dict[str, str]]:
        try:
            if modelType == "hk":
                item = self._format_hk_item(record)
            elif modelType == "jd":
                item = self._format_jd_item(record)
            else:
                raise ValueError(f"不支持的模型类型: {modelType}")
        except ValueError as error:
            logger.warning(f"训练样本质量校验失败，已跳过: {error}")
            return None

        if not self._validate_formatted_item(item):
            logger.warning(f"训练样本格式校验失败，已跳过: {record}")
            return None
        return item

    def create_train_data(self, modelID, modelType: Optional[str] = None):
        modelType = modelType or self.modelType
        if modelType != self.modelType:
            raise ValueError(
                f"数据集工具模型类型不一致: initialized={self.modelType}, requested={modelType}"
            )
        datasets = self.request_datasets()
        records = datasets.get("data") or []
        print(f"数据集: {records}")
        if not records:
            logger.error(f"获取数据集返回结果为空: {datasets}")
            raise ValueError("数据集没有数据")

        formatted_data: List[Dict[str, str]] = []
        seen_samples = set()
        duplicate_count = 0
        for idx, item in enumerate(records):
            record = item.get("record") if isinstance(item, dict) else None
            if not isinstance(record, dict):
                logger.warning(f"第{idx + 1}条数据缺少record字段，已跳过: {item}")
                continue

            record = dict(record)
            if not record.get("files") and isinstance(item, dict):
                record["files"] = item.get("files") or []

            formatted = self._format_record(record, modelType)
            if formatted:
                fingerprint = (formatted["input"], formatted["output"])
                if fingerprint in seen_samples:
                    duplicate_count += 1
                    logger.warning(f"第{idx + 1}条训练样本重复，已跳过")
                    continue
                seen_samples.add(fingerprint)
                formatted_data.append(formatted)

        if not formatted_data:
            raise ValueError("没有可用的有效训练样本")

        result_counts = self._validate_training_distribution(formatted_data, modelType)

        os.makedirs(settings.save_dataset_path, exist_ok=True)
        json_name = f"{modelID}_{int(time.time())}.jsonl"
        datasets_json_path = os.path.join(settings.save_dataset_path, json_name)
        preparing_path = f"{datasets_json_path}.preparing"

        # 先完整写入临时文件，再原子替换为最终JSONL。训练进程不会读到半成品。
        try:
            with open(preparing_path, "w", encoding="utf-8") as f:
                for item in formatted_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            if os.path.getsize(preparing_path) == 0:
                raise ValueError("整理后的训练数据集为空")
            os.replace(preparing_path, datasets_json_path)
        except Exception:
            if os.path.exists(preparing_path):
                try:
                    os.remove(preparing_path)
                except OSError as cleanup_error:
                    logger.warning(f"训练数据临时文件清理失败: {preparing_path}, {cleanup_error}")
            raise

        logger.info(
            f"训练数据已生成: {datasets_json_path}, "
            f"有效样本数={len(formatted_data)}, 原始样本数={len(records)}, "
            f"重复样本数={duplicate_count}, 标签分布={dict(result_counts)}"
        )
        return datasets_json_path
