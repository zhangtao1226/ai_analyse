# -*- coding: utf-8 -*-
"""离线规范检索与结果约束，用于 LoRA + RAG + 规则引擎审核管线。"""

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from core.LoggerDetector import logger


REVIEW_PROFILES = {
    "hk": {
        "name": "档案划控",
        "source_name": "《连云港市档案馆延期开放档案标准及范围（试用）》",
        "valid_results": ("开放", "控制"),
    },
    "jd": {
        "name": "档案鉴定",
        "source_name": "档案归档范围和保管期限规范",
        "valid_results": ("永久", "30年", "10年", "长期", "短期", "15年", "60年"),
    },
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def _char_ngrams(text: str, size: int = 2) -> Counter:
    normalized = _normalize(text)
    if len(normalized) < size:
        return Counter([normalized]) if normalized else Counter()
    return Counter(normalized[i:i + size] for i in range(len(normalized) - size + 1))


class RuleKnowledgeRetriever:
    """无需联网和额外模型的稀疏RAG；后续可替换为向量+BM25混合检索。"""

    def __init__(self, rule_paths: Dict[str, str], top_k: int = 3, max_chunk_chars: int = 1200):
        self.rule_paths = rule_paths
        self.top_k = top_k
        self.max_chunk_chars = max_chunk_chars
        self.documents: Dict[str, List[dict]] = {}
        self.document_frequency: Dict[str, Counter] = {}
        unknown_types = set(rule_paths) - set(REVIEW_PROFILES)
        if unknown_types:
            raise ValueError(f"不支持的规范知识库类型: {sorted(unknown_types)}")
        for model_type in rule_paths:
            self.documents[model_type] = self._load(model_type)
            self.document_frequency[model_type] = self._build_document_frequency(
                self.documents[model_type]
            )

    def _load(self, model_type: str) -> List[dict]:
        path = Path(self.rule_paths[model_type])
        documents = []
        if not path.exists():
            logger.error(f"规范知识库不存在: {path}")
            return documents
        seen = set()
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    text = str(json.loads(line).get("input") or "").strip()
                except Exception as error:
                    logger.warning(f"规范知识库第{line_no}行解析失败: {error}")
                    continue
                for chunk_no, chunk in enumerate(self._split(text), 1):
                    fingerprint = _normalize(chunk)
                    if len(fingerprint) < 20 or fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    documents.append({
                        "rule_id": f"{model_type.upper()}-{line_no:03d}-{chunk_no:02d}",
                        "source": REVIEW_PROFILES[model_type]["source_name"],
                        "text": chunk,
                        "terms": _char_ngrams(chunk),
                    })
        logger.info(f"规范知识库加载完成: type={model_type}, chunks={len(documents)}")
        return documents

    def _split(self, text: str) -> Iterable[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        buffer = ""
        for paragraph in paragraphs:
            if len(buffer) + len(paragraph) + 1 <= self.max_chunk_chars:
                buffer = f"{buffer}\n{paragraph}".strip()
                continue
            if buffer:
                yield buffer
            while len(paragraph) > self.max_chunk_chars:
                yield paragraph[:self.max_chunk_chars]
                paragraph = paragraph[self.max_chunk_chars:]
            buffer = paragraph
        if buffer:
            yield buffer

    @staticmethod
    def _build_document_frequency(documents: List[dict]) -> Counter:
        frequency = Counter()
        for document in documents:
            frequency.update(document["terms"].keys())
        return frequency

    def search(self, model_type: str, archive: dict) -> List[dict]:
        if model_type not in REVIEW_PROFILES:
            raise ValueError(f"不支持的modelType: {model_type}")
        metadata = " ".join(str(archive.get(key) or "") for key in (
            "title", "date_time", "archive_category", "organization_problem",
            "fonds_no", "document_no", "responsible_org",
        ))
        content = str(archive.get("content") or "")
        query = f"{metadata}\n{content[:8000]}"
        query_terms = _char_ngrams(query)
        if not query_terms:
            return []
        documents = self.documents.get(model_type, [])
        df = self.document_frequency.get(model_type, Counter())
        total = max(1, len(documents))
        ranked = []
        for document in documents:
            overlap = query_terms.keys() & document["terms"].keys()
            score = sum(
                min(query_terms[t], document["terms"][t]) * math.log((total + 1) / (df[t] + 1) + 1)
                for t in overlap
            )
            if score > 0:
                ranked.append((score, document))
        ranked.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, document in ranked[:self.top_k]:
            results.append({
                "rule_id": document["rule_id"],
                "source": document["source"],
                "text": document["text"],
                "score": round(score, 4),
            })
        return results


class ReviewRuleEngine:
    """负责modelType路由、合法结果约束和规范依据兜底。"""

    def __init__(self, model_type: str):
        if model_type not in REVIEW_PROFILES:
            raise ValueError(f"不支持的modelType: {model_type}")
        self.model_type = model_type
        self.profile = REVIEW_PROFILES[model_type]

    def validate(self, result: dict, retrieved_rules: List[dict]) -> dict:
        validated = dict(result)
        value = str(validated.get("审核结果") or "").strip()
        allowed_results = self.profile["valid_results"]
        if self.model_type == "jd" and retrieved_rules:
            retrieved_text = "\n".join(rule["text"] for rule in retrieved_rules)
            period_hits = tuple(period for period in allowed_results if period in retrieved_text)
            if period_hits:
                allowed_results = period_hits
        if self.model_type == "hk":
            if "控制" in value or "不开放" in value:
                matched = "控制"
            elif value == "开放" or "应开放" in value:
                matched = "开放"
            else:
                matched = ""
        else:
            matched = next((item for item in allowed_results if item == value), "")
        if not matched:
            matched = "控制" if self.model_type == "hk" else ""
        validated["审核结果"] = matched

        basis = str(validated.get("审核依据") or "").strip()
        if retrieved_rules:
            primary = retrieved_rules[0]
            source = primary["source"]
            rule_id = primary["rule_id"]
            excerpt = re.sub(r"\s+", " ", primary["text"]).strip()[:180]
            if source not in basis or rule_id not in basis:
                basis = f"依据{source}（检索条款编号：{rule_id}），{basis or excerpt}"
        elif not basis:
            basis = "未检索到可核验的规范条款，审核依据不足。"
        validated["审核依据"] = basis

        try:
            confidence = float(validated.get("置信度", 0))
        except (TypeError, ValueError):
            confidence = 0.0
        if not retrieved_rules:
            confidence = min(confidence, 5.0)
        validated["置信度"] = round(max(0.0, min(10.0, confidence)), 1)
        validated["思考过程"] = str(validated.get("思考过程") or "").strip()
        return validated
