# -*- coding: utf-8 -*-
"""在线档案审核请求模型。"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ReviewArchiveItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    arid: str = Field(min_length=1)
    title: str = Field(alias="题名", min_length=1)
    date_time: str = Field(default="", alias="成文日期")
    files: list[Any] = Field(default_factory=list)
    content: str = ""
    keywords: list[str] = Field(default_factory=list)
    audit_result: Literal["", "开放", "控制"] = ""
    archive_category: str = Field(default="", alias="门类")
    organization_problem: str = Field(default="", alias="机构或问题")
    fonds_no: str = Field(default="", alias="全宗号")
    document_no: str = Field(default="", alias="文号")
    responsible_org: str = Field(default="", alias="责任者")
    retention_period: str = Field(default="", alias="保管期限")
    archive_year: str = Field(default="", alias="归档年度")

    @field_validator("keywords")
    @classmethod
    def normalize_keywords(cls, values: list[str]) -> list[str]:
        normalized = []
        seen = set()
        for value in values:
            keyword = value.strip()
            if keyword and keyword not in seen:
                seen.add(keyword)
                normalized.append(keyword)
        return normalized

    @field_validator("date_time", mode="before")
    @classmethod
    def normalize_partial_date(cls, value: Any) -> str:
        text = str(value or "").strip()
        if len(text) == 8 and text.isdigit():
            if text[4:] == "0000":
                return text[:4]
            if text[6:] == "00":
                return f"{text[:4]}-{text[4:6]}"
        return text


class ReviewRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    model_id: str = Field(alias="modelId", min_length=1)
    ai_audit_id: str = Field(alias="aiAuditId", min_length=1)
    model_type: str = Field(alias="modelType", min_length=1)
    data: list[ReviewArchiveItem] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_keyword_result_contract(self):
        for index, item in enumerate(self.data, 1):
            if self.model_type != "hk":
                item.keywords = []
                item.audit_result = ""
                continue
            if item.keywords and not item.audit_result:
                raise ValueError(
                    f"data[{index}]存在keywords时，audit_result必须为开放或控制"
                )
            if not item.keywords and item.audit_result:
                raise ValueError(
                    f"data[{index}]的keywords为空时，audit_result必须为空"
                )
        return self
