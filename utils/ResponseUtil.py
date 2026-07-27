# -*-coding : utf-8 -*-
# @Author   : zhangtao
# @FileName : ResponseUtil.py
# @Desc     : 
# @Time     : 2025/11/7 14:57
# @Software : PyCharm

from typing import Any, Optional, Dict, List, Generic, TypeVar

from schemas.ResponseModel import BaseResponse, ErrorResponse

class ResponseUtil:
    """响应工具类"""

    @staticmethod
    def success(data: Any = None, message: str = "success") -> BaseResponse:
        """成功响应"""
        return BaseResponse(code=200, message=message, data=data)

    @staticmethod
    def error(code: int = 400, message: str = "error", details: Any = None) -> ErrorResponse:
        """错误响应"""
        return ErrorResponse(code=code, message=message, details=details)

    @staticmethod
    def created(data: Any = None, message: str = "创建成功") -> BaseResponse:
        """创建成功响应"""
        return BaseResponse(code=201, message=message, data=data)

    @staticmethod
    def updated(data: Any = None, message: str = "更新成功") -> BaseResponse:
        """更新成功响应"""
        return BaseResponse(code=200, message=message, data=data)

    @staticmethod
    def deleted(message: str = "删除成功") -> BaseResponse:
        """删除成功响应"""
        return BaseResponse(code=200, message=message)