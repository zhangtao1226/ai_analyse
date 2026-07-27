# -*-coding : utf-8 -*-
# @Author   : zhangtao
# @FileName : ResponseModel.py
# @Desc     : 
# @Time     : 2025/11/7 14:55
# @Software : PyCharm

from typing import Any, Optional, Dict, List, Generic, TypeVar
from pydantic import BaseModel
from fastapi import status

T = TypeVar('T')

class BaseResponse(BaseModel, Generic[T]):
    """基础响应模型"""
    code: int = 200
    message: str = "success"
    data: Optional[T] = None

class ErrorResponse(BaseResponse):
    """错误响应模型"""
    code: int = 400
    message: str = "error"
    details: Optional[Dict[str, Any]] = None