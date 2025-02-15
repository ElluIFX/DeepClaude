import os
from typing import Optional

from fastapi import Header, HTTPException
from loguru import logger

# 获取环境变量
ALLOW_API_KEY = os.getenv("ALLOW_API_KEY")
logger.info(f"ALLOW_API_KEY环境变量状态: {'已设置' if ALLOW_API_KEY else '未设置'}")

if not ALLOW_API_KEY:
    raise ValueError("ALLOW_API_KEY environment variable is not set")

# 打印API密钥的前4位用于调试
logger.info(
    f"Loaded API key starting with: {ALLOW_API_KEY[:4] if len(ALLOW_API_KEY) >= 4 else ALLOW_API_KEY}"
)


async def verify_api_key(authorization: Optional[str] = Header(None)) -> None:
    """验证API密钥

    Args:
        authorization (Optional[str], optional): Authorization header中的API密钥. Defaults to Header(None).

    Raises:
        HTTPException: 当Authorization header缺失或API密钥无效时抛出401错误
    """
    if authorization is None:
        logger.warning("请求缺少Authorization header")
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    api_key = authorization.replace("Bearer ", "").strip()
    if api_key != ALLOW_API_KEY:
        logger.warning(f"无效的API密钥: {api_key}")
        raise HTTPException(status_code=401, detail="Invalid API key")

    logger.info("API密钥验证通过")
