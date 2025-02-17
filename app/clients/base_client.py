"""基础客户端类，定义通用接口"""

import json
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Union

import aiohttp
from loguru import logger


class BaseClient(ABC):
    def __init__(self, api_key: str, api_url: str):
        """初始化基础客户端

        Args:
            api_key: API密钥
            api_url: API地址
        """
        self.api_key = api_key
        self.api_url = api_url

    async def _make_request(
        self, headers: dict, data: dict
    ) -> AsyncGenerator[bytes, None]:
        """发送请求并处理响应

        Args:
            headers: 请求头
            data: 请求数据

        Yields:
            bytes: 原始响应数据
        """
        logger.debug(f"发送请求体：{data}，请求头：{headers}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url, headers=headers, json=data
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"API 请求失败: {error_text}")
                        return

                    async for chunk in response.content.iter_any():
                        # logger.debug(f"chunk: {chunk}")
                        yield chunk

        except Exception as e:
            logger.error(f"请求 API 时发生错误: {e}")

    @abstractmethod
    async def stream_chat(
        self, messages: list, model: str
    ) -> AsyncGenerator[tuple[str, str], None]:
        """流式对话，由子类实现

        Args:
            messages: 消息列表
            model: 模型名称

        Yields:
            tuple[str, str]: (内容类型, 内容)
        """
        pass

    def _parse_chunk(self, chunk: str, valid_chunks: list) -> list[Union[dict, None]]:
        """解析响应块为json

        Args:
            chunk: 响应块
            valid_chunks: 有效块缓冲区
            响应块格式:
            data: <valid_json>\n\ndata: <valid_json>\n\n...(不定长)

        Returns:
            list[dict/None]: 解析后的json列表 / None: 解析结束
        """
        chunks = chunk.splitlines()
        # 检查每部分是否data: 开头，如果不是就拼接回到前一个块
        for chunk in chunks:
            chunk = chunk.strip()
            if chunk.startswith("data: [DONE]"):
                valid_chunks.append("[DONE]")
            elif chunk.startswith("data: {"):
                valid_chunks.append(chunk.removeprefix("data: "))
            else:
                if len(valid_chunks) == 0:
                    logger.warning(f"非法数据头：{chunk}")
                else:
                    valid_chunks[-1] += " " + chunk
        jsons = []
        while valid_chunks:
            chunk = valid_chunks.pop(0)
            if chunk.startswith("[DONE]"):
                jsons.append(None)
                break
            try:
                jsons.append(json.loads(chunk))
            except json.JSONDecodeError as e:
                if len(valid_chunks) == 0:  # 数据可能被截断了, 留待下次解析
                    valid_chunks.append(chunk)
                    break
                else:  # 后面还有，这是真错了
                    logger.warning(
                        f"JSON解析失败 {e}\nchunk: {chunk}\nvalid_chunks: {valid_chunks}"
                    )
        return jsons
