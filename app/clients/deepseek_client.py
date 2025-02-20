"""DeepSeek API 客户端"""

import os
from typing import AsyncGenerator

from loguru import logger

from .base_client import BaseClient


class DeepSeekClient(BaseClient):
    def __init__(
        self,
        api_key: str,
        api_url: str = "https://api.siliconflow.cn/v1/chat/completions",
        provider: str = "deepseek",
    ):
        """初始化 DeepSeek 客户端

        Args:
            api_key: DeepSeek API密钥
            api_url: DeepSeek API地址
        """
        super().__init__(api_key, api_url)
        self.provider = provider

    def _process_think_tag_content(self, content: str) -> tuple[bool, str]:
        """处理包含 think 标签的内容

        Args:
            content: 需要处理的内容字符串

        Returns:
            tuple[bool, str]:
                bool: 是否检测到完整的 think 标签对
                str: 处理后的内容
        """
        has_start = "<think>" in content
        has_end = "</think>" in content

        if has_start and has_end:
            return True, content
        elif has_start:
            return False, content
        elif not has_start and not has_end:
            return False, content
        else:
            return True, content

    async def stream_chat(
        self,
        messages: list,
        model_arg: dict,
        model: str = "deepseek-ai/DeepSeek-R1",
    ) -> AsyncGenerator[tuple[str, str], None]:
        """流式对话

        Args:
            messages: 消息列表
            model: 模型名称
            model_arg: 模型参数
            tools: 工具列表

        Yields:
            tuple[str, str]: (内容类型, 内容)
                内容类型: "reasoning" 或 "content"
                内容: 实际的文本内容
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        data = {
            "model": model,
            "messages": messages,
            "stream": True,
            "reasoning_effort": model_arg["reasoning_effort"],
            "temperature": model_arg["temperature"],
            "top_p": model_arg["top_p"],
            "presence_penalty": model_arg["presence_penalty"],
            "frequency_penalty": model_arg["frequency_penalty"],
        }

        accumulated_content = ""
        is_collecting_think = False
        valid_chunks = []

        async for chunk in self._make_request(headers, data):
            try:
                chunk_str = chunk.decode("utf-8")
                for data in self._parse_chunk(chunk_str, valid_chunks):
                    if not data:
                        return
                    if data and data.get("choices") and data["choices"][0].get("delta"):
                        delta = data["choices"][0]["delta"]

                        if os.getenv("IS_ORIGIN_REASONING", "True").lower() == "true":
                            # 处理 reasoning_content
                            if delta.get("reasoning_content"):
                                content = delta["reasoning_content"]
                                # logger.debug(f"提取推理内容：{content}")
                                yield "reasoning", content

                            if delta.get("reasoning_content") is None and delta.get(
                                "content"
                            ):
                                content = delta["content"]
                                # logger.info(f"提取内容信息，推理阶段结束: {content}")
                                yield "content", content
                        else:
                            # 处理其他模型的输出
                            if delta.get("content"):
                                content = delta["content"]
                                if content == "":  # 只跳过完全空的字符串
                                    continue
                                # logger.debug(f"非原生推理内容：{content}")
                                accumulated_content += content

                                # 检查累积的内容是否包含完整的 think 标签对
                                is_complete, processed_content = (
                                    self._process_think_tag_content(accumulated_content)
                                )

                                if "<think>" in content and not is_collecting_think:
                                    # 开始收集推理内容
                                    logger.debug(f"开始收集推理内容：{content}")
                                    is_collecting_think = True
                                    yield (
                                        "reasoning",
                                        content.replace("<think>", ""),
                                    )
                                elif is_collecting_think:
                                    if "</think>" in content:
                                        # 推理内容结束
                                        logger.debug(f"推理内容结束：{content}")
                                        is_collecting_think = False
                                        yield (
                                            "reasoning",
                                            content.replace("</think>", ""),
                                        )
                                        yield "content", ""
                                        # 重置累积内容
                                        accumulated_content = ""
                                    else:
                                        # 继续收集推理内容
                                        yield "reasoning", content
                                else:
                                    # 普通内容
                                    yield "content", content
            except Exception as e:
                logger.error(f"处理 chunk 时发生错误: {e}")
