"""OpenAI API Client"""

import json
from typing import AsyncGenerator

from loguru import logger

from .base_client import BaseClient


class OpenAIClient(BaseClient):
    def __init__(
        self,
        api_key: str,
        api_url: str = "https://api.openai.com/v1/chat/completions",
        provider: str = "OpenAI",
    ):
        """Initialize OpenAI client

        Args:
            api_key: OpenAI API key
            api_url: OpenAI API base URL
        """
        self.provider = provider
        super().__init__(api_key, api_url)

    async def stream_chat(
        self,
        messages: list,
        model_arg: dict,
        model: str = "gpt-3.5-turbo",
        tools: list = [],
    ) -> AsyncGenerator[tuple[str, str], None]:
        """Stream chat with OpenAI API

        Args:
            messages: List of messages
            model_arg: Model parameters dict [temperature, top_p, presence_penalty, frequency_penalty]
            model: Model name
            tools: tools list
            stream: Whether to use streaming output

        Yields:
            tuple[str, str]: (content_type, content)
                content_type: "answer" or "tool_calls"
                content: Actual text content
                tool_calls: Actual tool calls content in list
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": 1
            if model_arg["temperature"] < 0 or model_arg["temperature"] > 1
            else model_arg["temperature"],
            "top_p": model_arg["top_p"],
            "presence_penalty": model_arg["presence_penalty"],
            "frequency_penalty": model_arg["frequency_penalty"],
        }

        if tools:
            data["tools"] = tools

        logger.info("开始流式对话")

        async for chunk in self._make_request(headers, data):
            chunk_str = chunk.decode("utf-8")
            if not chunk_str.strip():
                continue

            for line in chunk_str.split("\n"):
                if line.startswith("data: "):
                    json_str = line[6:]  # Remove 'data: ' prefix
                    if json_str.strip() == "[DONE]":
                        return

                    try:
                        data = json.loads(json_str)
                        content = (
                            data.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        tool_calls = (
                            data.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("tool_calls", [])
                        )
                        if content:
                            yield "answer", content
                        if tool_calls:
                            yield "tool_calls", tool_calls
                    except json.JSONDecodeError:
                        continue
