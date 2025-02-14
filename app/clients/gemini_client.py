"""Gemini API Client"""

import json
from typing import AsyncGenerator

from app.utils.logger import logger

from .base_client import BaseClient


class GeminiClient(BaseClient):
    def __init__(
        self,
        api_key: str,
        api_url: str = "https://generativelanguage.googleapis.com",
        provider: str = "google",
    ):
        """Initialize Gemini client

        Args:
            api_key: Google API key
            api_url: Gemini API base URL
            provider: API provider name
        """
        super().__init__(api_key, api_url)
        self.provider = provider
        self.base_url = api_url

    async def stream_chat(
        self,
        messages: list,
        model_arg: dict,
        model: str = "gemini-pro",
        stream: bool = True,
    ) -> AsyncGenerator[tuple[str, str], None]:
        """Stream chat with Gemini API

        Args:
            messages: List of messages
            model_arg: Model parameters dict [temperature, top_p, presence_penalty, frequency_penalty]
                Note: Gemini only supports temperature and top_p
            model: Model name (currently only supports gemini-pro)
            stream: Whether to use streaming output

        Yields:
            tuple[str, str]: (content_type, content)
                content_type: "answer"
                content: Actual text content
        """
        # Convert messages to Gemini format
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Map roles to Gemini format
            if role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})
            else:
                contents.append({"role": "user", "parts": [{"text": content}]})

        headers = {
            "Content-Type": "application/json",
        }

        # Construct full API URL with model and API key
        self.api_url = (
            f"{self.base_url}/v1/models/{model}:generateContent?key={self.api_key}"
        )

        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": 1
                if model_arg["temperature"] < 0 or model_arg["temperature"] > 1
                else model_arg["temperature"],
                "topP": model_arg["top_p"],
                "topK": 1,
            },
            "stream": stream,
        }

        logger.debug(f"Starting chat: {data}")

        if stream:
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
                                data.get("candidates", [{}])[0]
                                .get("content", {})
                                .get("parts", [{}])[0]
                                .get("text", "")
                            )
                            if content:
                                yield "answer", content
                        except json.JSONDecodeError:
                            continue
        else:
            # Non-streaming output
            async for chunk in self._make_request(headers, data):
                try:
                    response = json.loads(chunk.decode("utf-8"))
                    content = (
                        response.get("candidates", [{}])[0]
                        .get("content", {})
                        .get("parts", [{}])[0]
                        .get("text", "")
                    )
                    if content:
                        yield "answer", content
                except json.JSONDecodeError:
                    continue
