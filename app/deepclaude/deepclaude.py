"""DeepClaude 服务，用于协调 DeepSeek 和 Claude API 的调用"""

import asyncio
import json
import time
from copy import deepcopy
from typing import AsyncGenerator

import aiohttp

from app.clients import (
    ClaudeClient,
    DeepSeekClient,
    OpenAIClient,
)
from app.utils.logger import logger

WEB_SEARCH_CHECK_PROMPT = """
Your task: determine if the latest conversation requires a web search, based on the conversation's background and context.
Consider the following:
1) Current information such as news, events or current affairs
2) Technical or professional data to be validated
3) Recent developments in technology, science or any field
4) Statistical data or factual information that may need to be updated
5) Complex themes that benefit from multiple sources
6) Regional or cultural information that may vary by location or time of day
7) New internet slang, viral content or MEME.
If the query contains the above elements, please confirm the Google search keywords needed for the query and output them, the output can only include the keywords, don't output any other content, the keywords are separated by spaces.
Please respond "NO" if the query is common sense, or involves role-playing, fictional scenarios, or ongoing storylines, or if the information is established static information, or if the user is engaging in casual conversation, or if the context is explicitly fictional/unrelated to real-world data. Please do not output any other content.
Note: If a user is requesting up-to-date information, please use "up-to-date" directly as a search term: your knowledge base ends in 2023, but the current time has changed and you don't know the latest time, so you can't assume the current time.
Note: The language used for keywords should match the content and context of the requested query. If the query language is specified in the dialog, then you should use the language specified in the dialog to write the keywords. If not specified, then according to the dialog statement to determine: If the dialog is in Chinese, please give Chinese keywords. if the user is using English, then you should give English keywords.
Emphasize: your output includes only two cases: keywords separated by spaces, or "NO".
"""


class DeepClaude:
    """处理 DeepSeek 和 Claude API 的流式输出衔接"""

    def __init__(
        self,
        deepseek_api_key: str,
        claude_api_key: str,
        deepseek_api_url: str,
        claude_api_url: str,
        claude_provider: str,
        is_origin_reasoning: bool,
        enable_web_search: bool,
        web_search_token: str,
        web_search_max_results: int,
        web_search_crawl_results: int,
        web_search_model: str,
        web_search_api_key: str,
        web_search_api_url: str,
    ):
        """初始化 API 客户端

        Args:
            deepseek_api_key: DeepSeek API密钥
            claude_api_key: Claude API密钥
            enable_web_search: 是否启用 Web 搜索
        """
        self.deepseek_client = DeepSeekClient(deepseek_api_key, deepseek_api_url)
        if claude_provider in ["anthropic", "oneapi"]:
            self.claude_client = ClaudeClient(
                claude_api_key, claude_api_url, claude_provider
            )
        elif claude_provider == "openai":
            self.claude_client = OpenAIClient(claude_api_key, claude_api_url)
        else:
            raise ValueError(f"不支持的 Claude Provider: {claude_provider}")
        self.is_origin_reasoning = is_origin_reasoning
        self.enable_web_search = enable_web_search
        self.web_search_token = web_search_token
        self.web_search_max_results = web_search_max_results
        self.web_search_crawl_results = web_search_crawl_results
        self.web_search_model = web_search_model
        if enable_web_search:
            self.web_search_client = OpenAIClient(
                web_search_api_key, web_search_api_url
            )

    async def chat_completions_with_stream(
        self,
        messages: list,
        model_arg: dict,
        deepseek_model: str = "deepseek-reasoner",
        claude_model: str = "claude-3-5-sonnet-20241022",
    ) -> AsyncGenerator[bytes, None]:
        """处理完整的流式输出过程

        Args:
            messages: 初始消息列表
            model_arg: 模型参数
            deepseek_model: DeepSeek 模型名称
            claude_model: Claude 模型名称

        Yields:
            字节流数据，格式如下：
            {
                "id": "chatcmpl-xxx",
                "object": "chat.completion.chunk",
                "created": timestamp,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "reasoning_content": reasoning_content,
                        "content": content
                    }
                }]
            }
        """
        # 生成唯一的会话ID和时间戳
        chat_id = f"chatcmpl-{hex(int(time.time() * 1000))[2:]}"
        created_time = int(time.time())

        # 创建队列，用于收集输出数据
        output_queue = asyncio.Queue()
        # 队列，用于传递 DeepSeek 推理内容给 Claude
        claude_queue = asyncio.Queue()

        # 用于存储 DeepSeek 的推理累积内容
        reasoning_content = []
        if self.enable_web_search:
            # 检查是否需要进行网络搜索
            web_search_check_prompt = WEB_SEARCH_CHECK_PROMPT
            web_search_messages = [
                {
                    "role": "system",
                    "content": web_search_check_prompt,
                }
            ]
            web_search_messages.extend(messages)
            logger.info(
                f"开始检查是否需要进行网络搜索, 使用模型: {self.web_search_model}, 提供商: OpenAI"
            )
            web_search_keys = ""
            async for content_type, content in self.web_search_client.stream_chat(
                messages=web_search_messages,
                model=self.web_search_model,
                model_arg={
                    "max_tokens": 1000,
                    "temperature": 0.3,
                    "top_p": 1,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                },
            ):
                if content_type == "answer":
                    web_search_keys += content
            web_search_keys = web_search_keys.strip()
            logger.info(f"检查模型返回: {web_search_keys}")

            web_search_result = ""
            if web_search_keys.lower() == "no":
                logger.info("不需要进行网络搜索")
            else:
                logger.info(f"发起网络搜索，关键词: {web_search_keys}")
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            url="https://api.search1api.com/search",
                            headers={
                                "Authorization": f"Bearer {self.web_search_token}",
                                "Content-Type": "application/json",
                            },
                            json={
                                "query": web_search_keys,
                                "search_service": "google",
                                "max_results": self.web_search_max_results,
                                "crawl_results": self.web_search_crawl_results,
                                "image": False,
                            },
                        ) as response:
                            result = await response.text()
                            logger.debug(
                                f"网络搜索 返回值: {result}，状态码: {response.status}"
                            )
                            if response.status != 200:
                                logger.error(f"网络搜索失败，状态码: {response.status}")
                            else:
                                logger.info(f"网络搜索完成，返回长度: {len(result)}")
                                web_search_result = result
                except Exception as e:
                    logger.error(f"网络搜索失败: {e}")

        async def process_deepseek(messages: list):
            logger.info(
                f"开始处理 DeepSeek 流，使用模型：{deepseek_model}, 提供商: {self.deepseek_client.provider}"
            )

            for message in messages:
                if "image_url" in str(message.get("content", "")):
                    content = message["content"]
                    if not isinstance(content, list):
                        continue
                    text = ""
                    for item in content:
                        if item.get("type", "") == "text":
                            text += item.get("text", "")
                        else:
                            text += "\n<image>\n"
                    message["content"] = (
                        text
                        + "\n\n<system>The above message contains images, which have been hidden by the system because the MODEL cannot process images. You only need to assume that the images exist and think about the user's question in the language used by the user</system>"
                    )
            if web_search_result:
                messages[-1]["content"] = (
                    f"<original_prompt>\n{messages[-1]['content']}\n</original_prompt>\n\n<web_search_result>\n{web_search_result}\n</web_search_result><system>The above message is a result of an online search, both the time and content are real results, which may not match your database because your database is up to an earlier time, but the current time has changed. Please think based on the search results and use the language of the user's question</system>"
                )
            try:
                async for content_type, content in self.deepseek_client.stream_chat(
                    messages,
                    deepseek_model,
                    self.is_origin_reasoning,
                    model_arg["reasoning_effort"],
                ):
                    if content_type == "reasoning":
                        reasoning_content.append(content)
                        response = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": deepseek_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "reasoning_content": content,
                                        "content": "",
                                    },
                                }
                            ],
                        }
                        await output_queue.put(
                            f"data: {json.dumps(response)}\n\n".encode("utf-8")
                        )
                    elif content_type == "content":
                        # 当收到 content 类型时，将完整的推理内容发送到 claude_queue，并结束 DeepSeek 流处理
                        full_reasoning = "".join(reasoning_content).strip()
                        logger.info(
                            f"DeepSeek 推理完成，收集到的推理内容长度：{len(full_reasoning)}"
                        )
                        await claude_queue.put(full_reasoning)
                        break
            except Exception as e:
                logger.error(f"处理 DeepSeek 流时发生错误: {e}")
                await claude_queue.put("")
            # 用 None 标记 DeepSeek 任务结束
            logger.info("DeepSeek 任务处理完成，标记结束")
            await output_queue.put(None)

        async def process_claude(messages: list):
            try:
                logger.info("等待获取 DeepSeek 的推理内容...")
                reasoning = await claude_queue.get()
                logger.debug(
                    f"获取到推理内容，内容长度：{len(reasoning) if reasoning else 0}，内容：{reasoning}"
                )
                # 构造 Claude 的输入消息
                last_message = {}
                for message in messages[::-1]:
                    if message.get("role", "") == "user":
                        last_message = message
                        break
                last_message_text = ""
                if isinstance(last_message["content"], list):
                    for item in last_message["content"]:
                        if item.get("type", "") == "text":
                            last_message_text = item["text"]
                            break

                    def set_last_message_text(text):
                        for item in last_message:
                            if item.get("type", "") == "text":
                                item["text"] = text
                                return
                else:
                    last_message_text = last_message["content"]

                    def set_last_message_text(text):
                        last_message["content"] = text

                if web_search_result:
                    last_message_text = f"<original_prompt>\n{last_message_text}\n</original_prompt>\n\n<web_search_result>\n{web_search_result}\n<system>The above message is a network search result, and the time and content are both real results, which may not match your database because your database is up to an earlier time, but the current time has changed. Please consider based on the search results.</system>\n</web_search_result>"

                if not reasoning:
                    logger.info("推理内容为空，将使用默认提示继续")
                else:
                    if "<original_prompt>" in last_message_text:
                        last_message_text += f"\n\n<reasoning_assistant>\n{reasoning}\n</reasoning_assistant>"
                    else:
                        last_message_text = f"<original_prompt>\n{last_message_text}\n</original_prompt>\n\n<reasoning_assistant>\n{reasoning}\n</reasoning_assistant>"

                set_last_message_text(last_message_text)
                # # 处理可能 messages 内存在 role = system 的情况，如果有，则去掉当前这一条的消息对象
                # messages = [
                #     message
                #     for message in messages
                #     if message.get("role", "") != "system"
                # ]
                # logger.debug(f"Claude 的输入消息: {messages}")

                logger.info(
                    f"开始处理 Claude 流，使用模型: {claude_model}, 提供商: {self.claude_client.provider}"
                )

                async for content_type, content in self.claude_client.stream_chat(
                    messages=messages,
                    model_arg=model_arg,
                    model=claude_model,
                ):
                    if content_type == "answer":
                        response = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": claude_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": content},
                                }
                            ],
                        }
                        await output_queue.put(
                            f"data: {json.dumps(response)}\n\n".encode("utf-8")
                        )
            except Exception as e:
                logger.error(f"处理 Claude 流时发生错误: {e}")
            # 用 None 标记 Claude 任务结束
            logger.info("Claude 任务处理完成，标记结束")
            await output_queue.put(None)

        # 创建并发任务

        deepseek_messages = deepcopy(messages)
        claude_messages = deepcopy(messages)

        asyncio.create_task(process_deepseek(deepseek_messages))
        asyncio.create_task(process_claude(claude_messages))

        # 等待两个任务完成，通过计数判断
        finished_tasks = 0
        while finished_tasks < 2:
            item = await output_queue.get()
            if item is None:
                finished_tasks += 1
            else:
                yield item

        # 发送结束标记
        yield b"data: [DONE]\n\n"
