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

WEB_SEARCH_CHECK_PROMPT = """You are a large-model online search assistance engine, and your task is to determine whether the user's latest news warrants a web search based on the context of the current conversation, in order to supplement the latest information that is missing from the large-model fixed knowledge base. YOU DONT NEED TO ANSWER USER's QUESTION, just judge wheather a search request should be sent and what is in it.
- If you don't think the user's latest command requires searching online content, just output "NO".
- - For simple question that existing in your database, like programing/role play/daily talk/common knowledge, just output "NO" for saving credits.
- And if you think the user's latest command really requires searching online content, analyze the user's command and output what you think are reasonable Google search terms in the following format:
``
keyword1 keyword2 ... keywordB keywordB ...
``
- - Each keyword search needs to be separated by spaces, you can use Google Engine's search syntax to specify the search target more precisely (e.g. "-" excludes unwanted keywords).
- - Multiple search requests for keywords need to be separated by a semicolon, e.g. the above example will create two search requests, respectively (search: keyword1 keyword2 ...) and (search: keywordA keywordB ...) .
- - For single searches, it directly returns something like `keywords1 keywords2 keywords3 ...', without the semicolon. `, without the semicolon
-The number of keywords per search request should be 3 to 6, and single keywords can be used for requests with strong pointers such as names of people and places.
- - You need to determine the complexity of the search, for simple questions try to use only 1 request, while for complex questions (e.g. multi-language integrated searches, multi-subject topics, etc.) you can create 2-4 search requests, not more than 5 at most.
- Finally, if the content of the user's command specifies that an online search is required, summarize the keywords according to the user's command even if you don't think you need to invoke a search.
- IMPORTANT: You can only output the keyword sequence, or "NO". You don't need to add any explanations or answer anything from the user's command, they are the tasks for another model.
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
        self.web_search_token = web_search_token
        self.web_search_max_results = web_search_max_results
        self.web_search_crawl_results = web_search_crawl_results
        self.web_search_model = web_search_model
        self.web_search_client = OpenAIClient(web_search_api_key, web_search_api_url)

    async def chat_completions_with_stream(
        self,
        messages: list,
        model_arg: dict,
        deepseek_model: str = "deepseek-reasoner",
        claude_model: str = "claude-3-5-sonnet-20241022",
        enable_web_search: bool = False,
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
        # 队列，用于传递网络搜索结果给 Claude 和 DeepSeek
        web_search_queue_1 = asyncio.Queue()
        web_search_queue_2 = asyncio.Queue()

        # 用于存储 DeepSeek 的推理累积内容
        reasoning_content = []

        async def process_web_search(messages: list):
            web_search_content = []
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
            web_search_keys = []
            ret_content = ""
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
                    ret_content += content
            web_search_keys = ret_content.strip()
            logger.info(f"检查模型返回: {web_search_keys}")

            if web_search_keys.lower() == "no":
                logger.info("不需要进行网络搜索")
            else:
                info = "**Web Searching...**\n\n"
                for key in web_search_keys.split(";"):
                    info += f"- {key}\n"
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
                                "reasoning_content": info + "\n\n",
                                "content": "",
                            },
                        }
                    ],
                }
                await output_queue.put(
                    f"data: {json.dumps(response)}\n\n".encode("utf-8")
                )
                web_search_keys = web_search_keys.split(";")
                MAX_CONCURRENT_SEARCHES = 8

                async def perform_search(search_key):
                    try:
                        logger.info(f"发起网络搜索，关键词: {search_key}")
                        async with aiohttp.ClientSession() as session:
                            async with session.post(
                                url="https://api.search1api.com/search",
                                headers={
                                    "Authorization": f"Bearer {self.web_search_token}",
                                    "Content-Type": "application/json",
                                },
                                json={
                                    "query": search_key,
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
                                    logger.error(
                                        f"网络搜索失败，状态码: {response.status}"
                                    )
                                    return None
                                else:
                                    logger.info(
                                        f"网络搜索完成，返回长度: {len(result)}"
                                    )
                                    return result
                    except Exception as e:
                        logger.error(f"网络搜索失败: {e}")
                        return None

                # 使用 asyncio.Semaphore 限制并发数
                semaphore = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)

                async def bounded_search(search_key):
                    async with semaphore:
                        return await perform_search(search_key)

                # 并发执行所有搜索
                search_tasks = [bounded_search(key.strip()) for key in web_search_keys]
                search_results = await asyncio.gather(*search_tasks)

                # 过滤掉 None 结果并添加到 web_search_content
                web_search_content.extend([r for r in search_results if r is not None])
                logger.info(
                    f"网络搜索聚合完成，总长度: {sum([len(r) for r in web_search_content])}"
                )
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
                                "reasoning_content": f"**Finished {len(web_search_content)} searches**\n\n---\n\n",
                                "content": "",
                            },
                        }
                    ],
                }
                await output_queue.put(
                    f"data: {json.dumps(response)}\n\n".encode("utf-8")
                )

            await web_search_queue_1.put(web_search_content)
            await web_search_queue_2.put(web_search_content)

        async def process_deepseek(messages: list):
            web_search_content = await web_search_queue_1.get()
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
            if web_search_content:
                messages[-1]["content"] = (
                    f"<original_prompt>\n{messages[-1]['content']}\n</original_prompt>\n\n"
                )
                for i, content in enumerate(web_search_content):
                    messages[-1]["content"] += (
                        f"<web_search_result_{i}>\n{content}\n</web_search_result_{i}>\n"
                    )
                messages[-1]["content"] += (
                    "\n\n<system>The above content are the results of online searching, both the time and content are real, which may not match your database because your database is up to an earlier time, but the current time has changed. Please think based on the search results. **Use the language of the original prompt for reasoning and answer**</system>"
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
            web_search_content = await web_search_queue_2.get()
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

                if web_search_content:
                    last_message_text = f"<original_prompt>\n{last_message_text}\n</original_prompt>\n\n"
                    for i, content in enumerate(web_search_content):
                        last_message_text += f"<web_search_result_{i}>\n{content}\n</web_search_result_{i}>\n"
                    last_message_text += "<system>The above content are the results of online searching, both the time and content are real, which may not match your database because your database is up to an earlier time, but the current time has changed. Please consider based on the search results. **Use the language of the original prompt for answer**</system>"

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

        if enable_web_search:
            asyncio.create_task(process_web_search(messages))
        else:
            await web_search_queue_1.put([])
            await web_search_queue_2.put([])

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
