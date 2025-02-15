"""DeepClaude 服务，用于协调 DeepSeek 和 Claude API 的调用"""

import asyncio
import json
import os
import time
from copy import deepcopy
from typing import AsyncGenerator

import aiohttp
from loguru import logger

from app.clients import DeepSeekClient, OpenAIClient

WEB_SEARCH_CHECK_PROMPT = """You are a large-model online search assistance engine, and your task is to determine whether the user's latest prompt warrants a web search based on the context of the current conversation, in order to supplement the latest information that is missing from the large-model fixed knowledge base.
- YOU DONT NEED TO ANSWER USER's QUESTION, just judge wheather a search request should be sent and what is in it.
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
- About "latest": The latest time now is far away from year 2023, when user request latest news, just search without time that based on your knowledge.
- IMPORTANT: You can only output the keyword sequence, or "NO". You don't need to add any explanations or answer anything from the user's command, they are the tasks for another model.
"""


class DeepClaude:
    def __init__(self):
        """初始化 API 客户端"""
        self.reasoning_client = DeepSeekClient(
            os.getenv("REASONING_API_KEY"), os.getenv("REASONING_API_URL")
        )
        self.answering_client = OpenAIClient(
            os.getenv("ANSWERING_API_KEY"), os.getenv("ANSWERING_API_URL")
        )
        self.web_search_client = OpenAIClient(
            os.getenv("WEB_SEARCH_API_KEY"), os.getenv("WEB_SEARCH_API_URL")
        )
        logger.info("DeepClaude 初始化完成")

    async def chat_completions_with_stream(
        self,
        messages: list,
        model_arg: dict,
        reasoning_model: str,
        answering_model: str,
        enable_web_search: bool = False,
        enable_answering: bool = True,
    ) -> AsyncGenerator[bytes, None]:
        """处理完整的流式输出过程

        Args:
            messages: 初始消息列表
            model_arg: 模型参数
            reasoning_model: 推理模型名称
            answering_model: 回答模型名称
            enable_web_search: 是否启用 Web 搜索
            enable_answering: 是否启用二级模型回答

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
        # 队列，用于传递 DeepSeek 推理内容给二级模型
        answering_queue = asyncio.Queue()
        # 队列，用于传递网络搜索结果给一级模型和二级模型
        web_search_queue_1 = asyncio.Queue()
        web_search_queue_2 = asyncio.Queue()

        # 用于存储 DeepSeek 的推理累积内容
        reasoning_content = []

        async def send_reasoning_response(reasoning_content: str):
            response = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": reasoning_model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "reasoning_content": reasoning_content,
                            "content": "",
                        },
                    }
                ],
            }
            await output_queue.put(f"data: {json.dumps(response)}\n\n".encode("utf-8"))

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
                f"开始检查是否需要进行网络搜索, 使用模型: {os.getenv('WEB_SEARCH_MODEL')}, 提供商: OpenAI"
            )
            web_search_keys = []
            ret_content = ""
            async for content_type, content in self.web_search_client.stream_chat(
                messages=web_search_messages,
                model=os.getenv("WEB_SEARCH_MODEL"),
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
            web_search_keys = ret_content.strip().replace("\n", " ")
            web_search_keys = " ".join(web_search_keys.split())
            logger.info(f"检查模型返回: {web_search_keys}")

            if web_search_keys.lower() == "no":
                logger.info("不需要进行网络搜索")
            else:
                await send_reasoning_response("**Web Searching...**\n\n")
                web_search_keys = web_search_keys.split(";")
                MAX_CONCURRENT_SEARCHES = 8

                async def perform_search(search_key):
                    try:
                        logger.info(f"发起网络搜索，关键词: {search_key}")
                        await send_reasoning_response(f"- {search_key}\n")
                        async with aiohttp.ClientSession() as session:
                            async with session.post(
                                url="https://api.search1api.com/search",
                                headers={
                                    "Authorization": f"Bearer {os.getenv('WEB_SEARCH_TOKEN')}",
                                    "Content-Type": "application/json",
                                },
                                json={
                                    "query": search_key,
                                    "search_service": "google",
                                    "max_results": os.getenv("WEB_SEARCH_MAX_RESULTS"),
                                    "crawl_results": os.getenv(
                                        "WEB_SEARCH_CRAWL_RESULTS"
                                    ),
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
                await send_reasoning_response(
                    f"\n\n**Finished {len(web_search_content)} searches**\n\n---\n\n"
                )

            await web_search_queue_1.put(web_search_content)
            await web_search_queue_2.put(web_search_content)

        async def process_reasoning(messages: list):
            web_search_content = await web_search_queue_1.get()
            logger.info(
                f"开始处理一级模型流，使用模型：{reasoning_model}, 提供商: {self.reasoning_client.provider}"
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
                async for content_type, content in self.reasoning_client.stream_chat(
                    messages,
                    model_arg=model_arg,
                    model=reasoning_model,
                ):
                    if content_type == "reasoning":
                        reasoning_content.append(content)
                        await send_reasoning_response(content)
                    elif content_type == "content":
                        if enable_answering:
                            # 当收到 content 类型时，将完整的推理内容发送到 claude_queue，并结束 DeepSeek 流处理
                            full_reasoning = "".join(reasoning_content).strip()
                            logger.info(
                                f"一级模型推理完成，收集到的推理内容长度：{len(full_reasoning)}"
                            )
                            await answering_queue.put(full_reasoning)
                            break
                        else:
                            response = {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": reasoning_model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "role": "assistant",
                                            "content": content,
                                        },
                                    }
                                ],
                            }
                            await output_queue.put(
                                f"data: {json.dumps(response)}\n\n".encode("utf-8")
                            )
            except Exception as e:
                logger.error(f"处理一级模型流时发生错误: {e}")
                await answering_queue.put("")
            # 用 None 标记一级模型任务结束
            logger.info("一级模型任务处理完成，标记结束")
            await output_queue.put(None)

        async def process_answering(messages: list):
            web_search_content = await web_search_queue_2.get()
            try:
                logger.info("等待获取一级模型的推理内容...")
                reasoning = await answering_queue.get()
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

                logger.info(
                    f"开始处理二级模型流，使用模型: {answering_model}, 提供商: {self.answering_client.provider}"
                )

                async for content_type, content in self.answering_client.stream_chat(
                    messages=messages,
                    model_arg=model_arg,
                    model=answering_model,
                ):
                    if content_type == "answer":
                        response = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": answering_model,
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
                logger.error(f"处理二级模型流时发生错误: {e}")
            # 用 None 标记二级模型任务结束
            logger.info("二级模型任务处理完成，标记结束")
            await output_queue.put(None)

        # 创建并发任务

        if enable_web_search:
            asyncio.create_task(process_web_search(messages))
        else:
            await web_search_queue_1.put([])
            await web_search_queue_2.put([])

        reasoning_messages = deepcopy(messages)
        answering_messages = deepcopy(messages)
        asyncio.create_task(process_reasoning(reasoning_messages))
        if enable_answering:
            asyncio.create_task(process_answering(answering_messages))

        # 等待任务完成，通过计数判断
        finished_tasks = 0
        while finished_tasks < (2 if enable_answering else 1):
            item = await output_queue.get()
            if item is None:
                finished_tasks += 1
            else:
                yield item

        # 发送结束标记
        yield b"data: [DONE]\n\n"
