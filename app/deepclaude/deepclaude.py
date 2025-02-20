"""DeepClaude 服务，用于协调 DeepSeek 和 Claude API 的调用"""

import asyncio
import datetime
import json
import os
import time
import uuid
from copy import deepcopy
from typing import AsyncGenerator

import aiohttp
from app.clients import DeepSeekClient, OpenAIClient
from fastapi import HTTPException
from loguru import logger

WEB_SEARCH_CHECK_PROMPT = """You are a LLM online search assistance engine, and your task is to determine whether the user's latest prompt warrants a web search based on the context of the current conversation, in order to supplement the latest information that is missing from the LLM trained knowledge base. You should answer referring to the following rules:
- YOU DONT NEED TO ANSWER USER's QUESTION, just judge wheather a search request should be sent and what is in it.
- If you don't think the user's latest command requires searching online content, just output "NO".
- - For simple question that existing in your database, like programing/role play/daily talk/common knowledge etc., just output "NO" for saving credits.
- And if you think the user's latest command really requires searching online content, analyze the user's command and output what you think are reasonable Google search terms in the following format:
keyword1 keyword2 ...;keywordB keywordB ...
- - Each keyword search needs to be separated by spaces, you can use Google Engine's search syntax to specify the search target more precisely (e.g. "-" excludes unwanted keywords).
- - Multiple search requests for keywords need to be separated by a semicolon, e.g. the above example will create two search requests, respectively (search: keyword1 keyword2 ...) and (search: keywordA keywordB ...) .
- - For single searches, it directly returns something like keywords1 keywords2 keywords3 ..., without the semicolon.
-The number of keywords per search request should be 3 to 6, and single keywords can be used for requests with strong pointers such as names of people and places.
- - You need to determine the complexity of the search, for simple questions try to use only 1 or 2 request(s) (2 is best), while for complex questions (e.g. multi-language integrated searches, multi-subject topics, etc.) you can create 3-5 search requests, not more than 4 at most.
- Finally, if the user's command specifies that an online search is required, just do it even if you don't think it's necessary.
- IMPORTANT: You can only output the keyword sequence, or "NO". You don't need to add any explanations or answer anything from the user's command, they are the tasks for another model.
- About "latest": The time now is {year}-{month}-{day} {hour}:{minute}:{second} (UTC+0), if the user request latest news, just search based on this time.
"""


def gen_web_search_check_prompt():
    now = datetime.datetime.now(datetime.UTC)
    return WEB_SEARCH_CHECK_PROMPT.format(
        year=now.year,
        month=now.month,
        day=now.day,
        hour=now.hour,
        minute=now.minute,
        second=now.second,
    )


DEEPSEEK_REASONING_PROMPT = """\n\n<system_notice>你是一个大模型前置思考辅助引擎，负责对用户的问题进行深入、具体、全面的思考，为后续其他大模型的最终回答提供思路和判断依据，你不需要回答用户的问题，只需要对用户的每个最新问题进行思考，最终回答的任务交由其他大模型负责。</system_notice>
"""

IMAGE_NOTICE_PROMPT = """\n\n<system_notice>The message below contains images, which have been hidden by the system because the MODEL cannot process images. You only need to assume that the images exist and think about the user's question in the language used by the user</system_notice>\n\n"""

TOOLCALL_NOTICE_PROMPT = """\n\n<system_notice>The message below is result returned by the tool calls, which are not visible to the user. You only need to assume that the tool calls finished and continue thinking about the user's last question in the language used by the user based on the results of the tool calls.</system_notice>\n\n"""


def build_original_prompt(message: str):
    return f'<original_prompt description="the messages sent by the user are as below">\n{message}\n</original_prompt>\n\n'


def build_web_search_prompt(contents: str):
    now = datetime.datetime.now(datetime.UTC)
    c = f'<web_search_results description="The jsons below are the results returned by web searching tools, both the time and content are real, which may not match your database because your database is up to an earlier time, but the time now is {now.year}-{now.month}-{now.day} {now.hour}:{now.minute}:{now.second} (UTC+0). Please think based on the search results. When you are using the content from searching, cite the original source url using the following format: [[CITE_NO]](SOURCE_URL) like this: [[1]](https://www.google.com), starting from 1. **Use the language of the original prompt for reasoning and answer**">\n'
    for i, content in enumerate(contents):
        c += f"<result id={i}>\n{content}\n</result>\n"
    c += "</web_search_results>\n\n"
    return c


def build_reasoning_prompt(reasoning: str):
    return f'<reasoning description="The content below is from the reasoning assistant, which is the reasoning process for the message from the user, aiming to assist you to manage your answer, and which is not visible to the user. You should answer based on this reasoning">\n{reasoning}\n</reasoning>\n\n'


def build_system_message(message: str):
    return {"role": "system", "content": message}


async def web_search(search_keys, max_concurrent_searches: int = 8):
    async def perform_search(search_key):
        try:
            logger.info(f"发起网络搜索，关键词: {search_key}")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url="https://api.search1api.com/search",
                    headers={
                        "Authorization": f"Bearer {os.getenv('WEB_SEARCH_TOKEN')}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "query": search_key,
                        "search_service": os.getenv("WEB_SEARCH_SERVICE"),
                        "max_results": os.getenv("WEB_SEARCH_MAX_RESULTS"),
                        "crawl_results": os.getenv("WEB_SEARCH_CRAWL_RESULTS"),
                        "image": False,
                    },
                ) as response:
                    result = await response.text()
                    logger.debug(
                        f"网络搜索 {search_key} 返回值: {result}，状态码: {response.status}"
                    )
                    if response.status != 200:
                        logger.error(
                            f"网络搜索 {search_key} 失败，状态码: {response.status}"
                        )
                        return None
                    else:
                        logger.info(
                            f"网络搜索 {search_key} 完成，返回长度: {len(result)}"
                        )
                        return result
        except Exception as e:
            logger.error(f"网络搜索 {search_key} 失败: {e}")
            return None

    # 使用 asyncio.Semaphore 限制并发数
    semaphore = asyncio.Semaphore(max_concurrent_searches)

    async def bounded_search(search_key):
        async with semaphore:
            return await perform_search(search_key)

    # 并发执行所有搜索
    search_tasks = [bounded_search(key.strip()) for key in search_keys]
    search_results = await asyncio.gather(*search_tasks)
    search_results = [r for r in search_results if r is not None]
    logger.info(
        f"网络搜索聚合完成，有效结果数量: {len(search_results)}, 内容长度: {sum([len(r) for r in search_results])}"
    )
    return search_results


class DeepClaude:
    def __init__(self):
        """初始化 API 客户端"""
        self.reasoning_client = DeepSeekClient(
            os.getenv("REASONING_API_KEY"), os.getenv("REASONING_API_URL")
        )
        self.answering_client = OpenAIClient(
            os.getenv("ANSWERING_API_KEY"), os.getenv("ANSWERING_API_URL")
        )
        if "deepseek" in os.getenv("WEB_SEARCH_MODEL"):
            self.web_search_client = DeepSeekClient(
                os.getenv("WEB_SEARCH_API_KEY"), os.getenv("WEB_SEARCH_API_URL")
            )
        else:
            self.web_search_client = OpenAIClient(
                os.getenv("WEB_SEARCH_API_KEY"), os.getenv("WEB_SEARCH_API_URL")
            )
        logger.info("DeepClaude 初始化完成")

    async def chat_completions_with_stream(
        self,
        messages: list,
        tools: list,
        model_arg: dict,
        reasoning_model: str,
        answering_model: str,
        enable_web_search: bool = False,
        enable_reasoning: bool = True,
        enable_answering: bool = True,
    ) -> AsyncGenerator[bytes, None]:
        """处理完整的流式输出过程

        Args:
            messages: 初始消息列表
            model_arg: 模型参数
            reasoning_model: 推理模型名称
            answering_model: 回答模型名称
            enable_web_search: 是否启用 Web 搜索
            enable_reasoning: 是否启用推理模型
            enable_answering: 是否启用回答模型回答

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

        assert enable_reasoning or enable_answering  # 总得有事干吧
        # 生成唯一的会话ID和时间戳
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created_time = int(time.time())

        # 创建队列，用于收集输出数据
        output_queue = asyncio.Queue()
        # 队列，用于传递 DeepSeek 推理内容给回答模型
        reasoning_queue = asyncio.Queue()
        # 队列，用于传递网络搜索结果给推理模型和回答模型
        web_search_queue_1 = asyncio.Queue()
        web_search_queue_2 = asyncio.Queue()

        # 用于存储 DeepSeek 的推理累积内容
        reasoning_content = []

        logger.info(f"[{chat_id}] 对话请求接受，开始处理")

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
            web_search_check_prompt = gen_web_search_check_prompt()
            web_search_messages = [
                {
                    "role": "system",
                    "content": web_search_check_prompt,
                }
            ]
            web_search_messages.extend(messages)
            logger.info(
                f"[{chat_id}] 开始检查是否需要进行网络搜索, 使用模型: {os.getenv('WEB_SEARCH_MODEL')}, 提供商: {self.web_search_client.provider}"
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
                    "reasoning_effort": "medium",
                },
            ):
                if content_type == "content":
                    ret_content += content
            web_search_keys = ret_content.strip().replace("\n", " ").replace("`", "")
            web_search_keys = " ".join(web_search_keys.split())
            logger.info(f"[{chat_id}] 检查模型返回: {web_search_keys}")

            if web_search_keys.lower() == "no" or not web_search_keys.strip():
                logger.info(f"[{chat_id}] 不需要进行网络搜索")
            else:
                await send_reasoning_response("**Web Searching...**\n\n")
                web_search_keys = [key.strip() for key in web_search_keys.split(";")]
                for key in web_search_keys:
                    await send_reasoning_response(f"- `{key}`\n")
                search_results = await web_search(web_search_keys)
                web_search_content.extend(search_results)
                await send_reasoning_response(
                    f"\n\n**Finished {len(web_search_content)} searches**\n\n---\n\n"
                )

            logger.info(f"[{chat_id}] 网络搜索任务处理完成")
            await web_search_queue_1.put(web_search_content)
            await web_search_queue_2.put(web_search_content)
            await output_queue.put(None)  # 标记网络搜索任务结束

        async def process_reasoning(messages: list):
            new_messages = []
            if enable_answering:
                new_messages.append(build_system_message(DEEPSEEK_REASONING_PROMPT))
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
                    message["content"] = IMAGE_NOTICE_PROMPT + text

                if message.get("role", "") == "tool":
                    message["content"] = TOOLCALL_NOTICE_PROMPT + message["content"]
                    message["role"] = "user"

                new_message = {
                    "content": message["content"],
                    "role": message["role"],
                }
                new_messages.append(new_message)

            messages = new_messages

            if enable_web_search:
                web_search_content = await web_search_queue_1.get()
                if web_search_content:
                    messages[-1]["content"] = build_original_prompt(
                        messages[-1]["content"]
                    ) + build_web_search_prompt(web_search_content)

            logger.info(
                f"[{chat_id}] 开始处理推理模型流，使用模型：{reasoning_model}, 提供商: {self.reasoning_client.provider}"
            )
            try:
                async for content_type, content in self.reasoning_client.stream_chat(
                    messages,
                    model_arg=model_arg,
                    model=reasoning_model,
                ):
                    if content_type == "reasoning" and content.strip():
                        reasoning_content.append(content)
                        await send_reasoning_response(content)
                    elif content_type == "content":
                        if enable_answering:
                            full_reasoning = "".join(reasoning_content).strip()
                            logger.info(
                                f"[{chat_id}] 推理模型推理完成，收集到的推理内容长度：{len(full_reasoning)}"
                            )
                            logger.debug(f"[{chat_id}] 推理内容：{full_reasoning}")
                            await reasoning_queue.put(full_reasoning)
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
                logger.error(f"[{chat_id}] 处理推理模型流时发生错误: {e}")
                await reasoning_queue.put("")
                raise HTTPException(
                    status_code=500, detail=f"处理推理模型流时发生错误: {e}"
                )
            logger.info(f"[{chat_id}] 推理模型任务处理完成")
            await output_queue.put(None)  # 用 None 标记推理模型任务结束

        async def process_answering(messages: list):
            try:
                if enable_web_search:
                    web_search_content = await web_search_queue_2.get()
                    if web_search_content:
                        messages.append(
                            build_system_message(
                                build_web_search_prompt(web_search_content)
                            )
                        )
                if enable_reasoning:
                    reasoning = await reasoning_queue.get()
                    if reasoning:
                        messages.append(
                            build_system_message(build_reasoning_prompt(reasoning))
                        )

                logger.info(
                    f"[{chat_id}] 开始处理回答模型流，使用模型: {answering_model}, 提供商: {self.answering_client.provider}"
                )

                async for content_type, content in self.answering_client.stream_chat(
                    messages=messages,
                    model_arg=model_arg,
                    model=answering_model,
                    tools=tools,
                ):
                    if content_type == "content":
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
                    elif content_type == "tool_calls":
                        response = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": answering_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": "",
                                        "tool_calls": content,
                                    },
                                }
                            ],
                        }
                    else:
                        continue
                    await output_queue.put(
                        f"data: {json.dumps(response)}\n\n".encode("utf-8")
                    )

            except Exception as e:
                logger.error(f"[{chat_id}] 处理回答模型流时发生错误: {e}")
                raise HTTPException(
                    status_code=500, detail=f"处理回答模型流时发生错误: {e}"
                )
            logger.info(f"[{chat_id}] 回答模型任务处理完成")
            await output_queue.put(None)  # 标记回答模型任务结束

        # 创建并发任务

        if enable_web_search:
            asyncio.create_task(process_web_search(deepcopy(messages)))
        else:
            await web_search_queue_1.put([])
            await web_search_queue_2.put([])

        if enable_reasoning:
            asyncio.create_task(process_reasoning(deepcopy(messages)))
        else:
            await reasoning_queue.put("")

        if enable_answering:
            asyncio.create_task(process_answering(deepcopy(messages)))

        # 等待任务完成，通过计数判断
        finished_tasks = 0
        while finished_tasks < (
            int(enable_web_search) + int(enable_reasoning) + int(enable_answering)
        ):
            item = await output_queue.get()
            if item is None:
                finished_tasks += 1
            else:
                yield item

        logger.info(f"[{chat_id}] 对话请求处理完成")
        # 发送结束标记
        yield b"data: [DONE]\n\n"
