import os
import sys
from copy import deepcopy

import aiohttp
from dotenv import load_dotenv
from loguru import logger

# 加载 .env 文件
logger.info(f"当前工作目录: {os.getcwd()}")
logger.info("尝试加载.env文件...")
load_dotenv(override=True)  # 添加override=True强制覆盖已存在的环境变量

logger.remove()
if os.environ.get("NO_STDERR").lower() != "true":
    logger.add(sys.stderr, level=os.getenv("LOG_LEVEL"))

logger.add(
    os.getcwd() + "/logs/{time:YYYY-MM-DD}.log",
    level=os.getenv("LOG_LEVEL"),
    rotation="1 day",
    retention=10,
)


from fastapi import Depends, FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402

from app.deepclaude.auth import verify_api_key  # noqa: E402
from app.deepclaude.deepclaude import DeepClaude  # noqa: E402

app = FastAPI(title="DeepClaude API")

# CORS设置
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")
allow_origins_list = (
    ALLOW_ORIGINS.split(",") if ALLOW_ORIGINS else []
)  # 将逗号分隔的字符串转换为列表

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

deep_claude = DeepClaude()

# 验证日志级别
logger.debug(f"当前日志级别为 {os.getenv('LOG_LEVEL')}")
logger.info("开始请求")


@app.get("/", dependencies=[Depends(verify_api_key)])
async def root():
    logger.info("访问了根路径")
    return {"message": "Welcome to DeepClaude API"}


@app.get("/v1/models")
async def list_models():
    """
    获取可用模型列表
    返回格式遵循 OpenAI API 标准
    """

    def build_model(id: str):
        return {
            "id": id,
            "object": "model",
            "created": 1677610602,
            "owned_by": "deepclaude",
        }

    # 尝试请求二级api获取二级模型列表
    PREFIX_WHITELIST = ["claude", "gpt", "o1", "o3", "gemini"]
    models = [
        build_model("deepclaude"),
        build_model("deepclaude-net"),
        build_model("deepseek-net"),
    ]
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                os.getenv("MODEL_LIST_API_URL"),
                headers={
                    "Authorization": f"Bearer {os.getenv('MODEL_LIST_API_KEY')}",
                },
            ) as response:
                assert response.status == 200
                ret = await response.json()
                models_secondary = [
                    m
                    for m in ret["data"]
                    if any(m["id"].startswith(prefix) for prefix in PREFIX_WHITELIST)
                ]
                # logger.debug(f"二级模型列表: {models_secondary}")
            for model in models_secondary:
                _ = deepcopy(model)
                _["id"] = _["id"] + "-deep"
                models.append(_)
                _ = deepcopy(model)
                _["id"] = _["id"] + "-net"
                models.append(_)
                _ = deepcopy(model)
                _["id"] = _["id"] + "-deepnet"
                models.append(_)
    except Exception as e:
        logger.error(f"获取二级模型列表失败: {e}")

    # logger.debug(f"返回模型列表: {models}")
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: Request):
    """处理聊天完成请求，支持流式和非流式输出

    请求体格式应与 OpenAI API 保持一致，包含：
    - messages: 消息列表
    - model: 模型名称（可选）
    - stream: 是否使用流式输出（可选，默认为 True)
    - temperature: 随机性 (可选)
    - top_p: top_p (可选)
    - presence_penalty: 话题新鲜度（可选）
    - frequency_penalty: 频率惩罚度（可选）
    - reasoning_effort: 推理努力度（可选）
    """

    try:
        # 1. 获取基础信息
        body = await request.json()
        logger.debug(f"请求体: {body}，请求头: {request.headers}")
        messages = body.get("messages")
        tools = body.get("tools", [])

        # 2. 获取并验证参数
        model_arg = get_and_validate_params(body)
        logger.debug(f"解析模型参数: {model_arg}")

        stream = model_arg["stream"]  # 获取 stream 参数

        # 3. 根据 stream 参数返回相应的响应
        if stream:
            return StreamingResponse(
                deep_claude.chat_completions_with_stream(
                    messages=messages,
                    tools=tools,
                    model_arg=model_arg,
                    reasoning_model=model_arg["reasoning_model"],
                    answering_model=model_arg["answering_model"],
                    enable_web_search=model_arg["enable_web_search"],
                    enable_answering=model_arg["enable_answering"],
                    enable_reasoning=model_arg["enable_reasoning"],
                ),
                media_type="text/event-stream",
            )
        else:
            raise HTTPException(status_code=400, detail="Only streaming is supported")

    except Exception as e:
        logger.error(f"处理请求时发生错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def get_and_validate_params(body):
    """提取获取和验证请求参数的函数"""
    temperature: float = body.get("temperature", 0.6)
    top_p: float = body.get("top_p", 0.9)
    presence_penalty: float = body.get("presence_penalty", 0.0)
    frequency_penalty: float = body.get("frequency_penalty", 0.0)
    stream: bool = body.get("stream", True)
    reasoning_effort: str = body.get("reasoning_effort", "medium")
    model: str = body.get("model", "")

    if "sonnet" in model:  # Only Sonnet 设定 temperature 必须在 0 到 1 之间
        if (
            not isinstance(temperature, (float))
            or temperature < 0.0
            or temperature > 1.0
        ):
            raise ValueError("Sonnet 设定 temperature 必须在 0 到 1 之间")

    enable_web_search = {
        "deepclaude": False,
        "deepclaude-net": True,
        "deepseek-net": True,
    }.get(model, False)
    enable_answering = {
        "deepclaude": True,
        "deepclaude-net": True,
        "deepseek-net": False,
    }.get(model, False)
    enable_reasoning = {
        "deepclaude": True,
        "deepclaude-net": True,
        "deepseek-net": True,
    }.get(model, False)

    if model not in ["deepclaude", "deepclaude-net", "deepseek-net"]:
        if model.endswith("-deepnet"):
            model = model.removesuffix("-deepnet")
            enable_reasoning = True
            enable_answering = True
            enable_web_search = True
        elif model.endswith("-deep"):
            model = model.removesuffix("-deep")
            enable_reasoning = True
            enable_answering = True
            enable_web_search = False
        elif model.endswith("-net"):
            model = model.removesuffix("-net")
            enable_reasoning = False
            enable_answering = True
            enable_web_search = True

    if not any([enable_reasoning, enable_answering, enable_web_search]):
        raise HTTPException(status_code=400, detail="Invalid model name")

    reasoning_model = os.getenv("REASONING_MODEL")
    answering_model = (
        os.getenv("ANSWERING_MODEL")
        if model in ["deepclaude", "deepclaude-net", "deepseek-net"]
        else model
    )

    return {
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "reasoning_effort": reasoning_effort,
        "stream": stream,
        "model": model,
        "enable_reasoning": enable_reasoning,
        "enable_web_search": enable_web_search,
        "enable_answering": enable_answering,
        "reasoning_model": reasoning_model,
        "answering_model": answering_model,
    }
