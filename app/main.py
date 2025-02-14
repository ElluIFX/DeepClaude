import os
import sys

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from fastapi import Depends, FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402

from app.deepclaude.deepclaude import DeepClaude  # noqa: E402
from app.utils.auth import verify_api_key  # noqa: E402
from app.utils.logger import logger  # noqa: E402

app = FastAPI(title="DeepClaude API")

# 从环境变量获取 CORS配置, API 密钥、地址以及模型名称
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL")
CLAUDE_PROVIDER = os.getenv("CLAUDE_PROVIDER")
CLAUDE_API_URL = os.getenv("CLAUDE_API_URL")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL")

IS_ORIGIN_REASONING = os.getenv("IS_ORIGIN_REASONING", "True").lower() == "true"

ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "False").lower() == "true"
WEB_SEARCH_MODEL = os.getenv("WEB_SEARCH_MODEL")
WEB_SEARCH_API_KEY = os.getenv("WEB_SEARCH_API_KEY")
WEB_SEARCH_API_URL = os.getenv("WEB_SEARCH_API_URL")
WEB_SEARCH_TOKEN = os.getenv("WEB_SEARCH_TOKEN")
WEB_SEARCH_MAX_RESULTS = os.getenv("WEB_SEARCH_MAX_RESULTS", 12)
WEB_SEARCH_CRAWL_RESULTS = os.getenv("WEB_SEARCH_CRAWL_RESULTS", 4)
if not WEB_SEARCH_TOKEN:
    logger.warning("未设置 WEB_SEARCH_TOKEN，将禁用 Web 搜索")
    ENABLE_WEB_SEARCH = False

# CORS设置
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

# 创建 DeepClaude 实例, 提出为Global变量
if not DEEPSEEK_API_KEY or not CLAUDE_API_KEY:
    logger.critical("请设置环境变量 CLAUDE_API_KEY 和 DEEPSEEK_API_KEY")
    sys.exit(1)

deep_claude = DeepClaude(
    DEEPSEEK_API_KEY,
    CLAUDE_API_KEY,
    DEEPSEEK_API_URL,
    CLAUDE_API_URL,
    CLAUDE_PROVIDER,
    IS_ORIGIN_REASONING,
    ENABLE_WEB_SEARCH,
    WEB_SEARCH_TOKEN,
    WEB_SEARCH_MAX_RESULTS,
    WEB_SEARCH_CRAWL_RESULTS,
    WEB_SEARCH_MODEL,
    WEB_SEARCH_API_KEY,
    WEB_SEARCH_API_URL,
)

# 验证日志级别
logger.debug("当前日志级别为 DEBUG")
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
    models = [
        {
            "id": "deepclaude",
            "object": "model",
            "created": 1677610602,
            "owned_by": "deepclaude",
            "permission": [
                {
                    "id": "modelperm-deepclaude",
                    "object": "model_permission",
                    "created": 1677610602,
                    "allow_create_engine": False,
                    "allow_sampling": True,
                    "allow_logprobs": True,
                    "allow_search_indices": False,
                    "allow_view": True,
                    "allow_fine_tuning": False,
                    "organization": "*",
                    "group": None,
                    "is_blocking": False,
                }
            ],
            "root": "deepclaude",
            "parent": None,
        }
    ]
    logger.debug(f"返回模型列表: {models}")
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
        logger.debug(f"请求体: {body}")
        messages = body.get("messages")

        # 2. 获取并验证参数
        model_arg = get_and_validate_params(body)
        stream = model_arg["stream"]  # 获取 stream 参数

        # 3. 根据 stream 参数返回相应的响应
        if stream:
            return StreamingResponse(
                deep_claude.chat_completions_with_stream(
                    messages=messages,
                    model_arg=model_arg,
                    deepseek_model=DEEPSEEK_MODEL,
                    claude_model=CLAUDE_MODEL,
                ),
                media_type="text/event-stream",
            )
        else:
            return {"error": "Only streaming is supported"}

    except Exception as e:
        logger.error(f"处理请求时发生错误: {e}")
        return {"error": str(e)}


def get_and_validate_params(body):
    """提取获取和验证请求参数的函数"""
    # TODO: 默认值设定允许自定义
    temperature: float = body.get("temperature", 0.6)
    top_p: float = body.get("top_p", 0.9)
    presence_penalty: float = body.get("presence_penalty", 0.0)
    frequency_penalty: float = body.get("frequency_penalty", 0.0)
    stream: bool = body.get("stream", True)
    reasoning_effort: str = body.get("reasoning_effort", "medium")

    if "sonnet" in body.get(
        "model", ""
    ):  # Only Sonnet 设定 temperature 必须在 0 到 1 之间
        if (
            not isinstance(temperature, (float))
            or temperature < 0.0
            or temperature > 1.0
        ):
            raise ValueError("Sonnet 设定 temperature 必须在 0 到 1 之间")

    return {
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "reasoning_effort": reasoning_effort,
        "stream": stream,
    }
