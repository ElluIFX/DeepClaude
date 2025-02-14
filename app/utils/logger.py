import os
import sys

from dotenv import load_dotenv
from loguru import logger

# 确保环境变量被加载
load_dotenv()

logger.remove()
if not os.environ.get("NO_STDERR"):
    logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))

cwd = os.getcwd()
logger.add(cwd + "/logs/{time}.log", level=os.getenv("LOG_LEVEL", "INFO"))
