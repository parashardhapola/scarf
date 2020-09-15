from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, colorize=True, format="<level>{level}</level>: {message}")

__all__ = [logger]
