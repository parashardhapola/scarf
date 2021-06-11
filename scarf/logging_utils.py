"""
Utility functions for logging.
"""
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, colorize=True, format="<level>{level}</level>: {message}", level="INFO")

__all__ = ['logger']
