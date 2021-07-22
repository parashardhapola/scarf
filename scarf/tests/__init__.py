import os
import shutil
import sys
from ..logging_utils import logger

logger.remove()
logger.add(sys.stderr, level="ERROR")

__all__ = ["full_path", "remove"]


def full_path(fn, *args):
    if fn == "" or fn is None:
        return os.path.join("scarf", "tests", "datasets")
    else:
        return os.path.join("scarf", "tests", "datasets", fn, *args)


def remove(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    elif os.path.exists(dir_path):
        os.unlink(dir_path)
    else:
        pass
