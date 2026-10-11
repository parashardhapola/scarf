import os
import shutil
import sys
import sysconfig
from pathlib import Path

import pytest


def test_sgtsne_is_installed():
    executable = Path(sysconfig.get_path("scripts")) / "sgtsne"
    source = Path(__file__).resolve().parents[1] / "bin" / "sgtsne"
    assert executable.is_file()
    assert executable.read_bytes() == source.read_bytes()


@pytest.mark.skipif(sys.platform != "linux", reason="SG-tSNE targets Linux")
def test_sgtsne_is_on_path():
    executable = Path(sysconfig.get_path("scripts")) / "sgtsne"
    assert os.access(executable, os.X_OK)
    command = shutil.which("sgtsne")
    assert command is not None
    assert Path(command).resolve() == executable.resolve()
