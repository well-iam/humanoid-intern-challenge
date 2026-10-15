"""Colab installs `requirements.txt` on top of its own PyTorch: the file must follow the lock, without PyTorch."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO / "requirements.txt"


def pins() -> dict[str, str]:
    """{package: version} for every `name==version` line of requirements.txt."""
    text = REQUIREMENTS.read_text()
    return {m[1].lower(): m[2] for m in re.finditer(r"^([A-Za-z0-9_.-]+)==([^\s;]+)", text, re.M)}


def test_requirements_leave_out_pytorch_and_the_cuda_libraries_it_brings():
    assert [n for n in pins() if n in ("torch", "triton") or n.startswith("nvidia-")] == []


def test_requirements_pin_the_tested_versions_of_the_key_packages():
    names = pins()
    assert names["transformers"] == "5.19.0"
    assert names["mujoco"] == "3.16.0"
    assert names["zarr"] == "3.4.1"
    assert names["gym-pusht"] == "0.1.8"


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
def test_requirements_are_what_the_lock_exports_now():
    # uv writes the exact export command into the file's header; rerunning it must give the same file.
    text = REQUIREMENTS.read_text()
    command = text.splitlines()[1].removeprefix("#").split()
    assert command[:3] == ["uv", "export", "--frozen"]
    exported = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=True).stdout
    assert exported == text
