"""Every dependency actually required on linux/aarch64 must be installable there.

arm64 is a first-class target, and the failure mode is nasty: resolution succeeds on
an x86 developer machine and the arm64 image build dies minutes in with "doesn't have
a source distribution or wheel for the current platform".

Checking the *latest* release of each package on PyPI does not catch this -- a
transitive cap can pin an older version that predates aarch64 wheels.

The walk is marker-aware rather than allow-listed: plenty of locked packages are
legitimately x86-only (the whole ``nvidia-*-cu12`` CUDA set, ``triton``) or
Windows-only, and they are correctly gated behind environment markers. Evaluating
those markers is what separates "not needed here" from "will break the build".
"""

import tomllib
from pathlib import Path
from typing import Any, cast

from packaging.markers import Marker

LOCK = Path(__file__).resolve().parent.parent / "uv.lock"
ROOT = "medmcp-cardiac"

# A linux/aarch64 CPython 3.12 target -- the arm64 image build leg.
_ARM64_ENV: dict[str, str] = {
    "sys_platform": "linux",
    "platform_system": "Linux",
    "platform_machine": "aarch64",
    "os_name": "posix",
    "python_version": "3.12",
    "python_full_version": "3.12.11",
    "implementation_name": "cpython",
    "platform_python_implementation": "CPython",
    "platform_release": "",
    "platform_version": "",
    "implementation_version": "3.12.11",
    "extra": "",
}


def _required_on_arm64(packages: dict[str, dict[str, Any]]) -> set[str]:
    """Names reachable from the root whose dependency markers hold on linux/aarch64."""
    seen: set[str] = set()
    queue: list[str] = [ROOT]
    while queue:
        name = queue.pop()
        if name in seen or name not in packages:
            continue
        seen.add(name)
        for edge in packages[name].get("dependencies", []):
            marker = edge.get("marker")
            if marker and not Marker(str(marker)).evaluate(_ARM64_ENV):
                continue
            queue.append(str(edge["name"]))
    return seen


def _installable_on_arm64(package: dict[str, Any]) -> bool:
    """Whether this locked package has an artifact usable on linux/aarch64."""
    if package.get("sdist"):
        return True  # builds from source; slow at worst, not a hard failure
    raw_wheels = package.get("wheels")
    if not isinstance(raw_wheels, list):
        return True  # no registry artifacts recorded (local/git source)
    for wheel in cast(list[dict[str, Any]], raw_wheels):
        name = str(wheel.get("url", "")).rsplit("/", 1)[-1]
        if name.endswith("-py3-none-any.whl") or "py2.py3-none-any" in name:
            return True
        if "aarch64" in name and ("manylinux" in name or "musllinux" in name):
            return True
    return False


def test_every_required_dependency_is_installable_on_arm64() -> None:
    """Nothing the arm64 build actually needs is missing an aarch64 artifact."""
    lock = tomllib.loads(LOCK.read_text())
    packages = {str(p["name"]): p for p in lock["package"]}
    assert ROOT in packages, f"{ROOT} not found in uv.lock"

    required = _required_on_arm64(packages)
    assert len(required) > 1, "dependency walk found nothing -- lock format changed?"

    broken = sorted(name for name in required if not _installable_on_arm64(packages[name]))
    assert broken == [], (
        "these packages are required on linux/aarch64 but have no aarch64 wheel and no "
        f"sdist, so the arm64 image cannot be built: {broken}. Add a [tool.uv] "
        "override-dependencies entry lifting whatever caps them to a version that "
        "publishes aarch64 wheels."
    )


def test_torch_is_the_fleet_pin_from_the_cu128_index() -> None:
    """The torch pin stays at the fleet's validated version, from the CUDA 12.8 index."""
    lock = tomllib.loads(LOCK.read_text())
    torch = next(p for p in lock["package"] if p["name"] == "torch")
    assert torch["version"].startswith("2.7.1"), torch["version"]
    assert "download.pytorch.org/whl/cu128" in str(torch.get("source", {})), torch.get("source")
