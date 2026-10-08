"""The README's tool and checkpoint tables must match what the stack actually offers.

Both are hand-written prose next to machine truth: the registered MCP tools and the
checkpoint catalogue. Without this they drift silently.
"""

import re
from pathlib import Path

import pytest

from medmcp_cardiac.server import mcp
from medmcp_cardiac.tools import checkpoints

README = Path(__file__).resolve().parent.parent / "README.md"

_TOOL_ROW = re.compile(r"^\|\s*`([a-z_]+)`\s*\|", re.MULTILINE)
_CHECKPOINT_ROW = re.compile(r"^\|\s*`([a-z0-9]+)`\s*\|", re.MULTILINE)


def _section(title: str) -> str:
    text = README.read_text()
    start = text.index(f"## {title}")
    end = text.find("\n## ", start + 1)
    return text[start : end if end > 0 else None]


@pytest.mark.asyncio
async def test_readme_tool_inventory_matches_registered_tools() -> None:
    """One row per registered tool, no more, no less."""
    rows = set(_TOOL_ROW.findall(_section("Tool inventory")))
    registered = {tool.name for tool in await mcp.list_tools()}
    assert rows == registered, f"README rows {rows} vs registered tools {registered}"


def test_readme_checkpoint_table_matches_the_catalogue() -> None:
    """Every shipped checkpoint is listed, and nothing that is not shipped."""
    rows = set(_CHECKPOINT_ROW.findall(_section("Checkpoints")))
    assert rows == set(checkpoints.CHECKPOINTS), rows


def test_readme_names_the_default_checkpoint() -> None:
    """The reader can tell which one runs when they do not choose."""
    section = _section("Checkpoints")
    assert f"`{checkpoints.DEFAULT_CHECKPOINT}`" in section and "default" in section.lower()
