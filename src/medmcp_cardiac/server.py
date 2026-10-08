"""MCP server entrypoint for medmcp-cardiac."""

from importlib.resources import files as _pkg_files

from mcp.server.fastmcp import FastMCP

from medmcp_cardiac.tools.calcium import cardiac_calcium_score
from medmcp_cardiac.tools.function import cardiac_function
from medmcp_cardiac.tools.regional import regional_wall_analysis
from medmcp_cardiac.tools.segmentation import segment_cine_lax4c, segment_cine_sax

mcp = FastMCP("medmcp-cardiac")

mcp.add_tool(segment_cine_sax)
mcp.add_tool(segment_cine_lax4c)
mcp.add_tool(cardiac_function)
mcp.add_tool(regional_wall_analysis)
mcp.add_tool(cardiac_calcium_score)


def server_config() -> dict[str, object]:
    """Return MCP server metadata for autodiscovery by the local agent."""
    return {
        "name": "medmcp-cardiac",
        "command": "medmcp-cardiac",
        # A 30-frame cine is well under a minute on a GPU, but on CPU every frame is a
        # few forward passes of a 100M-parameter network and a series can take an hour.
        # Keep in sync with the Dockerfile label and segmentation.SUBPROCESS_TIMEOUT_SEC.
        "tool_timeout_sec": 3600.0,
        "skills_path": str(_pkg_files("medmcp_cardiac") / "skills"),
    }


def main() -> None:
    """Launch the MCP server over stdio (JSON-RPC)."""
    mcp.run(transport="stdio")
