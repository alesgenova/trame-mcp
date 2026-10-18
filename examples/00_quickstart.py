"""Quick start: functional style.

Run:  python examples/00_quickstart.py --server
Then: claude mcp add --transport http quickstart http://localhost:8080/mcp

Register what the agent may use on the server's MCP, then turn agent access
on. Launch with --no-mcp to keep it off.
"""

from trame.app import get_server
from trame.ui.html import DivLayout
from trame.widgets import html

from trame_mcp import get_mcp

server = get_server("quickstart")
state = server.state
mcp = get_mcp(server)

state.resolution = 6


@mcp.tool
def reset_resolution() -> str:
    """Reset the resolution to its default value (6)."""
    with state:
        state.resolution = 6
    return "Resolution reset to 6"


mcp.state.expose(
    "resolution",
    description="Number of facets of the cone",
    type=int,
    schema={"minimum": 3, "maximum": 60},  # enforced on every MCP write
)
mcp.enabled = True  # serve what is declared above (--no-mcp overrides)

with DivLayout(server):
    html.H1("Resolution: {{ resolution }}")
    html.Input(
        type="range",
        min=3,
        max=60,
        step=1,
        v_model_number=("resolution",),
    )

if __name__ == "__main__":
    server.start()
