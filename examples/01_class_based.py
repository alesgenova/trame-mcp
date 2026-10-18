"""Class-based app: MCPApp exposes its @tool methods when built.

Run:  python examples/01_class_based.py --server          # agent access off
      python examples/01_class_based.py --server --mcp    # agent access on

The tools are declared either way; nothing is served until agent access is
on. The "Allow agent access" checkbox toggles `mcp.enabled` at runtime.
"""

from typing import Annotated

from trame.decorators import change
from trame.ui.html import DivLayout
from trame.widgets import html

from trame_mcp import MCPApp, tool


class ConeApp(MCPApp):
    def __init__(self, server=None):
        super().__init__(server)
        self.mcp.name = "cone-app"
        self.state.resolution = 6
        self.state.log = []
        self.state.agent_access = self.mcp.enabled
        self._build_ui()

    @tool
    def set_resolution(
        self, resolution: Annotated[int, "Number of facets", {"minimum": 3, "maximum": 60}]
    ) -> str:
        """Set the number of facets of the cone."""
        with self.state:
            self.state.resolution = resolution
        return f"Resolution set to {resolution}"

    @tool(read_only=True)
    def get_log(self) -> list:
        """Recent resolution changes, oldest first."""
        return list(self.state.log)

    @change("agent_access")
    def _on_agent_access(self, agent_access, **_):
        self.mcp.enabled = agent_access

    @change("resolution")
    def _on_resolution(self, resolution, **_):
        with self.state:
            self.state.log = [*self.state.log, f"resolution -> {resolution}"][-10:]

    def _build_ui(self):
        with DivLayout(self.server):
            with html.Label():
                html.Input(type="checkbox", v_model=("agent_access",))
                html.Span("Allow agent access")
            html.H1("Resolution: {{ resolution }}")
            html.Input(
                type="range", min=3, max=60, step=1, v_model_number=("resolution",)
            )
            html.Pre("{{ log.join('\\n') }}")


def main():
    app = ConeApp()
    app.server.start()


if __name__ == "__main__":
    main()
