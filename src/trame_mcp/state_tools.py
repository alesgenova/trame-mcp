"""Generic tools over plain trame state variables, reached through
``mcp.state``: state_get / state_update."""

from __future__ import annotations

from dataclasses import dataclass

from trame_mcp.schema import ValidationError, type_to_schema, validate

__all__ = ["StateLayer"]

STATE_GET_DESCRIPTION = """\
Read the current values of the exposed trame state variables.
Pass `names` to read a subset, or omit it to read all exposed variables.
Exposed variables:
{names}"""

STATE_UPDATE_DESCRIPTION = """\
Update writable trame state variables. `updates` maps variable names to their
new values; nothing is applied if any of them is rejected. Connected browser
clients see the change immediately and any server-side reactions are
triggered. Writable variables:
{names}"""


@dataclass
class _StateVar:
    writable: bool
    description: str | None
    schema: dict


def _summary(schema):
    if "enum" in schema:
        return f"one of {schema['enum']}"
    kind = schema.get("type")
    if kind == "array" and isinstance(schema.get("items"), dict) and "type" in schema["items"]:
        return f"array of {schema['items']['type']}"
    return kind if isinstance(kind, str) else None


class StateLayer:
    """Exposes trame state variables by name through ``state_get`` and
    ``state_update``. Obtained as ``mcp.state``."""

    def __init__(self, mcp):
        self._mcp = mcp
        self._entries: dict[str, _StateVar] = {}
        self._tools = set()  # names of the generic tools registered so far

    def expose(self, *names, read_only=False, description=None, type=None, schema=None):
        """Expose state variables to the agent.

        :param names: state variable names.
        :param read_only: the agent may read them with ``state_get`` but not
            assign them with ``state_update``.
        :param description: shown to the client next to each name.
        :param type: Python type of the values, e.g. ``int`` or ``list[float]``;
            advertised as JSON Schema and enforced on writes.
        :param schema: extra JSON Schema keywords (``minimum``, ``enum``, ...),
            also enforced on writes.

        Exposing a name again replaces its options.
        """
        if not names:
            msg = "mcp.state.expose() requires at least one variable name"
            raise ValueError(msg)
        if not isinstance(read_only, bool):
            msg = "read_only must be True or False; call mcp.state.expose() once per access level"
            raise TypeError(msg)

        var_schema = type_to_schema(type) if type is not None else {}
        var_schema.update(schema or {})
        for name in names:
            self._entries[name] = _StateVar(
                writable=not read_only, description=description, schema=dict(var_schema)
            )
        self._sync_tools()

    def unexpose(self, *names):
        """Withdraw state variables from ``state_get`` / ``state_update``."""
        for name in names:
            self._entries.pop(name, None)
        self._sync_tools()

    # -------------------------------------------------------------------------

    def _instructions(self):
        """The line this layer adds to the server instructions, if any."""
        if not self._entries:
            return None
        if self.writable_names:
            return "Shared state: read it with state_get and change it with state_update."
        return "Shared state: read it with state_get."

    @property
    def exposed_names(self):
        return sorted(self._entries)

    @property
    def writable_names(self):
        return sorted(n for n, e in self._entries.items() if e.writable)

    def _describe(self, names):
        lines = []
        for name in names:
            entry = self._entries[name]
            summary = _summary(entry.schema)
            details = [summary] if summary else []
            details.append("writable" if entry.writable else "read-only")
            line = f"- `{name}` ({', '.join(details)})"
            if entry.description:
                line += f": {entry.description}"
            lines.append(line)
        return "\n".join(lines) if lines else "(none)"

    def _input_schemas(self):
        get_schema = {
            "type": "object",
            "properties": {
                "names": {
                    "type": "array",
                    "items": {"enum": self.exposed_names},
                    "description": "Subset of exposed variable names to read",
                }
            },
            "additionalProperties": False,
        }
        update_schema = {
            "type": "object",
            "properties": {
                "updates": {
                    "type": "object",
                    "description": "Map of writable state variable names to new values",
                    "properties": {
                        name: self._entries[name].schema for name in self.writable_names
                    },
                    "additionalProperties": False,
                }
            },
            "required": ["updates"],
            "additionalProperties": False,
        }
        return get_schema, update_schema

    def _sync_tools(self):
        """Register state_get on the first exposure and state_update on the
        first writable one; keep their descriptions and schemas current."""
        get_schema, update_schema = self._input_schemas()
        if self._entries or "state_get" in self._tools:
            self._sync_tool(
                "state_get",
                self._state_get,
                STATE_GET_DESCRIPTION.format(names=self._describe(self.exposed_names)),
                get_schema,
                {"readOnlyHint": True},
            )
        if self.writable_names or "state_update" in self._tools:
            self._sync_tool(
                "state_update",
                self._state_update,
                STATE_UPDATE_DESCRIPTION.format(names=self._describe(self.writable_names)),
                update_schema,
                {"idempotentHint": True},
            )

    def _sync_tool(self, name, fn, description, input_schema, annotations):
        if name not in self._tools:
            self._mcp._add_tool(
                fn,
                name=name,
                description=description,
                input_schema=input_schema,
                annotations=annotations,
            )
            self._tools.add(name)
        else:
            entry = self._mcp.get_tool(name)
            entry.description = description
            entry.input_schema = input_schema

    # -------------------------------------------------------------------------
    # Tool implementations
    # -------------------------------------------------------------------------

    def _state_get(self, names=None):
        state = self._mcp.server.state
        if names is None:
            names = self.exposed_names
        unknown = [n for n in names if n not in self._entries]
        if unknown:
            msg = f"Variables {unknown} are not exposed. Exposed variables: {self.exposed_names}"
            raise ValueError(msg)
        return {name: state[name] for name in names}

    def _state_update(self, updates):
        if not isinstance(updates, dict):
            msg = "`updates` must be an object mapping variable names to values"
            raise ValueError(msg)

        not_writable = [
            n for n in updates if n not in self._entries or not self._entries[n].writable
        ]
        if not_writable:
            msg = (
                f"Nothing was applied: variables {not_writable} are not writable. "
                f"Writable variables: {self.writable_names}"
            )
            raise ValueError(msg)

        values, errors = {}, []
        for name, value in updates.items():
            try:
                values[name] = validate(value, self._entries[name].schema, name)
            except ValidationError as e:
                errors.append(str(e))
        if errors:
            msg = f"Nothing was applied: {'; '.join(errors)}"
            raise ValidationError(msg)

        with self._mcp.server.state as state:
            for name, value in values.items():
                state[name] = value

        return {"updated": sorted(values)}
