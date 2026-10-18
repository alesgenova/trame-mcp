"""Permissions: powers an application lets the people running it withhold
from agents (``mcp.permissions``, ``--mcp-deny``).

The application declares each power it gates, and names the one a tool, a
model method or a constructor needs with ``requires=``:

.. code-block:: python

    mcp.permissions.declare("files", "Read files and folders on this computer")
    mcp.permissions.declare("scripts", "Run Python code the agent writes", implies="files")

    @tool(requires="files")
    def list_files(self, path: str) -> dict: ...

While a permission is denied, what needs it is left out of what agents are
offered (``tools/list``, ``dataclass_schemas``) and calls to it are refused
with :class:`PermissionDeniedError`: a client that listed the tools earlier
still has them, and nothing tells it the list changed. ``require()`` checks
inside a call, for a need that depends on the arguments.

``implies`` declares that a permission grants another anyway (a script can
read files): allowing it allows the ones it implies, and denying one denies
the ones implying it, so no setting holds one without the other.

``--mcp-deny NAME[,NAME]`` (repeatable) and ``TRAME_MCP_DENY`` deny
permissions for the whole run: whoever launched the application decided,
and neither its code nor its UI can allow them again (``locked`` tells a UI
which switches to disable). A name the application does not declare
refuses every agent call once the server runs: the user meant to withhold
something, and a typo must not leave it allowed.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = ["DENY_ENV", "PermissionDeniedError", "Permissions", "requirements"]

DENY_ENV = "TRAME_MCP_DENY"


class PermissionDeniedError(PermissionError):
    """A call needs a permission that is denied (``mcp.permissions``)."""


def requirements(value, label="requires") -> tuple[str, ...]:
    """``requires=`` (or ``implies=``) as a tuple of permission names: None,
    a name, or a list of names."""
    if value is None:
        return ()
    names = (value,) if isinstance(value, str) else value
    try:
        names = tuple(names)
    except TypeError:
        names = None
    if names is None or not all(isinstance(n, str) and n for n in names):
        msg = f"{label} must be a permission name or a list of them, got {value!r}"
        raise TypeError(msg)
    return names


def launch_names(values) -> tuple[str, ...]:
    """Names given at launch, each value comma-separated."""
    return tuple(n.strip() for v in values if v for n in v.split(",") if n.strip())


@dataclass
class _Permission:
    description: str
    implies: tuple[str, ...]
    allowed: bool


class Permissions(Mapping):
    """``mcp.permissions``: the declared permissions, as a mapping of name
    to whether agents are allowed it. Assign ``permissions[name] = bool`` or
    :meth:`update` several at once, at any time."""

    def __init__(self):
        self._declared: dict[str, _Permission] = {}
        self._launch_denied = frozenset()
        self._started = False

    def __getitem__(self, name) -> bool:
        return self._get(name).allowed

    def __iter__(self):
        return iter(self._declared)

    def __len__(self):
        return len(self._declared)

    def __setitem__(self, name, allowed):
        self.update({name: allowed})

    def __repr__(self):
        return f"Permissions({dict(self)!r})"

    def declare(self, name, description, *, implies=None, allowed=True):
        """Declare a permission, allowed unless ``allowed=False``.

        :param description: what it lets an agent do, in a few words; an
            agent reads it when a call is refused.
        :param implies: the permissions this one grants anyway, declared
            before it. It is never allowed without them, so it starts denied
            when one of them is.
        """
        if not isinstance(name, str) or not name:
            msg = f"A permission name is a non-empty string, got {name!r}"
            raise TypeError(msg)
        if name in self._declared:
            msg = f"The permission {name!r} is already declared"
            raise ValueError(msg)
        implies = requirements(implies, "implies")
        unknown = [n for n in implies if n not in self._declared]
        if unknown:
            msg = f"{name!r} implies {unknown}, which are not declared (declare them first)"
            raise ValueError(msg)
        allowed = (
            bool(allowed)
            and name not in self._launch_denied
            and all(self._declared[n].allowed for n in implies)
        )
        self._declared[name] = _Permission(str(description), implies, allowed)

    def update(self, values=(), /, **kwargs):
        """Allow (True) or deny (False) several permissions at once.

        Allowing one also allows those it implies, and denying one also
        denies those implying it; allowing and denying the same one that
        way raises ``ValueError``. A permission denied at launch
        (``locked``) stays denied.
        """
        changes = {**dict(values), **kwargs}
        for name in changes:
            self._get(name)
        allowed = {n for n, v in changes.items() if v}
        deny = self._closure({n for n, v in changes.items() if not v}, self._implying)
        clash = self._closure(allowed, self._implied) & deny
        if clash:
            msg = (
                f"Cannot allow and deny {sorted(clash)} at once: allowing a permission "
                f"allows those it implies, and denying one denies those implying it"
            )
            raise ValueError(msg)
        # A refused allowance brings nothing it implies either; the others
        # imply no locked permission (one implying it is locked too).
        held = allowed & set(self.locked)
        if held:
            logger.info(
                "trame-mcp: %s denied at launch for this run; not allowed",
                ", ".join(sorted(held)),
            )
        for name in self._closure(allowed - held, self._implied):
            self._declared[name].allowed = True
        for name in deny:
            self._declared[name].allowed = False

    @property
    def locked(self) -> tuple[str, ...]:
        """The permissions denied at launch, and those implying them: denied
        for the whole run, whatever is assigned."""
        held = self._closure(self._launch_denied & self._declared.keys(), self._implying)
        return tuple(name for name in self._declared if name in held)

    @property
    def denied(self) -> tuple[str, ...]:
        """The names of the denied permissions, in declaration order."""
        return tuple(name for name, p in self._declared.items() if not p.allowed)

    def description(self, name) -> str:
        return self._get(name).description

    def allows(self, names) -> bool:
        """Whether every permission in ``names`` is declared and allowed."""
        return all(name in self._declared and self._declared[name].allowed for name in names)

    def require(self, *names, what="This call"):
        """Raise :class:`PermissionDeniedError` unless every permission in
        ``names`` is allowed; ``what`` names the refused action."""
        for name in names:
            permission = self._declared.get(name)
            if permission is None:
                msg = f"{what} needs the permission {name!r}, which the application never declared"
                raise PermissionDeniedError(msg)
            if not permission.allowed:
                msg = (
                    f"{what} needs the {name!r} permission ({permission.description}), "
                    f"which is turned off. Only the people running the application can "
                    f"turn it on."
                )
                raise PermissionDeniedError(msg)

    # -------------------------------------------------------------------------
    # Launch options and lifecycle (driven by the MCP)
    # -------------------------------------------------------------------------

    def _deny_at_launch(self, names):
        self._launch_denied = frozenset(names)

    def _start(self):
        """The server starts: unknown launch names are reported, and refuse
        every agent call from now on."""
        self._started = True
        unknown = self._unknown_at_launch()
        if unknown:
            logger.error(
                "trame-mcp: --mcp-deny names %s, which this application does not declare "
                "(it declares: %s); every agent call is refused",
                ", ".join(unknown),
                ", ".join(self._declared) or "none",
            )

    def _unknown_at_launch(self) -> list[str]:
        return sorted(self._launch_denied - self._declared.keys())

    def _check_launch(self):
        """Refuse agent calls while a launch name matches no permission."""
        unknown = self._started and self._unknown_at_launch()
        if unknown:
            msg = (
                f"Agent access is refused: the application was launched denying "
                f"{', '.join(unknown)}, which it does not declare"
            )
            raise PermissionDeniedError(msg)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _get(self, name) -> _Permission:
        try:
            return self._declared[name]
        except (KeyError, TypeError):
            msg = f"No permission {name!r} is declared; declared: {list(self._declared)}"
            raise KeyError(msg) from None

    def _implied(self, name):
        return self._declared[name].implies

    def _implying(self, name):
        return [n for n, p in self._declared.items() if name in p.implies]

    @staticmethod
    def _closure(names, step) -> set[str]:
        found, stack = set(), list(names)
        while stack:
            name = stack.pop()
            if name not in found:
                found.add(name)
                stack.extend(step(name))
        return found
