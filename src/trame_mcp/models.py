"""The MCP-visible variant of trame-dataclass's ``StateDataModel``."""

from __future__ import annotations

from trame_dataclass.v2 import StateDataModel

from trame_mcp.decorators import (
    MODEL_MARKER,
    check_model_spec,
    get_model_spec,
    model_spec,
)

__all__ = ["MCPStateDataModel"]


class MCPStateDataModel(StateDataModel):
    """A ``StateDataModel`` whose class agents may see, as ``MCPApp`` is to
    ``TrameApp``.

    Being visible only declares the class: an agent reaches its instances
    through ``mcp.dataclass.expose_instance()`` entry points, and sees the
    fields marked with ``mcp_field()`` and the methods marked with
    ``@tool``; a classmethod marked with ``@constructor`` creates instances
    in a ``dataclass_batch``. Options are class keywords:

    .. code-block:: python

        class Person(MCPStateDataModel):
            name = mcp_field(Sync(str, ""), "Full name")
            nickname = Sync(str, "")  # not marked: not exposed

            @tool
            def celebrate_birthday(self) -> str: ...

        class AddressBook(MCPStateDataModel, all_fields=True, all_tools=True):
            ...

    :param description: what the agent reads about the model; defaults to the
        class docstring. Applies to this class only, like a docstring.
    :param all_fields: also expose unmarked fields with their descriptor-mode
        access (``Sync`` writable, ``ClientOnly`` read-only; ``ServerOnly``
        stays hidden). Off: only ``mcp_field``-marked or listed fields.
    :param all_tools: also expose unmarked public methods (no leading
        underscore, not part of ``StateDataModel``'s own API, not ``@watch``
        callbacks). Off: only ``@tool``-marked or listed methods.
    :param fields: names to expose, as a list, or a dict of name ->
        ``{"description", "read_only", "hidden", "schema"}`` (same meaning as
        :func:`trame_mcp.mcp_field`, overriding a marker key by key).
    :param tools: method names to expose, as a list, or a dict of name ->
        ``{"description", "read_only", "hidden"}``; ``hidden`` excludes a
        method that ``all_tools`` or a ``@tool`` marker would expose.

    Subclasses inherit the options; keywords given on a subclass replace
    ``all_fields`` / ``all_tools`` and add ``fields`` / ``tools`` entries.
    ``mcp.dataclass.expose_model()`` overrides all of this for one
    application.
    """

    def __init_subclass__(
        cls,
        *,
        description=None,
        all_fields=None,
        all_tools=None,
        fields=None,
        tools=None,
        **kwargs,
    ):
        try:
            super().__init_subclass__(**kwargs)
        except TypeError:
            if not kwargs:
                raise
            msg = (
                f"{cls.__name__}: unknown class keywords {sorted(kwargs)}; "
                f"MCPStateDataModel takes description, all_fields, all_tools, fields and tools"
            )
            raise TypeError(msg) from None
        # Descriptors have run __set_name__ by now, so names can be checked.
        spec = model_spec(
            description, all_fields, all_tools, fields, tools, parent=get_model_spec(cls)
        )
        check_model_spec(cls, spec)
        setattr(cls, MODEL_MARKER, spec)


setattr(MCPStateDataModel, MODEL_MARKER, model_spec())
