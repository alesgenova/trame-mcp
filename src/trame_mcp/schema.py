"""JSON Schema generation from Python type annotations, and validation of
incoming MCP values against the schemas we advertise."""

from __future__ import annotations

import inspect
import json
import re
import types
import typing

__all__ = [
    "ToolError",
    "ValidationError",
    "bind_arguments",
    "from_json",
    "function_input_schema",
    "model_parameter_schema",
    "parameter_hints",
    "resolve_model_arguments",
    "type_to_schema",
    "validate",
]

_SCALARS = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    type(None): {"type": "null"},
}


def type_to_schema(annotation, custom=None) -> dict:
    """Best-effort conversion of a type annotation to a JSON Schema fragment.

    Unknown or missing annotations yield ``{}`` (any type). ``custom`` is an
    optional ``annotation -> schema | None`` hook consulted first at every
    level (used to describe nested model references).

    ``Annotated`` metadata contributes a description (``str``) or extra
    schema keywords (``dict``), e.g.
    ``Annotated[int, "Number of facets", {"minimum": 3, "maximum": 60}]``.
    """
    if annotation is None or annotation is inspect.Parameter.empty:
        return {}

    if custom is not None:
        schema = custom(annotation)
        if schema is not None:
            return schema

    # isinstance first: Annotated metadata may be unhashable (dict schemas)
    if isinstance(annotation, type) and annotation in _SCALARS:
        return dict(_SCALARS[annotation])

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    def convert(a):
        return type_to_schema(a, custom)

    if origin is None:
        if annotation is dict:
            return {"type": "object"}
        if annotation in (list, set, frozenset, tuple):
            return {"type": "array"}
        return {}

    if origin is typing.Annotated:
        schema = convert(args[0])
        for extra in args[1:]:
            if isinstance(extra, str):
                schema["description"] = extra
            elif isinstance(extra, dict):
                schema.update(extra)
        return schema

    if origin is typing.Literal:
        return {"enum": list(args)}

    if origin in (typing.Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        optional = len(non_none) != len(args)
        if len(non_none) == 1:
            schema = convert(non_none[0])
            return nullable(schema) if optional else schema
        variants = [convert(a) for a in non_none]
        if optional:
            variants.append({"type": "null"})
        return {"anyOf": variants}

    if origin is tuple and not (len(args) == 2 and args[1] is Ellipsis):
        # Fixed-length tuple, e.g. tuple[float, float, float]
        items = [convert(a) for a in args]
        schema = {"type": "array", "minItems": len(items), "maxItems": len(items)}
        if all(item == items[0] for item in items):
            schema["items"] = items[0]
        else:
            schema["prefixItems"] = items
        return schema

    if origin in (list, set, frozenset, tuple):
        return {"type": "array", "items": convert(args[0]) if args else {}}

    if origin is dict:
        value_schema = convert(args[1]) if len(args) == 2 else {}
        return {"type": "object", "additionalProperties": value_schema}

    return {}


def from_json(value, annotation):
    """``value``, validated JSON, in the Python shape ``annotation``
    declares: JSON has no tuples, so arrays become tuples where the
    annotation says so (``tuple[float, float, float]``, nested in lists,
    dicts and unions too). Anything else is returned as is."""
    if annotation is None or annotation is inspect.Parameter.empty:
        return value
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Annotated:
        return from_json(value, args[0])
    if origin in (typing.Union, types.UnionType):
        if value is None:
            return value
        kinds = (list, tuple) if isinstance(value, list) else (dict,)
        for arg in args:
            base = typing.get_args(arg)[0] if typing.get_origin(arg) is typing.Annotated else arg
            if (typing.get_origin(base) or base) in kinds:
                return from_json(value, base)
        return value
    if isinstance(value, list):
        if annotation is tuple:
            return tuple(value)
        if origin is tuple:
            if len(args) == 2 and args[1] is Ellipsis:
                return tuple(from_json(v, args[0]) for v in value)
            if len(args) == len(value):
                return tuple(from_json(v, a) for v, a in zip(value, args, strict=True))
            return tuple(value)
        if origin is list and args:
            return [from_json(v, args[0]) for v in value]
    if isinstance(value, dict) and origin is dict and len(args) == 2:
        return {k: from_json(v, args[1]) for k, v in value.items()}
    return value


def nullable(schema: dict) -> dict:
    """Return ``schema`` extended to also accept ``null``."""
    if not schema or _accepts_null(schema):
        return schema
    return {"anyOf": [schema, {"type": "null"}]}


def _accepts_null(schema):
    kind = schema.get("type")
    if kind == "null" or (isinstance(kind, list) and "null" in kind):
        return True
    return any(_accepts_null(s) for s in schema.get("anyOf", ()))


def _model_class(annotation):
    """``annotation`` when it is a trame-dataclass model class, else None
    (also when trame-dataclass is not installed)."""
    if not isinstance(annotation, type):
        return None
    try:
        from trame_dataclass.v2 import StateDataModel
    except ImportError:
        return None
    return annotation if issubclass(annotation, StateDataModel) else None


def model_parameter_schema(annotation):
    """``type_to_schema`` hook for parameters: a model instance travels as
    its id (``X | None`` and ``list[X]`` follow from the hook running at
    every level)."""
    cls = _model_class(annotation)
    if cls is None:
        return None
    name = cls.__name__
    return {"type": "string", "model": name, "description": f"Id of a {name} instance"}


def parameter_hints(fn):
    """The resolved annotations of ``fn``'s parameters. One annotation that
    cannot be resolved (a forward reference to a later class, typically the
    return type) does not lose the others."""
    try:
        return typing.get_type_hints(fn, include_extras=True)
    except Exception:
        pass
    func = inspect.unwrap(getattr(fn, "__func__", fn))
    globalns = getattr(func, "__globals__", {})
    hints = {}
    for name, annotation in getattr(func, "__annotations__", {}).items():
        if name == "return":
            continue
        holder = types.SimpleNamespace(__annotations__={name: annotation})
        try:
            hints[name] = typing.get_type_hints(holder, globalns=globalns, include_extras=True)[
                name
            ]
        except Exception:
            continue
    return hints


def _json_safe(value):
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def function_input_schema(fn, skip=("self",)) -> dict:
    """Build the MCP ``inputSchema`` for a callable from its signature.
    Parameters annotated with a model class take the instance's id."""
    sig = inspect.signature(fn)
    hints = parameter_hints(fn)

    properties = {}
    required = []
    var_keyword = False
    for pname, param in sig.parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            var_keyword = True
            continue
        if pname in skip or param.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        prop = type_to_schema(hints.get(pname), model_parameter_schema)
        if param.default is inspect.Parameter.empty:
            required.append(pname)
        else:
            if param.default is None:
                prop = nullable(prop)
            if _json_safe(param.default):
                prop["default"] = param.default
        properties[pname] = prop

    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    if not var_keyword:
        schema["additionalProperties"] = False
    return schema


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------


class ToolError(ValueError):
    """An expected, actionable failure whose message is complete as is: the
    agent reads it without the exception's type name."""


class ValidationError(ToolError):
    """An MCP-supplied value does not match the advertised schema."""


def _type_name(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _json_equal(a, b):
    return a == b and isinstance(a, bool) == isinstance(b, bool)


def _check_type(value, kind, path):
    """Return ``value`` coerced to ``kind`` (lossless numeric coercion only)."""
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "integer":
        if is_number and isinstance(value, int):
            return value
        if is_number and isinstance(value, float) and value.is_integer():
            return int(value)
    elif kind == "number":
        if is_number:
            return float(value) if isinstance(value, int) else value
    elif kind == "string":
        if isinstance(value, str):
            return value
    elif kind == "boolean":
        if isinstance(value, bool):
            return value
    elif kind == "null":
        if value is None:
            return value
    elif kind == "array":
        if isinstance(value, (list, tuple)):
            return list(value)
    elif kind == "object":
        if isinstance(value, dict):
            return value
    else:
        return value  # unknown type keyword: do not reject

    msg = f"{path}: expected {kind}, got {_type_name(value)}"
    raise ValidationError(msg)


def validate(value, schema, path="value"):
    """Validate ``value`` against the JSON Schema subset trame-mcp generates.

    Returns the value with lossless coercions applied (``3.0`` for an
    integer becomes ``3``, ``3`` for a number becomes ``3.0``); raises
    :class:`ValidationError` naming the offending path otherwise. Keywords
    outside the supported subset are ignored.
    """
    if not schema:
        return value

    branches = schema.get("anyOf") or schema.get("oneOf")
    if branches:
        errors = []
        for branch in branches:
            try:
                value = validate(value, branch, path)
                break
            except ValidationError as e:
                errors.append(str(e).removeprefix(f"{path}: "))
        else:
            msg = f"{path}: does not match any allowed form ({'; '.join(errors)})"
            raise ValidationError(msg)

    kind = schema.get("type")
    if isinstance(kind, list):
        for candidate in kind:
            try:
                value = _check_type(value, candidate, path)
                break
            except ValidationError:
                continue
        else:
            msg = f"{path}: expected {' or '.join(kind)}, got {_type_name(value)}"
            raise ValidationError(msg)
    elif kind is not None:
        value = _check_type(value, kind, path)

    if "const" in schema and not _json_equal(value, schema["const"]):
        msg = f"{path}: must be {schema['const']!r}"
        raise ValidationError(msg)
    if "enum" in schema and not any(_json_equal(value, e) for e in schema["enum"]):
        msg = f"{path}: must be one of {schema['enum']!r}, got {value!r}"
        raise ValidationError(msg)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        _check_bounds(value, schema, path)
    elif isinstance(value, str):
        _check_string(value, schema, path)
    elif isinstance(value, list):
        value = _check_array(value, schema, path)
    elif isinstance(value, dict):
        value = _check_object(value, schema, path)

    return value


def _check_bounds(value, schema, path):
    checks = (
        ("minimum", lambda v, b: v >= b, "at least"),
        ("maximum", lambda v, b: v <= b, "at most"),
        ("exclusiveMinimum", lambda v, b: v > b, "greater than"),
        ("exclusiveMaximum", lambda v, b: v < b, "less than"),
    )
    for keyword, ok, label in checks:
        if keyword in schema and not ok(value, schema[keyword]):
            msg = f"{path}: must be {label} {schema[keyword]}, got {value}"
            raise ValidationError(msg)


def _check_string(value, schema, path):
    if "minLength" in schema and len(value) < schema["minLength"]:
        msg = f"{path}: must be at least {schema['minLength']} characters"
        raise ValidationError(msg)
    if "maxLength" in schema and len(value) > schema["maxLength"]:
        msg = f"{path}: must be at most {schema['maxLength']} characters"
        raise ValidationError(msg)
    if "pattern" in schema and not re.search(schema["pattern"], value):
        msg = f"{path}: must match pattern {schema['pattern']!r}"
        raise ValidationError(msg)


def _check_array(value, schema, path):
    if "minItems" in schema and len(value) < schema["minItems"]:
        msg = f"{path}: expected at least {schema['minItems']} items, got {len(value)}"
        raise ValidationError(msg)
    if "maxItems" in schema and len(value) > schema["maxItems"]:
        msg = f"{path}: expected at most {schema['maxItems']} items, got {len(value)}"
        raise ValidationError(msg)

    prefix = schema.get("prefixItems", ())
    items = schema.get("items")
    result = []
    for i, item in enumerate(value):
        item_schema = prefix[i] if i < len(prefix) else items
        result.append(validate(item, item_schema, f"{path}[{i}]") if item_schema else item)
    return result


def _check_object(value, schema, path):
    properties = schema.get("properties", {})
    additional = schema.get("additionalProperties", True)

    missing = [k for k in schema.get("required", ()) if k not in value]
    if missing:
        msg = f"{path}: missing required {', '.join(repr(k) for k in missing)}"
        raise ValidationError(msg)

    result = {}
    for key, item in value.items():
        if not isinstance(key, str):
            msg = f"{path}: keys must be strings"
            raise ValidationError(msg)
        if key in properties:
            result[key] = validate(item, properties[key], f"{path}.{key}")
        elif additional is False:
            msg = (
                f"{path}: unexpected {key!r}"
                + (f" (allowed: {sorted(properties)})" if properties else "")
            )
            raise ValidationError(msg)
        elif isinstance(additional, dict):
            result[key] = validate(item, additional, f"{path}.{key}")
        else:
            result[key] = item
    return result


def bind_arguments(fn, arguments, schema, label="arguments", positional=()):
    """Validate MCP call arguments for ``fn`` and return the coerced dict.

    Checks them against ``schema`` and against ``fn``'s signature (catching
    missing/unexpected arguments the schema may not describe), so that a
    ``TypeError`` raised later can only come from the function body.
    ``positional`` are the leading arguments the caller supplies itself
    (a constructor's server). Model-typed parameters keep their ids; see
    :func:`resolve_model_arguments`.
    """
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        msg = f"{label}: expected object, got {_type_name(arguments)}"
        raise ValidationError(msg)
    arguments = validate(arguments, schema, label)
    try:
        inspect.signature(fn).bind(*positional, **arguments)
    except TypeError as e:
        msg = f"{label}: {e}"
        raise ValidationError(msg) from None
    hints = parameter_hints(fn)
    return {name: from_json(value, hints.get(name)) for name, value in arguments.items()}


def _mentions_model(annotation):
    if _model_class(annotation) is not None:
        return True
    return any(_mentions_model(a) for a in typing.get_args(annotation))


def _resolve_model(value, annotation, resolve, path):
    if value is None:
        return value
    cls = _model_class(annotation)
    if cls is not None:
        return resolve(value, cls, path) if isinstance(value, str) else value
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Annotated:
        return _resolve_model(value, args[0], resolve, path)
    if origin in (typing.Union, types.UnionType):
        for arg in args:
            base = typing.get_args(arg)[0] if typing.get_origin(arg) is typing.Annotated else arg
            if isinstance(value, str) and _model_class(base) is not None:
                return resolve(value, base, path)
            if isinstance(value, list) and typing.get_origin(base) in (list, tuple, set, frozenset):
                return _resolve_model(value, base, resolve, path)
            if isinstance(value, dict) and typing.get_origin(base) is dict:
                return _resolve_model(value, base, resolve, path)
        return value
    if isinstance(value, (list, tuple)) and origin in (list, tuple, set, frozenset) and args:
        items = [_resolve_model(v, args[0], resolve, f"{path}[{i}]") for i, v in enumerate(value)]
        return type(value)(items) if isinstance(value, tuple) else items
    if isinstance(value, dict) and origin is dict and len(args) == 2:
        return {k: _resolve_model(v, args[1], resolve, f"{path}.{k}") for k, v in value.items()}
    return value


def resolve_model_arguments(fn, arguments, resolve, label="arguments"):
    """``arguments`` (bound by :func:`bind_arguments`) with the ids given
    for model-typed parameters replaced by the instances:
    ``resolve(id, cls, path)`` returns the instance or raises."""
    hints = parameter_hints(fn)
    models = {name for name, hint in hints.items() if _mentions_model(hint)}
    if not models & arguments.keys():
        return arguments
    return {
        name: _resolve_model(value, hints[name], resolve, f"{label}.{name}")
        if name in models
        else value
        for name, value in arguments.items()
    }
