from typing import Annotated, Literal, Optional

import pytest

from trame_mcp.schema import (
    ValidationError,
    bind_arguments,
    function_input_schema,
    type_to_schema,
    validate,
)


def test_scalars():
    assert type_to_schema(str) == {"type": "string"}
    assert type_to_schema(int) == {"type": "integer"}
    assert type_to_schema(float) == {"type": "number"}
    assert type_to_schema(bool) == {"type": "boolean"}


def test_containers():
    assert type_to_schema(list[int]) == {"type": "array", "items": {"type": "integer"}}
    assert type_to_schema(dict[str, float]) == {
        "type": "object",
        "additionalProperties": {"type": "number"},
    }
    assert type_to_schema(list) == {"type": "array"}
    assert type_to_schema(dict) == {"type": "object"}


def test_optional_and_union():
    assert type_to_schema(Optional[int]) == {  # noqa: UP045 - Optional is the point
        "anyOf": [{"type": "integer"}, {"type": "null"}]
    }
    assert type_to_schema(int | str) == {
        "anyOf": [{"type": "integer"}, {"type": "string"}]
    }


def test_literal_and_annotated():
    assert type_to_schema(Literal["a", "b"]) == {"enum": ["a", "b"]}
    assert type_to_schema(Annotated[int, "number of facets"]) == {
        "type": "integer",
        "description": "number of facets",
    }
    assert type_to_schema(Annotated[int, "facets", {"minimum": 3}]) == {
        "type": "integer",
        "description": "facets",
        "minimum": 3,
    }


def test_fixed_length_tuples():
    assert type_to_schema(tuple[float, float, float]) == {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 3,
        "maxItems": 3,
    }
    assert type_to_schema(tuple[str, int]) == {
        "type": "array",
        "prefixItems": [{"type": "string"}, {"type": "integer"}],
        "minItems": 2,
        "maxItems": 2,
    }
    assert type_to_schema(tuple[int, ...]) == {"type": "array", "items": {"type": "integer"}}


def test_unknown_types_degrade_to_any():
    class Custom:
        pass

    assert type_to_schema(Custom) == {}
    assert type_to_schema(None) == {}


def test_function_input_schema():
    def fn(resolution: int, label: str = "cone", factor: float | None = None) -> str:
        return label

    schema = function_input_schema(fn)
    assert schema["type"] == "object"
    assert schema["required"] == ["resolution"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["resolution"] == {"type": "integer"}
    assert schema["properties"]["label"] == {"type": "string", "default": "cone"}
    assert schema["properties"]["factor"]["default"] is None


def test_function_input_schema_none_default_is_nullable():
    def fn(count: int = None):  # noqa: RUF013 - implicit Optional is the point
        return count

    assert function_input_schema(fn)["properties"]["count"] == {
        "anyOf": [{"type": "integer"}, {"type": "null"}],
        "default": None,
    }


def test_function_input_schema_var_keyword_allows_extras():
    def fn(a: int, **options):
        return a

    schema = function_input_schema(fn)
    assert "additionalProperties" not in schema
    assert list(schema["properties"]) == ["a"]


def test_function_input_schema_skips_self():
    class App:
        def method(self, value: int):
            pass

    schema = function_input_schema(App.method)
    assert list(schema["properties"]) == ["value"]


def test_validate_scalars_and_coercion():
    assert validate(3, {"type": "integer"}) == 3
    assert validate(3.0, {"type": "integer"}) == 3
    assert isinstance(validate(3, {"type": "number"}), float)
    for value, schema in (
        (3.5, {"type": "integer"}),
        (True, {"type": "integer"}),
        (True, {"type": "number"}),
        (1, {"type": "boolean"}),
        ("3", {"type": "integer"}),
        (None, {"type": "string"}),
    ):
        with pytest.raises(ValidationError):
            validate(value, schema)


def test_validate_unknown_and_empty_schemas_accept_anything():
    assert validate({"x": [1]}, {}) == {"x": [1]}
    assert validate("x", {"description": "no type", "model": "Person"}) == "x"


def test_validate_containers_report_paths():
    schema = type_to_schema(dict[str, list[float]])
    assert validate({"a": [1, 2.5]}, schema) == {"a": [1.0, 2.5]}
    with pytest.raises(ValidationError, match=r"value\.a\[1\]: expected number, got string"):
        validate({"a": [1, "x"]}, schema)

    schema = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
    with pytest.raises(ValidationError, match="at least 3 items"):
        validate([1, 2], schema, "color")
    with pytest.raises(ValidationError, match="at most 3 items"):
        validate([1, 2, 3, 4], schema, "color")


def test_validate_bounds_enum_union():
    assert validate(0.5, {"type": "number", "minimum": 0, "maximum": 1}) == 0.5
    with pytest.raises(ValidationError, match="at most 1"):
        validate(2, {"type": "number", "maximum": 1})
    with pytest.raises(ValidationError, match="must be one of"):
        validate("cubes", {"enum": ["surface", "wireframe"]})
    # True must not match 1 in an enum
    with pytest.raises(ValidationError):
        validate(True, {"enum": [1, 2]})

    schema = type_to_schema(int | str)
    assert validate("a", schema) == "a"
    with pytest.raises(ValidationError, match="does not match any allowed form"):
        validate([1], schema)
    assert validate(None, type_to_schema(Optional[int])) is None  # noqa: UP045
    assert validate("on", {"type": ["boolean", "string"]}) == "on"


def test_bind_arguments_checks_schema_and_signature():
    def fn(a: int, b: str = "x"):
        return a

    schema = function_input_schema(fn)
    assert bind_arguments(fn, {"a": 1.0}, schema) == {"a": 1}
    with pytest.raises(ValidationError, match="arguments: expected object"):
        bind_arguments(fn, [1], schema)
    with pytest.raises(ValidationError, match="unexpected 'c'"):
        bind_arguments(fn, {"a": 1, "c": 2}, schema)

    # a permissive hand-written schema still cannot smuggle unknown arguments
    with pytest.raises(ValidationError, match="unexpected keyword argument"):
        bind_arguments(fn, {"a": 1, "c": 2}, {"type": "object"})


def test_from_json_restores_declared_tuples():
    from trame_mcp.schema import from_json

    Vector = tuple[float, float, float]
    assert from_json([1.0, 2.0, 3.0], Vector) == (1.0, 2.0, 3.0)
    assert from_json([[0, 1], [2, 3]], list[tuple[int, int]]) == [(0, 1), (2, 3)]
    assert from_json([1, 2, 3], tuple[int, ...]) == (1, 2, 3)
    assert from_json({"a": [1, 2]}, dict[str, tuple[int, int]]) == {"a": (1, 2)}
    assert from_json([1.0, 2.0, 3.0], Vector | None) == (1.0, 2.0, 3.0)
    assert from_json(None, Vector | None) is None
    assert from_json([1.0, 2.0, 3.0], Annotated[Vector, "a point"]) == (1.0, 2.0, 3.0)
    assert from_json([1.0, 2.0, 3.0], Annotated[Vector, "a point"] | None) == (1.0, 2.0, 3.0)
    assert from_json([1.0, 2.0, 3.0], Optional[Annotated[Vector, "p"]]) == (1.0, 2.0, 3.0)  # noqa: UP045
    # anything else is left alone
    assert from_json([1, 2], list[int]) == [1, 2]
    assert from_json([1, 2], list) == [1, 2]
    assert from_json("x", Vector) == "x"
    assert from_json([1, 2], None) == [1, 2]


def test_tool_arguments_get_their_declared_tuples(mcp):
    from trame_mcp.schema import bind_arguments, function_input_schema

    def place(position: tuple[float, float, float], tags: list[str] | None = None):
        return position

    schema = function_input_schema(place)
    arguments = bind_arguments(place, {"position": [1, 2, 3], "tags": ["a"]}, schema)
    assert arguments == {"position": (1.0, 2.0, 3.0), "tags": ["a"]}
