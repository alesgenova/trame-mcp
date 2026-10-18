import gc
import logging

import pytest
from trame_dataclass.v2 import ClientOnly, ServerOnly, StateDataModel, Sync, watch

from trame_mcp import MCPStateDataModel, mcp_field, tool

# -----------------------------------------------------------------------------
# Owned models: MCPStateDataModel, fields opted in by markers
# -----------------------------------------------------------------------------


class Address(MCPStateDataModel):
    street = mcp_field(Sync(str, ""))
    city = mcp_field(Sync(str, ""))


class Person(MCPStateDataModel):
    """Someone in the address book."""

    name = mcp_field(Sync(str, ""), "Full name", schema={"minLength": 1})
    age = mcp_field(Sync(int, 0), schema={"minimum": 0})
    height = mcp_field(Sync(float, 1.7))
    nickname = mcp_field(Sync(str))  # no default -> None is a legitimate value
    hint = mcp_field(ClientOnly(str))
    score = mcp_field(ServerOnly(int, 0), read_only=True)
    unmarked = Sync(str, "")  # not exposed: no all_fields
    notes = ServerOnly(str)
    home = mcp_field(Sync(Address, None, has_dataclass=True))
    vault = mcp_field(Sync(Address, None, has_dataclass=True), hidden=True)

    @tool
    def celebrate_birthday(self) -> str:
        """Increment this person's age by one year."""
        self.age += 1
        return f"{self.name} is now {self.age}"

    @tool(read_only=True)
    def greeting(self, formal: bool = False) -> str:
        """Say hello."""
        return f"Good day, {self.name}" if formal else f"Hi {self.name}"

    def unmarked_method(self):
        return "never exposed"


class Employee(Person):  # inherits visibility and options
    name = Sync(str, "Employee")  # redeclared: inherits the marker
    badge = Sync(str, "")  # unmarked: hidden, like in the parent


class AddressBook(MCPStateDataModel):
    title = mcp_field(Sync(str, "My contacts"))
    contacts = mcp_field(Sync(list[Person], list, has_dataclass=True))

    @tool
    def add_contact(self, name: str, age: int = 0) -> dict:
        """Create a new contact and add it to the book."""
        person = Person(self.server, name=name, age=age)
        self.contacts = [*self.contacts, person]
        return {"id": person._id}

    @tool(name="remove")
    def remove_contact(self, contact_id: str) -> str:
        """Remove a contact from the book."""
        self.contacts = [p for p in self.contacts if p._id != contact_id]
        return "removed"


# -----------------------------------------------------------------------------
# "Library" models: plain StateDataModel, exposed with expose_model()
# -----------------------------------------------------------------------------


class Gadget(StateDataModel):
    label = Sync(str, "")
    level = Sync(int, 0)
    secret = Sync(str, "")
    internal = ServerOnly(int, 0)

    def reset(self) -> str:
        """Reset the level."""
        self.level = 0
        return "reset"

    def rename(self, label: str) -> str:
        self.label = label
        return label

    @watch("level")
    def on_level(self, level):
        pass

    def _private(self):
        pass


class Shelf(MCPStateDataModel):
    gadgets = mcp_field(Sync(list[Gadget], list, has_dataclass=True))
    people = Sync(list[Person], list, has_dataclass=True)  # unmarked reference


@pytest.fixture
def book(mcp):
    book = AddressBook()
    book.contacts = [Person(name="Ada", age=36)]
    mcp.dataclass.expose_instance(book, name="address_book")
    return book


def fn(mcp, name):
    return mcp.get_tool(name).fn


def ada_id(mcp):
    return fn(mcp, "dataclass_get")("address_book")["values"]["contacts"][0]["id"]


# -----------------------------------------------------------------------------
# Registration and validation
# -----------------------------------------------------------------------------


def test_expose_instance_registers_the_generic_tools(mcp, book):
    annotations = {
        "dataclass_schemas": {"readOnlyHint": True},
        "dataclass_get": {"readOnlyHint": True},
        "dataclass_update": {"idempotentHint": True},
        "dataclass_call": None,
        "dataclass_batch": None,
    }
    for name, expected in annotations.items():
        assert mcp.get_tool(name).annotations == expected, name


def test_expose_instance_rejects_bad_sources(mcp):
    with pytest.raises(TypeError, match="StateDataModel instance"):
        mcp.dataclass.expose_instance(object(), name="x")
    with pytest.raises(TypeError, match="StateDataModel instance"):
        mcp.dataclass.expose_instance(Person, name="x")  # a class is not a getter


def test_marker_and_spec_validation(mcp):
    with pytest.raises(TypeError, match="read_only must be True, False or None"):
        mcp_field(Sync(int, 0), read_only="yes")
    with pytest.raises(ValueError, match="hidden field takes no read_only"):
        mcp_field(Sync(int, 0), read_only=True, hidden=True)
    with pytest.raises(ValueError, match="hidden field takes no read_only"):
        mcp.dataclass.expose_model(Gadget, fields={"label": {"hidden": True, "read_only": False}})
    with pytest.raises(ValueError, match="has no fields \\['bogus'\\]"):
        mcp.dataclass.expose_model(Gadget, fields=["bogus"])
    with pytest.raises(ValueError, match="unknown options \\['acess'\\]"):
        mcp.dataclass.expose_model(Gadget, fields={"label": {"acess": "read-only"}})
    with pytest.raises(ValueError, match=r"Gadget\.bogus is not a method"):
        mcp.dataclass.expose_model(Gadget, tools=["bogus"])
    with pytest.raises(TypeError, match="list or a dict"):
        mcp.dataclass.expose_model(Gadget, fields="label")
    with pytest.raises(TypeError, match="StateDataModel subclass"):
        mcp.dataclass.expose_model(object)
    with pytest.raises(ValueError, match="no fields"):

        class Broken(MCPStateDataModel, fields=["bogus"]):
            value = Sync(int, 0)


def test_server_only_field_needs_explicit_access(mcp):
    class Stats(StateDataModel):
        total = mcp_field(ServerOnly(int, 0), "Running total")

    with pytest.raises(ValueError, match="ServerOnly field"):
        mcp.dataclass.expose_model(Stats)
    with pytest.raises(ValueError, match="ServerOnly field"):
        mcp.dataclass.expose_model(Gadget, fields=["internal"])


def test_expose_model_once_per_class(mcp):
    mcp.dataclass.expose_model(Gadget, fields=["label"])
    # models alone give the agent nothing to start from: no tools yet
    assert mcp.get_tool("dataclass_get") is None
    with pytest.raises(ValueError, match="one expose_model\\(\\) call per model"):
        mcp.dataclass.expose_model(Gadget, fields=["level"])


def test_base_api_cannot_be_exposed(mcp):
    with pytest.raises(ValueError, match="StateDataModel's API"):
        mcp.dataclass.expose_model(Gadget, tools=["flush"])
    # a failed call registers nothing
    mcp.dataclass.expose_model(Gadget, tools=["reset"])


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------


def test_schemas_opt_in_fields_and_methods(mcp, book):
    schemas = fn(mcp, "dataclass_schemas")()

    assert schemas["instances"] == {
        "address_book": {"id": book._id, "type": "AddressBook"}
    }
    # Employee and Address are listed before any instance of them exists:
    # Employee as a subclass of Person, Address through Person.home
    assert set(schemas["models"]) == {"AddressBook", "Person", "Employee", "Address"}

    person = schemas["models"]["Person"]
    assert person["description"] == "Someone in the address book."
    fields = person["fields"]
    assert set(fields) == {"name", "age", "height", "nickname", "hint", "score", "home"}
    assert fields["name"] == {
        "type": "string",
        "minLength": 1,
        "description": "Full name",
    }
    assert fields["hint"]["read_only"] is True  # ClientOnly
    assert fields["score"]["read_only"] is True  # ServerOnly, explicit
    assert fields["nickname"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}
    assert fields["home"]["read_only"] is True  # reference field
    assert {"model": "Address"} in fields["home"]["anyOf"]

    assert set(person["methods"]) == {"celebrate_birthday", "greeting"}
    assert person["methods"]["greeting"]["read_only"] is True

    # a subclass is listed as what it changes: the redeclared name keeps the
    # inherited marker, badge stays hidden, so nothing differs from Person
    assert schemas["models"]["Employee"] == {"extends": "Person", "fields": {}, "methods": {}}

    contacts = schemas["models"]["AddressBook"]["fields"]["contacts"]
    assert contacts["items"] == {"model": "Person"}
    assert set(schemas["models"]["AddressBook"]["methods"]) == {"add_contact", "remove"}


async def test_expose_model_with_field_and_tool_lists(mcp):
    mcp.dataclass.expose_model(
        Gadget,
        fields={
            "label": {"description": "Display name", "schema": {"minLength": 1}},
            "level": {"read_only": True},
        },
        tools=["reset"],
    )
    gadget = Gadget(label="g", level=3)
    mcp.dataclass.expose_instance(gadget, name="gadget")

    model = fn(mcp, "dataclass_schemas")()["models"]["Gadget"]
    assert model["fields"] == {
        "label": {"type": "string", "minLength": 1, "description": "Display name"},
        "level": {"type": "integer", "read_only": True},
    }
    assert set(model["methods"]) == {"reset"}

    update = fn(mcp, "dataclass_update")
    with pytest.raises(ValueError, match="at least 1 characters"):
        await update("gadget", {"label": ""})
    with pytest.raises(ValueError, match="level \\(read-only\\)"):
        await update("gadget", {"level": 1})
    with pytest.raises(ValueError, match="secret \\(unknown field\\)"):
        await update("gadget", {"secret": "x"})


def test_expose_model_all_fields_and_all_tools(mcp):
    mcp.dataclass.expose_model(
        Gadget,
        all_fields=True,
        fields={"secret": {"hidden": True}},
        all_tools=True,
    )
    mcp.dataclass.expose_instance(Gadget(), name="gadget")

    model = fn(mcp, "dataclass_schemas")()["models"]["Gadget"]
    # ServerOnly stays hidden under all_fields; secret hidden explicitly
    assert set(model["fields"]) == {"label", "level"}
    # public methods only: no watcher, no private method, no StateDataModel API
    assert set(model["methods"]) == {"reset", "rename"}


def test_all_tools_exclusion(mcp):
    mcp.dataclass.expose_model(Gadget, all_tools=True, tools={"rename": {"hidden": True}})
    mcp.dataclass.expose_instance(Gadget(), name="gadget")
    assert set(fn(mcp, "dataclass_schemas")()["models"]["Gadget"]["methods"]) == {"reset"}


def test_class_keywords_are_inherited_and_extended(mcp):
    class Base(MCPStateDataModel, all_tools=True, tools={"rename": {"hidden": True}}):
        label = Sync(str, "")
        secret = Sync(str, "")

        def rename(self, label: str) -> str:
            self.label = label
            return label

        def reset(self) -> str:
            return "reset"

    class Inheriting(Base):  # no keywords: same options
        pass

    class Extending(Base, all_fields=True, fields={"secret": {"hidden": True}}):
        def archive(self) -> str:
            return "archived"

    instances = {"base": Base(), "inheriting": Inheriting(), "extending": Extending()}
    for name, instance in instances.items():
        mcp.dataclass.expose_instance(instance, name=name)
    models = fn(mcp, "dataclass_schemas")()["models"]

    assert set(models["Base"]["fields"]) == set()  # all_fields off by default
    assert set(models["Base"]["methods"]) == {"reset"}
    assert models["Inheriting"] == {"extends": "Base", "fields": {}, "methods": {}}
    # all_fields replaced, fields= added to the inherited tools= entries
    assert models["Extending"]["extends"] == "Base"
    assert set(models["Extending"]["fields"]) == {"label"}
    assert set(models["Extending"]["methods"]) == {"archive"}  # reset is inherited


def test_catalog_lists_only_what_subclasses_change(mcp):
    class Shape(MCPStateDataModel, all_fields=True):
        """Something drawn in the scene."""

        color = Sync(str, "red")
        size = Sync(float, 1.0)

    class Abstract(Shape):  # never instantiated, only extended
        sides = Sync(int, 0)

    class Square(Abstract, description="A square, sides fixed at 4"):
        size = mcp_field(Sync(float, 1.0), "Edge length", schema={"minimum": 0})

    class Dot(Shape, fields={"size": {"hidden": True}}):
        pass

    mcp.dataclass.expose_instance([Square(), Dot()], name="shapes")
    models = fn(mcp, "dataclass_schemas")()["models"]

    assert models["Shape"]["description"] == "Something drawn in the scene."
    assert set(models["Shape"]["fields"]) == {"color", "size"}
    # the chain is listed down from Shape, though no Abstract instance exists
    assert models["Abstract"] == {
        "extends": "Shape",
        "fields": {"sides": {"type": "integer"}},
        "methods": {},
    }
    assert models["Square"] == {
        "extends": "Abstract",
        "description": "A square, sides fixed at 4",
        "fields": {"size": {"type": "number", "minimum": 0, "description": "Edge length"}},
        "methods": {},
    }
    # a subclass that hides an inherited field is listed in full
    assert "extends" not in models["Dot"]
    assert set(models["Dot"]["fields"]) == {"color"}


def test_model_description_option(mcp):
    class Plain(StateDataModel):
        """Developer notes the agent should not read."""

        value = Sync(int, 0)

    class Owned(MCPStateDataModel, description="Agent-facing text"):
        """Developer notes."""

        value = mcp_field(Sync(int, 0))

    class Child(Owned):  # descriptions are not inherited
        pass

    mcp.dataclass.expose_model(Plain, description="A plain value", all_fields=True)
    mcp.dataclass.expose_instance([Plain(), Owned(), Child()], name="all")
    models = fn(mcp, "dataclass_schemas")()["models"]
    assert models["Plain"]["description"] == "A plain value"
    assert models["Owned"]["description"] == "Agent-facing text"
    assert "description" not in models["Child"]
    with pytest.raises(TypeError, match="description must be a string"):
        mcp.dataclass.expose_model(Gadget, description=["x"])


def test_class_keywords_are_checked_when_the_class_is_created():
    with pytest.raises(ValueError, match="unknown options"):

        class Typo(MCPStateDataModel, tools={"go": {"readonly": True}}):
            def go(self):
                pass

    with pytest.raises(TypeError, match=r"unknown class keywords \['all_field'\]"):

        class Misspelled(MCPStateDataModel, all_field=True):
            value = Sync(int, 0)


async def test_expose_model_overrides_class_options(mcp, book):
    mcp.dataclass.expose_model(
        Person,
        fields={"age": {"read_only": True}, "height": {"hidden": True}},
        tools={"celebrate_birthday": {"hidden": True}},
    )
    fields = fn(mcp, "dataclass_schemas")()["models"]["Person"]["fields"]
    # markers still opt fields in; the app's entries override them key by key
    assert fields["name"]["description"] == "Full name"
    assert fields["age"] == {"type": "integer", "minimum": 0, "read_only": True}
    assert "height" not in fields

    with pytest.raises(LookupError, match="Unknown method"):
        await fn(mcp, "dataclass_call")(ada_id(mcp), "celebrate_birthday")


# -----------------------------------------------------------------------------
# Reading
# -----------------------------------------------------------------------------


def test_get_shallow_returns_refs(mcp, book):
    data = fn(mcp, "dataclass_get")("address_book")
    assert data["type"] == "AddressBook"
    assert "read_only" not in data
    assert data["values"]["title"] == "My contacts"
    ref = data["values"]["contacts"][0]
    assert set(ref) == {"id", "type"}
    assert ref["type"] == "Person"


def test_get_depth_inlines(mcp, book):
    data = fn(mcp, "dataclass_get")("address_book", depth=2)
    inlined = data["values"]["contacts"][0]
    assert inlined["values"]["name"] == "Ada"
    assert inlined["values"]["score"] == 0
    assert "notes" not in inlined["values"]
    assert "unmarked" not in inlined["values"]


def test_get_by_id_and_unreachable(mcp, book):
    get = fn(mcp, "dataclass_get")
    assert get(ada_id(mcp))["values"]["name"] == "Ada"
    with pytest.raises(LookupError, match="Unknown target"):
        get(Person(name="Eve")._id)


def test_hidden_reference_field_blocks_reachability(mcp, book):
    ada = book.contacts[0]
    ada.home = Address(street="1 Main St")
    ada.vault = Address(street="secret")

    get = fn(mcp, "dataclass_get")
    assert get(ada.home._id)["values"]["street"] == "1 Main St"
    with pytest.raises(LookupError):
        get(ada.vault._id)


def test_unexposed_classes_are_invisible(mcp):
    gadget = Gadget(label="g")
    shelf = Shelf(gadgets=[gadget])
    mcp.dataclass.expose_instance(shelf, name="shelf")

    get = fn(mcp, "dataclass_get")
    assert get("shelf")["values"]["gadgets"] == []
    with pytest.raises(LookupError):
        get(gadget._id)
    assert "Gadget" not in fn(mcp, "dataclass_schemas")()["models"]

    mcp.dataclass.expose_model(Gadget, fields=["label"])
    assert get("shelf")["values"]["gadgets"] == [{"id": gadget._id, "type": "Gadget"}]
    assert get(gadget._id)["values"] == {"label": "g"}


# -----------------------------------------------------------------------------
# Entry points
# -----------------------------------------------------------------------------


async def test_live_entry_points(mcp):
    class App:
        pass

    app = App()
    app.person = Person(name="First")
    app.selection = []
    mcp.dataclass.expose_instance(lambda: app.person, name="person")
    mcp.dataclass.expose_instance(lambda: app.selection, name="selection")
    get = fn(mcp, "dataclass_get")

    assert get("person")["values"]["name"] == "First"
    app.person = Person(name="Second")
    assert get("person")["values"]["name"] == "Second"

    assert get("selection") == {"name": "selection", "items": []}
    app.selection = [app.person, Gadget()]  # Gadget is not exposed
    assert get("selection") == {
        "name": "selection",
        "items": [{"id": app.person._id, "type": "Person"}],
    }
    items = get("selection", depth=2)["items"]
    assert items[0]["values"]["name"] == "Second"
    assert fn(mcp, "dataclass_schemas")()["instances"]["selection"] == {
        "collection": "list",
        "size": 1,
    }

    with pytest.raises(LookupError, match="collection"):
        await fn(mcp, "dataclass_update")("selection", {"name": "x"})

    app.person = None
    with pytest.raises(LookupError, match="currently empty"):
        get("person")


def test_static_dict_entry_point(mcp):
    people = {"lead": Person(name="Lead")}
    mcp.dataclass.expose_instance(people, name="team")
    team = fn(mcp, "dataclass_get")("team")
    assert team["items"] == {"lead": {"id": people["lead"]._id, "type": "Person"}}


def test_direct_instance_is_held(mcp):
    mcp.dataclass.expose_instance(Person(name="Guest"), name="guest")
    gc.collect()
    assert fn(mcp, "dataclass_get")("guest")["values"]["name"] == "Guest"


def test_unexpose_instance(mcp, book):
    mcp.dataclass.unexpose_instance("address_book")
    with pytest.raises(LookupError):
        fn(mcp, "dataclass_get")("address_book")
    with pytest.raises(KeyError):
        mcp.dataclass.unexpose_instance("address_book")


# -----------------------------------------------------------------------------
# Writing and calling
# -----------------------------------------------------------------------------


async def test_update_targeted_patch(mcp, book):
    result = await fn(mcp, "dataclass_update")(ada_id(mcp), {"age": 37, "nickname": None})
    assert result["updated"] == ["age", "nickname"]
    assert book.contacts[0].age == 37


async def test_update_validates_types_all_or_nothing(mcp, book):
    update = fn(mcp, "dataclass_update")
    target = ada_id(mcp)

    with pytest.raises(ValueError, match="age: expected integer, got string"):
        await update(target, {"name": "Ada L.", "age": "old"})
    with pytest.raises(ValueError, match="at least 0"):
        await update(target, {"age": -1})
    with pytest.raises(ValueError, match="expected string, got null"):
        await update(target, {"name": None})
    assert book.contacts[0].name == "Ada"
    assert book.contacts[0].age == 36

    # lossless numeric coercion keeps descriptor types exact
    await update(target, {"height": 2, "age": 40.0})
    assert isinstance(book.contacts[0].height, float)
    assert book.contacts[0].age == 40
    assert isinstance(book.contacts[0].age, int)


async def test_update_rejects_fields_all_or_nothing(mcp, book):
    update = fn(mcp, "dataclass_update")
    target = ada_id(mcp)

    with pytest.raises(ValueError, match="nested model reference"):
        await update("address_book", {"contacts": []})
    with pytest.raises(ValueError, match="hint \\(read-only\\)"):
        await update(target, {"hint": "x"})
    # unmarked and hidden fields are indistinguishable from unknown ones
    for name in ("bogus", "unmarked", "notes", "vault"):
        with pytest.raises(ValueError, match=f"{name} \\(unknown field\\)"):
            await update(target, {name: "x"})

    with pytest.raises(ValueError, match="Nothing was applied"):
        await update("address_book", {"title": "New", "bogus": 1})
    assert book.title == "My contacts"


async def test_call_structural_change(mcp, book):
    call = fn(mcp, "dataclass_call")
    result = await call("address_book", "add_contact", {"name": "Grace", "age": 36})
    new_id = result["id"]
    assert len(book.contacts) == 2
    assert fn(mcp, "dataclass_get")(new_id)["values"]["name"] == "Grace"

    # custom tool name works, method name does not
    await call("address_book", "remove", {"contact_id": new_id})
    assert len(book.contacts) == 1
    with pytest.raises(LookupError, match="Unknown method"):
        await call("address_book", "remove_contact", {"contact_id": new_id})
    with pytest.raises(LookupError, match="Unknown method"):
        await call(ada_id(mcp), "unmarked_method")


async def test_call_validates_method_arguments(mcp, book):
    call = fn(mcp, "dataclass_call")
    with pytest.raises(ValueError, match=r"args\.age: expected integer"):
        await call("address_book", "add_contact", {"name": "Grace", "age": "36"})
    with pytest.raises(ValueError, match="unexpected 'bogus'"):
        await call("address_book", "add_contact", {"name": "Grace", "bogus": 1})
    with pytest.raises(ValueError, match="missing required 'name'"):
        await call("address_book", "add_contact", {})
    assert len(book.contacts) == 1


async def test_read_only_entry(mcp):
    person = Person(name="Solo", age=1)
    mcp.dataclass.expose_instance(person, name="solo", read_only=True)

    assert fn(mcp, "dataclass_schemas")()["instances"]["solo"]["read_only"] is True
    assert fn(mcp, "dataclass_get")("solo")["read_only"] is True

    with pytest.raises(PermissionError, match="exposed read-only"):
        await fn(mcp, "dataclass_update")("solo", {"age": 2})
    call = fn(mcp, "dataclass_call")
    with pytest.raises(PermissionError, match=r"only read-only methods.*greeting"):
        await call("solo", "celebrate_birthday")
    assert person.age == 1
    assert await call("solo", "greeting", {"formal": True}) == "Good day, Solo"


async def test_writable_entry_wins_over_read_only_entry(mcp, book):
    ada = book.contacts[0]
    mcp.dataclass.expose_instance(ada, name="ada_view", read_only=True)
    await fn(mcp, "dataclass_update")("ada_view", {"age": 50})
    assert ada.age == 50


# -----------------------------------------------------------------------------
# Startup audit
# -----------------------------------------------------------------------------


def test_audit_reports_wiring_mistakes(mcp, caplog):
    mcp.dataclass.expose_instance(Shelf(), name="shelf")
    with caplog.at_level(logging.WARNING, logger="trame_mcp"):
        mcp.dataclass._audit()

    messages = "\n".join(caplog.messages)
    # visible reference to a class nobody exposed
    assert "Shelf.gadgets references models that are not exposed (Gadget)" in messages
    # exposed models behind an unmarked reference field
    assert "Shelf.people holds exposed models (Employee, Person)" in messages


class Thermostat(MCPStateDataModel, all_fields=True):
    target = Sync(float, 20.0)
    heating = Sync(bool, False)  # derived by a watcher

    @watch("target")
    def _on_target(self, target):
        self.heating = target > 20

    @tool
    def warm_up(self) -> float:
        """Raise the target by two degrees."""
        self.target += 2
        return self.target


async def test_update_and_call_return_once_watchers_ran(mcp):
    # trame-dataclass runs watchers in a task, after the write returns
    thermostat = Thermostat()
    mcp.dataclass.expose_instance(thermostat, name="thermostat")

    await fn(mcp, "dataclass_update")("thermostat", {"target": 22})
    assert thermostat.heating is True

    thermostat.target = 18
    await thermostat.completion()
    assert thermostat.heating is False
    await fn(mcp, "dataclass_call")("thermostat", "warm_up")
    await fn(mcp, "dataclass_call")("thermostat", "warm_up")
    assert thermostat.heating is True


async def test_update_writes_declared_tuples(mcp):
    class Camera(MCPStateDataModel):
        position = mcp_field(Sync(tuple[float, float, float], (0.0, 0.0, 1.0)))

    camera = Camera()
    mcp.dataclass.expose_instance(camera, name="camera")
    await fn(mcp, "dataclass_update")("camera", {"position": [0, 2, 5]})
    assert camera.position == (0.0, 2.0, 5.0)


async def test_a_failing_watcher_does_not_fail_the_update(mcp, caplog):
    class Fragile(MCPStateDataModel):
        level = mcp_field(Sync(int, 0))

        @watch("level")
        def _on_level(self, level):
            if level > 5:
                msg = "too high"
                raise RuntimeError(msg)

    fragile = Fragile()
    mcp.dataclass.expose_instance(fragile, name="fragile")
    with caplog.at_level(logging.ERROR, logger="trame_mcp"):
        result = await fn(mcp, "dataclass_update")("fragile", {"level": 9})
    assert result["updated"] == ["level"]
    assert fragile.level == 9
    assert "a watcher of Fragile" in caplog.text


async def test_call_honors_the_method_serialize_option(mcp):
    import asyncio

    class Probe(MCPStateDataModel):
        count = mcp_field(Sync(int, 0), read_only=True)

        @tool(read_only=True, serialize=False)
        def peek(self) -> int:
            """Read the count without waiting for other calls."""
            return self.count

        @tool
        def bump(self) -> int:
            """Count one more."""
            self.count += 1
            return self.count

    probe = Probe()
    mcp.dataclass.expose_instance(probe, name="probe")
    release = asyncio.Event()

    @mcp.tool
    async def hold():
        await release.wait()

    def call(method):
        return mcp.protocol.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "dataclass_call",
                    "arguments": {"target": "probe", "method": method},
                },
            }
        )

    holding = asyncio.ensure_future(
        mcp.protocol.handle(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "hold"}}
        )
    )
    await asyncio.sleep(0)

    response = await asyncio.wait_for(call("peek"), 1)
    assert response["result"]["content"][0]["text"] == "0"
    bumping = asyncio.ensure_future(call("bump"))
    await asyncio.sleep(0.01)
    assert not bumping.done()  # serialized: waits for hold
    assert probe.count == 0

    release.set()
    await asyncio.gather(holding, bumping)
    assert probe.count == 1


def test_expose_model_can_unserialize_a_method(mcp):
    class Queue(StateDataModel):
        size = Sync(int, 0)

        def drain(self) -> int:
            return self.size

    mcp.dataclass.expose_model(Queue, all_fields=True, tools={"drain": {"serialize": False}})
    policy = mcp.dataclass._policy(Queue)
    assert policy.tools["drain"].serialize is False


def test_a_reference_typed_with_a_base_model_class_reaches_its_subclasses(mcp):
    """A field declared with the abstract base (StateDataModel or
    MCPStateDataModel) reaches the exposed instances it holds; the bases
    themselves have no fields and are never described."""

    class Holder(MCPStateDataModel):
        anything = mcp_field(Sync(StateDataModel | None, None, has_dataclass=True))
        owned = mcp_field(Sync(list[MCPStateDataModel], list, has_dataclass=True))

    holder = Holder(anything=Address(street="1 Main St"), owned=[Person(name="Ada")])
    mcp.dataclass.expose_instance(holder, name="holder")

    catalog = fn(mcp, "dataclass_schemas")()
    assert {"Holder", "Address", "Person"} <= catalog["models"].keys()
    assert not {"StateDataModel", "MCPStateDataModel"} & catalog["models"].keys()
    assert catalog["models"]["Holder"]["fields"]["anything"].get("model") is None

    values = fn(mcp, "dataclass_get")("holder")["values"]
    assert values["anything"]["type"] == "Address"
    assert fn(mcp, "dataclass_get")(values["owned"][0]["id"])["values"]["name"] == "Ada"
