"""trame-dataclass address book exposed over MCP (DESIGN.md worked example).

Run:  python examples/02_address_book.py --server

An MCP client can then:
    dataclass_schemas()
    dataclass_get("address_book")
    dataclass_call("address_book", "add_contact", {"name": "Grace", "age": 36})
    dataclass_update("<person id>", {"age": 37})
"""

from trame.ui.html import DivLayout
from trame.widgets import html
from trame_dataclass.v2 import ServerOnly, Sync

from trame_mcp import MCPApp, MCPStateDataModel, mcp_field, tool


class Person(MCPStateDataModel):
    """A contact in the address book."""

    name = mcp_field(Sync(str, ""), "Full name", schema={"minLength": 1})
    age = mcp_field(Sync(int, 0), "Age in years", schema={"minimum": 0, "maximum": 150})
    notes = ServerOnly(str)  # not marked: invisible to MCP

    @tool
    def celebrate_birthday(self) -> str:
        """Increment this person's age by one year."""
        self.age += 1
        return f"{self.name} is now {self.age}"

    @tool(read_only=True)
    def summary(self) -> str:
        """One-line description of this contact."""
        return f"{self.name}, {self.age}"


class AddressBook(MCPStateDataModel):
    title = mcp_field(Sync(str, "My contacts"), "Title shown above the list")
    contacts = mcp_field(Sync(list[Person], list, has_dataclass=True))

    @tool
    def add_contact(self, name: str, age: int = 0) -> dict:
        """Create a new contact and add it to the book."""
        person = Person(self.server, name=name, age=age)
        self.contacts = [*self.contacts, person]
        return {"id": person._id}

    @tool
    def remove_contact(self, contact_id: str) -> str:
        """Remove a contact from the book."""
        self.contacts = [p for p in self.contacts if p._id != contact_id]
        return "removed"


class ContactsApp(MCPApp):
    def __init__(self, server=None):
        super().__init__(server)
        self.mcp.configure(name="address-book", enabled=True)
        self.book = AddressBook(self.server)
        self.book.contacts = [Person(self.server, name="Ada", age=36)]
        self.mcp.dataclass.expose_instance(self.book, name="address_book")
        self._build_ui()

    def _build_ui(self):
        with DivLayout(self.server):
            with self.book.provide_as("book"):
                html.H1("{{ book.title }}")
                html.Div(
                    "{{ contact.name }} ({{ contact.age }})",
                    v_for="contact in book.contacts",
                    key="contact._id",
                )


def main():
    app = ContactsApp()
    app.server.start()


if __name__ == "__main__":
    main()
