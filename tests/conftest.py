import itertools

import pytest

_counter = itertools.count()


@pytest.fixture
def trame_server():
    """A fresh, uniquely-named trame server per test."""
    from trame.app import get_server

    return get_server(f"trame_mcp_test_{next(_counter)}")


@pytest.fixture
def mcp(trame_server):
    from trame_mcp import get_mcp

    return get_mcp(trame_server, enabled=True)
