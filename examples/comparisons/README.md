# Why trame-mcp: rebuilding `03_wasm_quadrants` on trame-llm

`03_wasm_quadrants_trame_llm.py` is a faithful port of
[`examples/03_wasm_quadrants.py`](../03_wasm_quadrants.py) that replaces
trame-mcp with trame-llm's MCP utilities (`LLMClient.register_mcp` +
FastMCP). The application layer (trame-dataclass models, VTK pipelines,
reactive UI) is copied verbatim, so every diff between the two files is
attributable to the MCP layer alone.

Both versions work. The question is what it costs to get there.

## Run it

```console
pip install -e ../trame-llm   # trame-llm + fastmcp + openai + ~70 packages
python examples/comparisons/03_wasm_quadrants_trame_llm.py --server --timeout 0

# trame UI on :8080, MCP on a SECOND server:
claude mcp add --transport http wasm-quadrants-llm http://localhost:9000/mcp
```

No API key is needed: trame-llm's `LLMClient` is instantiated only because it
is the sole owner of the FastMCP registration hook.

## Scorecard

|                          | trame-mcp version | trame-llm port |
|--------------------------|-------------------|----------------|
| total lines              | 687               | 1017           |
| MCP-specific lines       | ~30 (decorators, `expose_dataclass`, `MCPApp`) | ~360 (facade + serving) |
| tools exposed            | 4 generic (`dataclass_*`) | 12 hand-written |
| endpoint                 | mounted on the trame server itself (`/mcp`) | second web server (uvicorn) on a second port |
| schema discovery         | `dataclass_schemas`, generated from the models, always current | prose in `instructions` + tool docstrings, drifts by hand |
| instance addressing      | automatic (roots + reachability) | hand-rolled id registries and lookups |
| field access policy      | derived from `Sync` / `ClientOnly` / `ServerOnly` | hand-maintained tables (`OBJECT_WRITABLE`, `TYPE_EXTRA_FIELDS`, skip-lists) |
| adding a model field     | edit the model | edit the model + serializer + `update_object` signature + validation + docstring |
| ServerOnly safety        | hidden by policy | one forgotten field leaks a live VTK object into JSON |
| tool-call serialization  | lock across async tool calls | none; safe only while every facade tool stays await-free |
| event-loop safety        | tools run on the trame loop by construction | convention: every tool must be `async def`, uvicorn must be started on trame's loop — nothing enforces either |
| prompts                  | `@prompt` decorator | unsupported by `register_mcp`; registered by reaching into the private FastMCP |
| auth / CLI               | `--mcp-auth-key`, `--mcp-disable` | not wired (FastMCP auth is a separate system) |
| extra dependencies       | none beyond trame | fastmcp, openai, uvicorn/starlette, ~70 packages |
| private API reach-ins    | none | `LLMClient._mcps[0]`, behind-the-back `prompt()` registration |

## What trame-llm actually provided

Of the ~360 new lines, trame-llm contributed the `controller.mcp` registration
idiom — about ten lines of convenience over calling `FastMCP(...)` directly.
Everything else (serving, addressing, schemas, policy, validation) had to be
built in the example, which is precisely the layer trame-mcp packages as a
library. The two libraries are complementary, not redundant: trame-llm's value
is the chat widget and LLM-provider loop; trame-mcp's value is the MCP server.

## Verified behavior of the port

Driven over MCP (fastmcp client against `http://localhost:9000/mcp`):
tools/list and prompts/list, grid resize with scene reconciliation, add/update
per-type objects (cone resolution/wireframe, torus thickness, text content),
full and summary reads, VTK source-property introspection and writes, camera
reset, object removal, scene rename/background, grid shrink invalidating
stale ids, and the error paths (unknown ids, type-inapplicable fields,
read-only `type`), with no ServerOnly leakage. The trame UI stays reactive
throughout, exactly like the original.
