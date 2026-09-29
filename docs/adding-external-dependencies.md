# Adding an External Package Dependency to a Custom Tool

The [Development](../README.md#development) section covers the basic pattern for adding a new tool: create a file in `src/mcp_server/tools/`, follow `_template.py`, and it's auto-discovered on startup. That covers tools that only need what's already installed.

This guide covers the next step: a tool that needs an **external Python package** — either a public package from PyPI, or your own separate local project — as part of its logic.

## Option A: A public package from PyPI

Add it to `pyproject.toml`'s `dependencies`, then re-sync:

```bash
uv add humanize
```

This updates `pyproject.toml` and `uv.lock` automatically. Now it's importable from any tool file, same as any other dependency.

## Option B: Your own separate local project

If your tool's logic lives in its own project (for example, you're wrapping an existing internal library, or you want to keep substantial business logic out of the MCP server repo entirely), `uv` supports a **local path dependency**. Add it to `pyproject.toml`:

```toml
[project]
dependencies = [
    # ...existing dependencies...
    "your-package-name",
]

[tool.uv.sources]
your-package-name = { path = "../your-package-name", editable = true }
```

`path` is relative to this repo's root — the example above assumes your other project is cloned as a **sibling directory** (both under the same parent folder). `editable = true` means changes to that project's source take effect immediately on the next server restart, without reinstalling.

Then:
```bash
uv sync
```
Check the output for a line like `+ your-package-name==0.1.0 (from file:///path/to/your-package-name)` — that confirms it resolved and installed correctly.

**Note**: a tool built this way isn't self-contained within this repo — anyone else running the server (including CI) needs that sibling project cloned too. This is fine for local/personal extensions, but isn't suitable for a tool you intend to upstream in a PR, since the dependency path won't exist on another machine.

## Example

See `src/mcp_server/tools/_example_external_dependency.py` — a minimal, self-contained example tool using `humanize` (Option A) to convert a raw byte count into a human-readable string (e.g. `1500000` → `"1.5 MB"`). It's prefixed with `_` like `_template.py`, so it's excluded from auto-discovery and won't register as a live tool — it's a reference only. Copy its structure (without the leading `_`) as a starting point for your own tool.
