"""Regression tests for the MCP tool contract exposed through FastMCP.

These pin behaviour that a FastMCP / MCP SDK upgrade can silently change:
the tool set, the LLM-visible descriptions (including index-shift safety
guidance), argument validation, project-scope enforcement through the MCP
layer, and the error-dict convention.
"""

import nbformat
import pytest
from fastmcp import Client

from jupyter_editor import server
from jupyter_editor import operations as server_operations

EXPECTED_TOOLS = {
    "ipynb_append_cell",
    "ipynb_apply_to_notebooks",
    "ipynb_clear_outputs",
    "ipynb_delete_cell",
    "ipynb_delete_cells_batch",
    "ipynb_extract_cells",
    "ipynb_filter_cells",
    "ipynb_get_cell",
    "ipynb_get_metadata",
    "ipynb_get_notebook_info",
    "ipynb_insert_cell",
    "ipynb_insert_cells_batch",
    "ipynb_list_available_kernels",
    "ipynb_list_cells",
    "ipynb_merge_notebooks",
    "ipynb_read_notebook",
    "ipynb_reorder_cells",
    "ipynb_replace_cell",
    "ipynb_replace_cells_batch",
    "ipynb_search_cells",
    "ipynb_search_notebooks",
    "ipynb_search_replace_all",
    "ipynb_set_kernel",
    "ipynb_split_notebook",
    "ipynb_str_replace_in_cell",
    "ipynb_sync_metadata",
    "ipynb_update_metadata",
    "ipynb_validate_notebook",
    "ipynb_validate_notebooks_batch",
}

# Guidance an LLM needs to avoid corrupting cell positions. FastMCP 4 drops
# docstring sections other than the leading prose (e.g. ``Note:``) from the
# tool description, so this guidance must live in the prose paragraph.
INDEX_SHIFT_GUIDANCE = {
    "ipynb_insert_cell": "shift by +1",
    "ipynb_delete_cell": "shift by -1",
    "ipynb_delete_cells_batch": "descending order",
    "ipynb_insert_cells_batch": "later indices",
}


@pytest.fixture(autouse=True)
def reset_server_scope():
    """Reset the scope on the module the server actually uses."""
    server_operations._project_scope = None
    yield
    server_operations._project_scope = None


@pytest.fixture
def notebook_path(tmp_path):
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_code_cell("x = 1"),
        nbformat.v4.new_markdown_cell("# Title"),
    ]
    path = tmp_path / "nb.ipynb"
    nbformat.write(nb, str(path))
    return path


async def _tools():
    async with Client(server.mcp) as client:
        return {t.name: t for t in await client.list_tools()}


@pytest.mark.asyncio
async def test_tool_set_is_unchanged():
    assert set(await _tools()) == EXPECTED_TOOLS


@pytest.mark.asyncio
async def test_every_parameter_has_a_description():
    for name, tool in (await _tools()).items():
        for param, schema in tool.input_schema["properties"].items():
            assert schema.get("description"), f"{name}.{param} lost its description"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name,phrase", sorted(INDEX_SHIFT_GUIDANCE.items()))
async def test_index_shift_guidance_is_visible(tool_name, phrase):
    tool = (await _tools())[tool_name]
    assert phrase in tool.description


@pytest.mark.asyncio
async def test_instructions_describe_error_convention():
    # Handshake-era clients (the stdio clients in use today) receive the
    # instructions in the initialize result.
    async with Client(server.mcp, mode="legacy") as client:
        result = client.initialize_result
    assert "'error' key" in result.instructions
    assert "clear_outputs" in result.instructions


@pytest.mark.asyncio
async def test_unknown_arguments_are_rejected(notebook_path):
    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "ipynb_get_cell",
            {"ipynb_filepath": str(notebook_path), "cell_index": 0, "bogus": 1},
            raise_on_error=False,
        )
    assert result.is_error


@pytest.mark.asyncio
async def test_edit_round_trip_preserves_valid_notebook(notebook_path):
    async with Client(server.mcp) as client:
        inserted = await client.call_tool(
            "ipynb_insert_cell",
            {"ipynb_filepath": str(notebook_path), "cell_index": 1, "content": "y = 2"},
        )
        assert inserted.data == {"success": True, "new_cell_count": 3}
        cell = await client.call_tool(
            "ipynb_get_cell", {"ipynb_filepath": str(notebook_path), "cell_index": 1}
        )
    assert cell.data == {"content": "y = 2"}
    nbformat.validate(nbformat.read(str(notebook_path), as_version=4))


@pytest.mark.asyncio
async def test_missing_notebook_returns_error_dict(tmp_path):
    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "ipynb_read_notebook", {"ipynb_filepath": str(tmp_path / "missing.ipynb")}
        )
    assert not result.is_error
    assert "error" in result.data


@pytest.mark.asyncio
async def test_project_scope_enforced_through_mcp(tmp_path, notebook_path):
    project = tmp_path / "project"
    project.mkdir()
    server_operations.set_project_scope(str(project))
    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "ipynb_replace_cell",
            {"ipynb_filepath": str(notebook_path), "cell_index": 0, "new_content": "pwned"},
        )
    assert "error" in result.data
    assert "outside project scope" in result.data["error"]
    nb = nbformat.read(str(notebook_path), as_version=4)
    assert nb.cells[0].source == "x = 1"
