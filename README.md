# Jupyter Notebook Editor MCP Server

A Model Context Protocol (MCP) server for programmatically editing Jupyter notebooks while preserving their format and structure.

## Features

- **29 specialized tools** for notebook manipulation
- **File-based operations** - no Jupyter server required
- **Format preservation** - automatic validation after modifications
- **Batch operations** - modify multiple cells or notebooks at once
- **Type-safe** - full type hints for all operations

## Installation

### One-Click Install

[![Add to Kiro](https://kiro.dev/images/add-to-kiro.svg)](https://kiro.dev/launch/mcp/add?name=jupyter-editor&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22jupyter-editor-mcp%22%5D%7D)

[![Install in Claude Code](https://img.shields.io/badge/Claude_Code-Install-5865F2?style=for-the-badge&logo=anthropic&logoColor=white)](https://claude.ai/mcp/install?name=jupyter-editor&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22jupyter-editor-mcp%22%5D%7D)

### From PyPI

```bash
uv tool install jupyter-editor-mcp
jupyter-editor-mcp
```

### From Source

```bash
git clone https://github.com/jsamuel1/jupyter-editor-mcp.git
cd jupyter-editor-mcp
uv venv
uv pip install -e ".[dev]"
```

See [INSTALL.md](INSTALL.md) for detailed configuration options.

## Usage

### With Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "jupyter-editor": {
      "command": "jupyter-editor-mcp"
    }
  }
}
```

### Example Interactions

**Read a notebook:**
```
"Show me the structure of my notebook.ipynb"
```

**Insert a cell:**
```
"Add a markdown cell at the beginning explaining what this notebook does"
```

**Batch operations:**
```
"Replace all occurrences of 'old_function' with 'new_function' in all code cells"
```

**Multi-notebook:**
```
"Merge analysis.ipynb and visualization.ipynb into combined.ipynb"
```

## Tool Categories

- **Read Operations** (4 tools): read_notebook, list_cells, get_cell, search_cells
- **Cell Modification** (5 tools): replace_cell, insert_cell, append_cell, delete_cell, str_replace_in_cell
- **Metadata Operations** (4 tools): get_metadata, update_metadata, set_kernel, list_available_kernels
- **Batch Operations - Multi-Cell** (6 tools): replace_cells_batch, delete_cells_batch, insert_cells_batch, search_replace_all, reorder_cells, filter_cells
- **Batch Operations - Multi-Notebook** (7 tools): merge_notebooks, split_notebook, apply_to_notebooks, search_notebooks, sync_metadata, extract_cells, clear_outputs
- **Validation** (3 tools): validate_notebook, get_notebook_info, validate_notebooks_batch

## Development

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov

# Install in development mode
uv pip install -e ".[dev]"
```

## Documentation

- [docs/RESEARCH.md](docs/RESEARCH.md) - Technical research and tool specifications
- [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) - User stories and acceptance criteria
- [docs/DESIGN.md](docs/DESIGN.md) - Architecture and API design
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines

## License

MIT
