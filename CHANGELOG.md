# Changelog

## [0.1.6] - 2026-01-02

### Added
- One-click install buttons for Kiro and Claude Desktop

### Changed
- Clarified tool descriptions apply to Jupyter Notebooks (.ipynb)
- Updated to official Kiro badge and correct URL format
- Updated dependencies

## [0.1.5] - 2025-11-23

### Fixed
- Add contents write permission for release uploads

## [0.1.4] - 2025-11-23

### Added
- CLI arguments and project scoping
- Upload build artifacts to GitHub releases
- PyPI publishing workflow and documentation

### Changed
- Improved tool descriptions and PyPI metadata
- Updated dependencies

## [0.1.3] - 2025-10-27

### Changed
- Clarified that cell content should be raw strings in documentation

## [0.1.2] - 2025-10-27

### Changed
- Clarified that cell content should be raw strings

## [0.1.1] - 2025-10-26

### Fixed
- Handle notebook format versions when inserting cells

### Changed
- Recommend absolute paths for all filepath parameters

## [0.1.0] - 2025-10-26

### Initial Release

Complete implementation of Jupyter Notebook Editor MCP Server with 29 specialized tools for programmatic notebook manipulation.

### Features

#### Read Operations (4 tools)
- `read_notebook` - Read notebook structure and metadata
- `list_cells` - List all cells with previews
- `get_cell` - Get specific cell content by index
- `search_cells` - Search for patterns across cells

#### Cell Modification (5 tools)
- `replace_cell` - Replace entire cell content
- `insert_cell` - Insert new cell at position
- `append_cell` - Append cell to end
- `delete_cell` - Delete cell by index
- `str_replace_in_cell` - Replace substring within cell

#### Metadata Operations (4 tools)
- `get_metadata` - Get notebook or cell metadata
- `update_metadata` - Update notebook or cell metadata
- `set_kernel` - Set kernel specification
- `list_available_kernels` - List common kernel configurations

#### Batch Operations - Multi-Cell (6 tools)
- `replace_cells_batch` - Replace multiple cells at once
- `delete_cells_batch` - Delete multiple cells by indices
- `insert_cells_batch` - Insert multiple cells in one operation
- `search_replace_all` - Find/replace across all cells
- `reorder_cells` - Reorganize cell order
- `filter_cells` - Keep only cells matching criteria

#### Batch Operations - Multi-Notebook (7 tools)
- `merge_notebooks` - Combine multiple notebooks
- `split_notebook` - Split notebook by criteria
- `apply_to_notebooks` - Apply operation to multiple notebooks
- `search_notebooks` - Search across multiple notebooks
- `sync_metadata` - Synchronize metadata across notebooks
- `extract_cells` - Extract matching cells into new notebook
- `clear_outputs` - Clear outputs from code cells

#### Validation (3 tools)
- `validate_notebook` - Validate notebook structure
- `get_notebook_info` - Get notebook summary information
- `validate_notebooks_batch` - Validate multiple notebooks

### Technical Details

- **Language**: Python 3.10+
- **Framework**: FastMCP for MCP server implementation
- **Library**: nbformat for notebook manipulation
- **Test Coverage**: 92% on core operations
- **Tests**: 52 passing unit and integration tests
- **Format Preservation**: Automatic validation after all modifications

### Installation

#### Option 1: Install as a Tool (Recommended)

```bash
# Install from local directory
uv tool install --from /path/to/jupyter-editor .

# Or install from git repository
uv tool install git+https://github.com/jsamuel1/jupyter-editor-mcp.git
```

Run the server:

```bash
jupyter-editor-mcp
```

#### Option 2: Development Installation

```bash
uv venv
uv pip install -e ".[dev]"
```

### Usage with Claude Desktop

#### If installed as a tool:

```json
{
  "mcpServers": {
    "jupyter-editor": {
      "command": "jupyter-editor-mcp"
    }
  }
}
```

#### If installed in development mode:

```json
{
  "mcpServers": {
    "jupyter-editor": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/jupyter-editor-mcp",
        "python",
        "-m",
        "jupyter_editor.server"
      ]
    }
  }
}
```

### Documentation

- [README.md](README.md) - Overview and quick start
- [docs/RESEARCH.md](docs/RESEARCH.md) - Technical research and specifications
- [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) - User stories and acceptance criteria
- [docs/DESIGN.md](docs/DESIGN.md) - Architecture and API design

### Contributors

Built following specification-driven development with TDD approach.
