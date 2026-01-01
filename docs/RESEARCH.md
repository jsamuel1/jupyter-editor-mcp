# Jupyter Notebook Editor MCP Server - Research

## Overview

A Model Context Protocol (MCP) server for editing Jupyter notebooks programmatically without breaking their format. Built with FastMCP and nbformat libraries in Python.

## Core Technologies

### FastMCP

**Purpose**: Framework for building MCP servers in Python

**Key Features**:
- Decorator-based tool definition (`@mcp.tool`)
- Automatic JSON schema generation from type hints
- Built-in STDIO transport for local clients
- Simple server creation with `FastMCP(name="...")`
- Automatic docstring-to-description conversion

**Basic Pattern**:
```python
from fastmcp import FastMCP

mcp = FastMCP(name="Server Name")

@mcp.tool
def tool_name(param: str) -> str:
    """Tool description for LLM."""
    return result

if __name__ == "__main__":
    mcp.run()
```

### nbformat

**Purpose**: Reference implementation for Jupyter Notebook format manipulation

**Key Features**:
- Read/write notebook files: `nbformat.read()`, `nbformat.write()`
- Create notebooks: `nbformat.v4.new_notebook()`
- Create cells: `nbformat.v4.new_code_cell()`, `nbformat.v4.new_markdown_cell()`, `nbformat.v4.new_raw_cell()`
- Access cells: `nb['cells']` (list of cell objects)
- Modify cell content: `cell['source']`
- Manage metadata: `nb.metadata`, `cell.metadata`
- Validate notebooks: `nbformat.validate()`

**Notebook Structure**:
```python
{
    'cells': [
        {
            'cell_type': 'code' | 'markdown' | 'raw',
            'source': 'cell content',
            'metadata': {},
            'execution_count': int | None,  # code cells only
            'outputs': []  # code cells only
        }
    ],
    'metadata': {
        'kernelspec': {
            'name': 'kernel_name',
            'display_name': 'Display Name',
            'language': 'python'
        },
        'language_info': {...}
    },
    'nbformat': 4,
    'nbformat_minor': 5
}
```

## Tool Design

### Design Principles

1. **File-based operations**: All tools take `filepath` as first parameter
2. **No server connection**: Pure file manipulation, no Jupyter server required
3. **Format preservation**: Maintain notebook structure and validity
4. **Type safety**: Use type hints for automatic schema generation
5. **Clear descriptions**: Docstrings guide LLM tool usage

### Proposed Tools

**Tool Categories Summary**:
- Read Operations: 4 tools
- Cell Modification: 5 tools  
- Metadata Operations: 4 tools
- Batch Operations (Multi-Cell): 6 tools
- Batch Operations (Multi-Notebook): 7 tools
- Validation: 3 tools
- **Total: 29 tools**

#### 1. Read Operations

**read_notebook**
- Read entire notebook structure
- Return formatted summary of cells and metadata
- Parameters: `filepath: str`

**list_cells**
- List all cells with indices, types, and preview
- Parameters: `filepath: str`

**get_cell**
- Get specific cell content by index
- Parameters: `filepath: str, cell_index: int`

**search_cells**
- Search cell content for pattern
- Parameters: `filepath: str, pattern: str, case_sensitive: bool = False`

#### 2. Cell Modification

**replace_cell**
- Replace entire cell content
- Parameters: `filepath: str, cell_index: int, new_content: str`

**insert_cell**
- Insert new cell at position
- Parameters: `filepath: str, cell_index: int, content: str, cell_type: str = "code"`

**append_cell**
- Append cell to end
- Parameters: `filepath: str, content: str, cell_type: str = "code"`

**delete_cell**
- Remove cell by index
- Parameters: `filepath: str, cell_index: int`

**str_replace_in_cell**
- Replace substring within cell (similar to fs_write str_replace)
- Parameters: `filepath: str, cell_index: int, old_str: str, new_str: str`

#### 3. Metadata Operations

**get_metadata**
- Get notebook or cell metadata
- Parameters: `filepath: str, cell_index: int | None = None`

**update_metadata**
- Update notebook or cell metadata
- Parameters: `filepath: str, metadata: dict, cell_index: int | None = None`

**set_kernel**
- Set kernel specification
- Parameters: `filepath: str, kernel_name: str, display_name: str, language: str = "python"`

**list_available_kernels**
- List common kernel configurations
- Parameters: None (returns predefined list)

#### 4. Batch Operations - Multi-Cell

**replace_cells_batch**
- Replace multiple cells in one operation
- Parameters: `filepath: str, replacements: list[dict]`
  - `replacements`: `[{"cell_index": 0, "content": "..."}, ...]`

**delete_cells_batch**
- Delete multiple cells by indices
- Parameters: `filepath: str, cell_indices: list[int]`

**insert_cells_batch**
- Insert multiple cells at specified positions
- Parameters: `filepath: str, insertions: list[dict]`
  - `insertions`: `[{"cell_index": 0, "content": "...", "cell_type": "code"}, ...]`

**search_replace_all**
- Search and replace across all cells
- Parameters: `filepath: str, pattern: str, replacement: str, cell_type: str | None = None`

**reorder_cells**
- Reorder cells by providing new index mapping
- Parameters: `filepath: str, new_order: list[int]`

**filter_cells**
- Keep only cells matching criteria, delete others
- Parameters: `filepath: str, cell_type: str | None = None, pattern: str | None = None`

#### 5. Batch Operations - Multi-Notebook

**merge_notebooks**
- Merge multiple notebooks into one
- Parameters: `output_filepath: str, input_filepaths: list[str], add_separators: bool = True`

**split_notebook**
- Split notebook into multiple files by criteria
- Parameters: `filepath: str, output_dir: str, split_by: str = "markdown_headers"`
  - `split_by`: "markdown_headers", "cell_count", "cell_indices"

**apply_to_notebooks**
- Apply same operation to multiple notebooks
- Parameters: `filepaths: list[str], operation: str, **operation_params`
  - `operation`: "set_kernel", "delete_outputs", "add_metadata", etc.

**search_notebooks**
- Search across multiple notebooks
- Parameters: `filepaths: list[str], pattern: str, return_context: bool = True`

**sync_metadata**
- Synchronize metadata across multiple notebooks
- Parameters: `filepaths: list[str], metadata: dict, merge: bool = False`

**extract_cells**
- Extract matching cells from multiple notebooks into new notebook
- Parameters: `output_filepath: str, input_filepaths: list[str], pattern: str | None = None, cell_type: str | None = None`

**clear_outputs**
- Clear all outputs from code cells in one or more notebooks
- Parameters: `filepaths: str | list[str]`
- Clears outputs array and resets execution_count to None

#### 6. Validation

**validate_notebook**
- Validate notebook structure
- Parameters: `filepath: str`

**get_notebook_info**
- Get summary: cell count, kernel, format version
- Parameters: `filepath: str`

**validate_notebooks_batch**
- Validate multiple notebooks, return status for each
- Parameters: `filepaths: list[str]`

## Implementation Strategy

### Project Structure

```
jupyter-editor/
├── src/
│   └── jupyter_editor/
│       ├── __init__.py
│       ├── server.py          # FastMCP server definition
│       ├── operations.py      # Core notebook operations
│       └── utils.py           # Helper functions
├── tests/
│   ├── test_operations.py
│   └── fixtures/
│       └── sample.ipynb
├── pyproject.toml
├── README.md
└── RESEARCH.md
```

### Core Operations Module

Separate business logic from MCP server:

```python
# operations.py
import nbformat
from pathlib import Path

def read_notebook_file(filepath: str):
    """Read and return notebook object."""
    with open(filepath, 'r') as f:
        return nbformat.read(f, as_version=4)

def write_notebook_file(filepath: str, nb):
    """Write notebook object to file."""
    with open(filepath, 'w') as f:
        nbformat.write(nb, f)

def get_cell_content(filepath: str, cell_index: int) -> str:
    """Get content of specific cell."""
    nb = read_notebook_file(filepath)
    return nb['cells'][cell_index]['source']

# ... more operations
```

### Server Module

Thin wrapper exposing operations as MCP tools:

```python
# server.py
from fastmcp import FastMCP
from . import operations

mcp = FastMCP(name="Jupyter Notebook Editor")

@mcp.tool
def read_notebook(filepath: str) -> dict:
    """Read Jupyter notebook and return structure summary."""
    return operations.read_notebook_summary(filepath)

@mcp.tool
def get_cell(filepath: str, cell_index: int) -> str:
    """Get content of specific cell by index."""
    return operations.get_cell_content(filepath, cell_index)

# ... more tools
```

## Error Handling

### Common Errors

1. **File not found**: Invalid filepath
2. **Invalid notebook**: Corrupted or non-notebook file
3. **Index out of range**: Invalid cell_index
4. **Validation errors**: Malformed notebook structure
5. **Permission errors**: Cannot read/write file

### Error Strategy

```python
@mcp.tool
def get_cell(filepath: str, cell_index: int) -> str:
    """Get content of specific cell by index."""
    try:
        return operations.get_cell_content(filepath, cell_index)
    except FileNotFoundError:
        return f"Error: Notebook file not found: {filepath}"
    except IndexError:
        return f"Error: Cell index {cell_index} out of range"
    except Exception as e:
        return f"Error: {str(e)}"
```

## Testing Strategy

### Test Categories

1. **Unit tests**: Individual operations
2. **Integration tests**: Full tool workflows
3. **Fixture notebooks**: Various notebook structures
4. **Validation tests**: Format preservation

### Sample Test

```python
def test_insert_cell():
    # Create test notebook
    nb = nbformat.v4.new_notebook()
    nb['cells'].append(nbformat.v4.new_code_cell("print('first')"))
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.ipynb', delete=False) as f:
        nbformat.write(nb, f)
        filepath = f.name
    
    # Test insert
    operations.insert_cell(filepath, 1, "print('second')", "code")
    
    # Verify
    nb = operations.read_notebook_file(filepath)
    assert len(nb['cells']) == 2
    assert nb['cells'][1]['source'] == "print('second')"
```

## Installation & Dependencies

### pyproject.toml

```toml
[project]
name = "jupyter-editor-mcp"
version = "0.1.0"
description = "MCP server for editing Jupyter notebooks"
requires-python = ">=3.10"
dependencies = [
    "fastmcp>=0.1.0",
    "nbformat>=5.10.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Installation

```bash
# Create virtual environment
uv venv

# Install in development mode
uv pip install -e ".[dev]"
```

## Usage Examples

### With Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "jupyter-editor": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/jupyter-editor",
        "python",
        "-m",
        "jupyter_editor.server"
      ]
    }
  }
}
```

### Example Interactions

**User**: "Add a new markdown cell at the beginning of my notebook explaining what it does"

**LLM uses**: `insert_cell(filepath="notebook.ipynb", cell_index=0, content="# Overview\nThis notebook...", cell_type="markdown")`

**User**: "Change the kernel to Python 3.11"

**LLM uses**: `set_kernel(filepath="notebook.ipynb", kernel_name="python3", display_name="Python 3.11")`

**User**: "Find all cells that import pandas"

**LLM uses**: `search_cells(filepath="notebook.ipynb", pattern="import pandas")`

**User**: "Replace all occurrences of 'old_function' with 'new_function' across all code cells"

**LLM uses**: `search_replace_all(filepath="notebook.ipynb", pattern="old_function", replacement="new_function", cell_type="code")`

**User**: "Delete cells 2, 5, and 7 from my notebook"

**LLM uses**: `delete_cells_batch(filepath="notebook.ipynb", cell_indices=[2, 5, 7])`

**User**: "Merge these three notebooks into one: analysis.ipynb, visualization.ipynb, conclusions.ipynb"

**LLM uses**: `merge_notebooks(output_filepath="combined.ipynb", input_filepaths=["analysis.ipynb", "visualization.ipynb", "conclusions.ipynb"])`

**User**: "Set all notebooks in this directory to use the same Python 3.11 kernel"

**LLM uses**: `apply_to_notebooks(filepaths=["nb1.ipynb", "nb2.ipynb", "nb3.ipynb"], operation="set_kernel", kernel_name="python3", display_name="Python 3.11")`

**User**: "Find all notebooks that mention 'machine learning' in any cell"

**LLM uses**: `search_notebooks(filepaths=["*.ipynb"], pattern="machine learning")`

**User**: "Clear all outputs from my notebook before committing to git"

**LLM uses**: `clear_outputs(filepaths="analysis.ipynb")`

**User**: "Clear outputs from all notebooks in this directory"

**LLM uses**: `clear_outputs(filepaths=["nb1.ipynb", "nb2.ipynb", "nb3.ipynb"])`

## Batch Operations Use Cases

### Multi-Cell Operations

**Refactoring workflows**:
- Replace function names across multiple cells
- Update import statements in all code cells
- Remove debug print statements from entire notebook

**Cleanup operations**:
- Delete all empty cells
- Remove cells with specific tags
- Filter out cells matching patterns

**Reorganization**:
- Reorder cells to group related code
- Move all imports to top
- Consolidate markdown documentation

### Multi-Notebook Operations

**Project-wide updates**:
- Update kernel across all notebooks in a project
- Synchronize common metadata (author, version)
- Apply consistent formatting standards

**Content aggregation**:
- Merge analysis notebooks into final report
- Extract all visualization cells into gallery notebook
- Combine tutorial notebooks into comprehensive guide

**Batch processing**:
- Validate all notebooks in directory
- Search for deprecated API usage across project
- Extract code cells for testing

**Documentation generation**:
- Extract markdown cells for documentation
- Create index of all notebooks with summaries
- Generate table of contents from headers

## Comparison with fs_read/fs_write

### Why Specialized Tools?

| Aspect | fs_read/fs_write | Jupyter Editor MCP |
|--------|------------------|-------------------|
| Format awareness | None (raw JSON) | Full notebook structure |
| Cell operations | Manual JSON manipulation | High-level cell operations |
| Validation | None | Automatic validation |
| Metadata | Manual dict updates | Structured metadata API |
| Error risk | High (easy to break format) | Low (format-preserving) |
| Usability | Complex for LLMs | Simple, purpose-built |

### Example Comparison

**Task**: Insert a cell at position 2

**With fs_read/fs_write**:
1. Read entire file as JSON
2. Parse JSON structure
3. Navigate to cells array
4. Insert new cell dict with correct structure
5. Ensure all required fields present
6. Write back entire JSON
7. Hope format is still valid

**With Jupyter Editor MCP**:
```python
insert_cell(filepath="notebook.ipynb", cell_index=2, content="print('hello')")
```

## Next Steps

1. **Implement core operations module** with nbformat
2. **Create FastMCP server** wrapping operations
3. **Add comprehensive tests** with fixture notebooks
4. **Document all tools** with clear examples
5. **Test with Claude Desktop** for real-world usage
6. **Add advanced features** (cell reordering, bulk operations)

## References

- [FastMCP Documentation](https://gofastmcp.com/)
- [nbformat Documentation](https://nbformat.readthedocs.io/)
- [MCP Specification](https://modelcontextprotocol.io/specification/)
- [Jupyter Notebook Format](https://nbformat.readthedocs.io/en/latest/format_description.html)
