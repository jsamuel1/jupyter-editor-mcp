# Jupyter Notebook Editor MCP Server - Design Document

## 1. Architecture Overview

### 1.1 System Context

```
┌─────────────────────────────────────────────────────────────┐
│                      MCP Client (LLM)                       │
│              (Claude Desktop, VS Code, etc.)                │
└────────────────────────┬────────────────────────────────────┘
                         │ STDIO/JSON-RPC
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  FastMCP Framework                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              MCP Server Layer                        │  │
│  │  - Tool registration (@mcp.tool decorators)          │  │
│  │  - Request/response handling                         │  │
│  │  - Schema generation from type hints                 │  │
│  └────────────────────┬─────────────────────────────────┘  │
└─────────────────────────┼────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│              jupyter_editor Package                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  server.py - MCP Tool Wrappers                       │   │
│  │  - Error handling & formatting                       │   │
│  │  - Parameter validation                              │   │
│  │  - Result serialization                              │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       │                                      │
│  ┌────────────────────▼─────────────────────────────────┐   │
│  │  operations.py - Core Business Logic                 │   │
│  │  - Notebook I/O (read/write)                         │   │
│  │  - Cell manipulation                                 │   │
│  │  - Metadata management                               │   │
│  │  - Batch operations                                  │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       │                                      │
│  ┌────────────────────▼─────────────────────────────────┐   │
│  │  utils.py - Helper Functions                         │   │
│  │  - Validation helpers                                │   │
│  │  - Error formatting                                  │   │
│  │  - Constants (kernel specs)                          │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│                  nbformat Library                            │
│  - Notebook format validation                                │
│  - Cell creation/manipulation                                │
│  - Metadata handling                                         │
└─────────────────────────┬────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────┐
│                  File System                                 │
│  - .ipynb files (JSON format)                                │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 Design Principles

1. **Separation of Concerns**: MCP layer separate from business logic
2. **Single Responsibility**: Each module has one clear purpose
3. **Fail-Safe**: Never corrupt notebooks, validate after modifications
4. **Type Safety**: Use type hints throughout for clarity and validation
5. **Testability**: Pure functions in operations.py, easy to unit test
6. **Error Transparency**: Clear error messages for LLM consumption

## 2. Module Design

### 2.1 operations.py - Core Business Logic

**Purpose**: Pure Python functions for notebook manipulation using nbformat

**Key Functions**:

```python
# File I/O
def read_notebook_file(filepath: str) -> nbformat.NotebookNode
def write_notebook_file(filepath: str, nb: nbformat.NotebookNode) -> None

# Read Operations
def get_notebook_summary(filepath: str) -> dict
def list_all_cells(filepath: str) -> list[dict]
def get_cell_content(filepath: str, cell_index: int) -> str
def search_cells(filepath: str, pattern: str, case_sensitive: bool) -> list[dict]

# Cell Modification
def replace_cell_content(filepath: str, cell_index: int, new_content: str) -> None
def insert_cell(filepath: str, cell_index: int, content: str, cell_type: str) -> None
def append_cell(filepath: str, content: str, cell_type: str) -> None
def delete_cell(filepath: str, cell_index: int) -> None
def str_replace_in_cell(filepath: str, cell_index: int, old_str: str, new_str: str) -> None

# Metadata Operations
def get_metadata(filepath: str, cell_index: int | None) -> dict
def update_metadata(filepath: str, metadata: dict, cell_index: int | None) -> None
def set_kernel_spec(filepath: str, kernel_name: str, display_name: str, language: str) -> None

# Batch Operations - Multi-Cell
def replace_cells_batch(filepath: str, replacements: list[dict]) -> None
def delete_cells_batch(filepath: str, cell_indices: list[int]) -> None
def insert_cells_batch(filepath: str, insertions: list[dict]) -> None
def search_replace_all(filepath: str, pattern: str, replacement: str, cell_type: str | None) -> int
def reorder_cells(filepath: str, new_order: list[int]) -> None
def filter_cells(filepath: str, cell_type: str | None, pattern: str | None) -> None

# Batch Operations - Multi-Notebook
def merge_notebooks(output_filepath: str, input_filepaths: list[str], add_separators: bool) -> None
def split_notebook(filepath: str, output_dir: str, split_by: str) -> list[str]
def apply_operation_to_notebooks(filepaths: list[str], operation: str, **params) -> dict[str, bool]
def search_across_notebooks(filepaths: list[str], pattern: str, return_context: bool) -> list[dict]
def sync_metadata_across_notebooks(filepaths: list[str], metadata: dict, merge: bool) -> None
def extract_cells_from_notebooks(output_filepath: str, input_filepaths: list[str], 
                                  pattern: str | None, cell_type: str | None) -> None
def clear_outputs(filepaths: str | list[str]) -> None

# Validation
def validate_notebook_file(filepath: str) -> tuple[bool, str | None]
def get_notebook_info(filepath: str) -> dict
def validate_multiple_notebooks(filepaths: list[str]) -> dict[str, tuple[bool, str | None]]
```

**Design Decisions**:
- All functions raise exceptions on error (caught by server layer)
- Read-modify-write pattern: read once, modify in memory, write once
- Validation after every write operation
- Use atomic writes (write to temp file, then rename)

### 2.2 server.py - MCP Tool Wrappers

**Purpose**: Expose operations as MCP tools with error handling

**Pattern**:

```python
from fastmcp import FastMCP
from . import operations

mcp = FastMCP(name="Jupyter Notebook Editor")

@mcp.tool
def read_notebook(filepath: str) -> dict:
    """Read Jupyter notebook and return structure summary.
    
    Args:
        filepath: Path to .ipynb file
        
    Returns:
        Dictionary with cell_count, cell_types, kernel_info, format_version
    """
    try:
        return operations.get_notebook_summary(filepath)
    except FileNotFoundError:
        return {"error": f"Notebook not found: {filepath}"}
    except Exception as e:
        return {"error": f"Failed to read notebook: {str(e)}"}

# ... 28 more tools following same pattern
```

**Error Handling Strategy**:
- Catch all exceptions from operations layer
- Return dict with "error" key for structured errors
- Include context (filepath, cell_index, etc.) in error messages
- Never expose stack traces to LLM

**Return Value Patterns**:
- Success: Return data as dict/list/str
- Error: Return dict with "error" key
- LLM checks for "error" key to determine success/failure

### 2.3 utils.py - Helper Functions

**Purpose**: Shared utilities and constants

```python
# Validation helpers
def validate_cell_type(cell_type: str) -> bool
def validate_cell_index(nb: NotebookNode, index: int) -> bool
def validate_filepath(filepath: str) -> bool

# Error formatting
def format_error(error_type: str, context: dict) -> dict

# Constants
VALID_CELL_TYPES = ["code", "markdown", "raw"]

COMMON_KERNELS = [
    {"name": "python3", "display_name": "Python 3", "language": "python"},
    {"name": "ir", "display_name": "R", "language": "R"},
    {"name": "julia-1.9", "display_name": "Julia 1.9", "language": "julia"},
    # ... more kernels
]

# Atomic write helper
def atomic_write(filepath: str, content: str) -> None:
    """Write to temp file, then rename to target (atomic operation)"""
    temp_path = f"{filepath}.tmp"
    with open(temp_path, 'w') as f:
        f.write(content)
    os.rename(temp_path, filepath)
```

## 3. Data Flow Diagrams

### 3.1 Single Cell Modification Flow

```
┌─────────┐
│   LLM   │
└────┬────┘
     │ insert_cell(filepath="nb.ipynb", cell_index=1, content="...", cell_type="code")
     ▼
┌─────────────────────────────────────────────────────────────┐
│ server.py: insert_cell()                                    │
│  1. Validate parameters                                     │
│  2. Call operations.insert_cell()                           │
│  3. Catch exceptions, format errors                         │
└────┬────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ operations.py: insert_cell()                                │
│  1. Read notebook: nb = read_notebook_file(filepath)        │
│  2. Validate cell_index in range                            │
│  3. Create new cell: cell = nbformat.v4.new_code_cell(...)  │
│  4. Insert: nb['cells'].insert(cell_index, cell)            │
│  5. Validate: nbformat.validate(nb)                         │
│  6. Write: write_notebook_file(filepath, nb)                │
└────┬────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ File System                                                 │
│  - Write to nb.ipynb.tmp                                    │
│  - Rename to nb.ipynb (atomic)                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Batch Operation Flow (Multi-Notebook)

```
┌─────────┐
│   LLM   │
└────┬────┘
     │ merge_notebooks(output="combined.ipynb", inputs=["a.ipynb", "b.ipynb"])
     ▼
┌─────────────────────────────────────────────────────────────┐
│ server.py: merge_notebooks()                                │
│  - Validate parameters                                      │
│  - Call operations.merge_notebooks()                        │
│  - Handle errors                                            │
└────┬────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ operations.py: merge_notebooks()                            │
│  1. Create new notebook: nb = nbformat.v4.new_notebook()    │
│  2. For each input file:                                    │
│     a. Read: input_nb = read_notebook_file(filepath)        │
│     b. Add separator cell (if enabled)                      │
│     c. Extend: nb['cells'].extend(input_nb['cells'])        │
│  3. Copy metadata from first notebook                       │
│  4. Validate: nbformat.validate(nb)                         │
│  5. Write: write_notebook_file(output_filepath, nb)         │
└─────────────────────────────────────────────────────────────┘
```

## 4. API Specifications

### 4.1 Tool Signatures

#### Read Operations

```python
@mcp.tool
def read_notebook(filepath: str) -> dict:
    """Returns: {cell_count, cell_types: {code: N, markdown: M}, kernel_info, format_version}"""

@mcp.tool
def list_cells(filepath: str) -> list[dict]:
    """Returns: [{index, type, preview}, ...]"""

@mcp.tool
def get_cell(filepath: str, cell_index: int) -> str:
    """Returns: cell source content"""

@mcp.tool
def search_cells(filepath: str, pattern: str, case_sensitive: bool = False) -> list[dict]:
    """Returns: [{cell_index, cell_type, match, context}, ...]"""
```

#### Cell Modification

```python
@mcp.tool
def replace_cell(filepath: str, cell_index: int, new_content: str) -> dict:
    """Returns: {success: true} or {error: "..."}"""

@mcp.tool
def insert_cell(filepath: str, cell_index: int, content: str, cell_type: str = "code") -> dict:
    """Returns: {success: true, new_cell_count} or {error: "..."}"""

@mcp.tool
def append_cell(filepath: str, content: str, cell_type: str = "code") -> dict:
    """Returns: {success: true, cell_index} or {error: "..."}"""

@mcp.tool
def delete_cell(filepath: str, cell_index: int) -> dict:
    """Returns: {success: true, new_cell_count} or {error: "..."}"""

@mcp.tool
def str_replace_in_cell(filepath: str, cell_index: int, old_str: str, new_str: str) -> dict:
    """Returns: {success: true} or {error: "..."}"""
```

#### Metadata Operations

```python
@mcp.tool
def get_metadata(filepath: str, cell_index: int | None = None) -> dict:
    """Returns: metadata dict"""

@mcp.tool
def update_metadata(filepath: str, metadata: dict, cell_index: int | None = None) -> dict:
    """Returns: {success: true} or {error: "..."}"""

@mcp.tool
def set_kernel(filepath: str, kernel_name: str, display_name: str, language: str = "python") -> dict:
    """Returns: {success: true} or {error: "..."}"""

@mcp.tool
def list_available_kernels() -> list[dict]:
    """Returns: [{name, display_name, language}, ...]"""
```

#### Batch Operations - Multi-Cell

```python
@mcp.tool
def replace_cells_batch(filepath: str, replacements: list[dict]) -> dict:
    """
    Args:
        replacements: [{"cell_index": 0, "content": "..."}, ...]
    Returns: {success: true, cells_modified: N} or {error: "..."}
    """

@mcp.tool
def delete_cells_batch(filepath: str, cell_indices: list[int]) -> dict:
    """Returns: {success: true, cells_deleted: N, new_cell_count} or {error: "..."}"""

@mcp.tool
def insert_cells_batch(filepath: str, insertions: list[dict]) -> dict:
    """
    Args:
        insertions: [{"cell_index": 0, "content": "...", "cell_type": "code"}, ...]
    Returns: {success: true, cells_inserted: N} or {error: "..."}
    """

@mcp.tool
def search_replace_all(filepath: str, pattern: str, replacement: str, 
                       cell_type: str | None = None) -> dict:
    """Returns: {success: true, replacements_made: N, cells_affected: M} or {error: "..."}"""

@mcp.tool
def reorder_cells(filepath: str, new_order: list[int]) -> dict:
    """Returns: {success: true} or {error: "..."}"""

@mcp.tool
def filter_cells(filepath: str, cell_type: str | None = None, 
                 pattern: str | None = None) -> dict:
    """Returns: {success: true, cells_kept: N, cells_deleted: M} or {error: "..."}"""
```

#### Batch Operations - Multi-Notebook

```python
@mcp.tool
def merge_notebooks(output_filepath: str, input_filepaths: list[str], 
                    add_separators: bool = True) -> dict:
    """Returns: {success: true, total_cells: N, notebooks_merged: M} or {error: "..."}"""

@mcp.tool
def split_notebook(filepath: str, output_dir: str, split_by: str = "markdown_headers") -> dict:
    """Returns: {success: true, files_created: [paths]} or {error: "..."}"""

@mcp.tool
def apply_to_notebooks(filepaths: list[str], operation: str, **operation_params) -> dict:
    """Returns: {success: true, results: {filepath: success/error}} or {error: "..."}"""

@mcp.tool
def search_notebooks(filepaths: list[str], pattern: str, return_context: bool = True) -> list[dict]:
    """Returns: [{filepath, cell_index, match, context}, ...]"""

@mcp.tool
def sync_metadata(filepaths: list[str], metadata: dict, merge: bool = False) -> dict:
    """Returns: {success: true, notebooks_updated: N} or {error: "..."}"""

@mcp.tool
def extract_cells(output_filepath: str, input_filepaths: list[str], 
                  pattern: str | None = None, cell_type: str | None = None) -> dict:
    """Returns: {success: true, cells_extracted: N, source_notebooks: M} or {error: "..."}"""

@mcp.tool
def clear_outputs(filepaths: str | list[str]) -> dict:
    """Returns: {success: true, notebooks_processed: N, outputs_cleared: M} or {error: "..."}"""
```

#### Validation

```python
@mcp.tool
def validate_notebook(filepath: str) -> dict:
    """Returns: {valid: true} or {valid: false, errors: [...]}"""

@mcp.tool
def get_notebook_info(filepath: str) -> dict:
    """Returns: {cell_count, cell_types, kernel, format_version, file_size}"""

@mcp.tool
def validate_notebooks_batch(filepaths: list[str]) -> dict:
    """Returns: {results: {filepath: {valid: bool, errors: [...]}}}"""
```

## 5. Error Handling Design

### 5.1 Error Categories

```python
class NotebookError(Exception):
    """Base exception for notebook operations"""

class NotebookNotFoundError(NotebookError):
    """Notebook file not found"""

class InvalidCellIndexError(NotebookError):
    """Cell index out of range"""

class InvalidCellTypeError(NotebookError):
    """Invalid cell type specified"""

class NotebookValidationError(NotebookError):
    """Notebook failed validation"""

class NotebookCorruptedError(NotebookError):
    """Notebook file is corrupted"""
```

### 5.2 Error Response Format

```python
{
    "error": "Error message for LLM",
    "error_type": "NotebookNotFoundError",
    "context": {
        "filepath": "notebook.ipynb",
        "cell_index": 5,
        # ... other relevant context
    }
}
```

### 5.3 Error Handling Pattern

```python
@mcp.tool
def tool_name(param: str) -> dict:
    """Tool description."""
    try:
        result = operations.operation_name(param)
        return {"success": True, "data": result}
    except NotebookNotFoundError as e:
        return {
            "error": f"Notebook not found: {param}",
            "error_type": "NotebookNotFoundError",
            "context": {"filepath": param}
        }
    except InvalidCellIndexError as e:
        return {
            "error": str(e),
            "error_type": "InvalidCellIndexError",
            "context": e.context
        }
    except Exception as e:
        return {
            "error": f"Unexpected error: {str(e)}",
            "error_type": type(e).__name__
        }
```

## 6. Data Structures

### 6.1 Notebook Structure (nbformat)

```python
NotebookNode = {
    'cells': [
        {
            'cell_type': 'code' | 'markdown' | 'raw',
            'source': str,  # Cell content
            'metadata': dict,
            'execution_count': int | None,  # code cells only
            'outputs': list[dict]  # code cells only
        }
    ],
    'metadata': {
        'kernelspec': {
            'name': str,
            'display_name': str,
            'language': str
        },
        'language_info': dict,
        # ... custom metadata
    },
    'nbformat': 4,
    'nbformat_minor': int
}
```

### 6.2 Internal Data Structures

```python
# Cell summary for list_cells
CellSummary = {
    'index': int,
    'type': str,
    'preview': str,  # First 100 chars
    'execution_count': int | None
}

# Search result
SearchResult = {
    'cell_index': int,
    'cell_type': str,
    'match': str,
    'context': str  # Surrounding lines
}

# Notebook info
NotebookInfo = {
    'cell_count': int,
    'cell_types': dict[str, int],  # {'code': 5, 'markdown': 3}
    'kernel': dict,
    'format_version': str,
    'file_size': int
}
```

## 7. Performance Considerations

### 7.1 Optimization Strategies

**File I/O**:
- Read notebook once per operation
- Batch operations read once, modify multiple times, write once
- Use atomic writes (temp file + rename)

**Memory Management**:
- Don't load multiple large notebooks simultaneously
- Stream operations where possible
- Clear notebook objects after write

**Validation**:
- Validate only after modifications, not on reads
- Cache validation results for batch operations

### 7.2 Performance Targets

| Operation Type | Target Time | Max Notebook Size |
|---------------|-------------|-------------------|
| Single read | < 100ms | 10MB |
| Single modify | < 1s | 10MB |
| Batch (10 notebooks) | < 5s | 10MB each |
| Validation | < 200ms | 10MB |

## 8. Testing Strategy

### 8.1 Test Structure

```
tests/
├── test_operations.py          # Unit tests for operations.py
├── test_server.py              # Integration tests for MCP tools
├── test_utils.py               # Tests for utility functions
├── test_batch_operations.py    # Tests for batch operations
├── fixtures/
│   ├── simple.ipynb           # Minimal valid notebook
│   ├── complex.ipynb          # Notebook with all cell types
│   ├── large.ipynb            # Large notebook for performance tests
│   ├── invalid.ipynb          # Corrupted notebook
│   └── empty.ipynb            # Empty notebook
└── conftest.py                 # Pytest fixtures
```

### 8.2 Test Fixtures (conftest.py)

```python
import pytest
import nbformat
import tempfile

@pytest.fixture
def simple_notebook():
    """Create a simple test notebook"""
    nb = nbformat.v4.new_notebook()
    nb['cells'] = [
        nbformat.v4.new_code_cell("print('hello')"),
        nbformat.v4.new_markdown_cell("# Title"),
        nbformat.v4.new_code_cell("x = 1")
    ]
    return nb

@pytest.fixture
def temp_notebook_file(simple_notebook):
    """Create a temporary notebook file"""
    with tempfile.NamedTemporaryFile(suffix='.ipynb', delete=False) as f:
        nbformat.write(simple_notebook, f)
        return f.name

@pytest.fixture
def mcp_client():
    """Create FastMCP test client"""
    from jupyter_editor.server import mcp
    return mcp.test_client()
```

### 8.3 Test Examples

```python
# Unit test
def test_insert_cell(temp_notebook_file):
    """Test US-006: Insert Cell"""
    operations.insert_cell(temp_notebook_file, 1, "print('inserted')", "code")
    
    nb = operations.read_notebook_file(temp_notebook_file)
    assert len(nb['cells']) == 4
    assert nb['cells'][1]['source'] == "print('inserted')"
    
    # Verify format preserved
    nbformat.validate(nb)

# Integration test
async def test_insert_cell_tool(mcp_client, temp_notebook_file):
    """Test insert_cell MCP tool"""
    result = await mcp_client.call_tool(
        "insert_cell",
        filepath=temp_notebook_file,
        cell_index=1,
        content="print('inserted')",
        cell_type="code"
    )
    
    assert "error" not in result
    assert result["success"] is True
    assert result["new_cell_count"] == 4
```

## 9. Deployment Configuration

### 9.1 pyproject.toml

```toml
[project]
name = "jupyter-editor-mcp"
version = "0.1.0"
description = "MCP server for editing Jupyter notebooks"
authors = [{name = "Your Name", email = "your.email@example.com"}]
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}

dependencies = [
    "fastmcp>=0.1.0",
    "nbformat>=5.10.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
]

[project.scripts]
jupyter-editor-mcp = "jupyter_editor.server:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "--cov=jupyter_editor --cov-report=html --cov-report=term"

[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
```

### 9.2 MCP Client Configuration

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):
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

**VS Code** (`.vscode/settings.json`):
```json
{
  "mcp.servers": {
    "jupyter-editor": {
      "command": "uv",
      "args": ["run", "python", "-m", "jupyter_editor.server"],
      "cwd": "${workspaceFolder}"
    }
  }
}
```

## 10. Implementation Checklist

### Phase 1: Foundation
- [ ] Create project structure
- [ ] Set up pyproject.toml
- [ ] Create test fixtures
- [ ] Implement file I/O functions
- [ ] Implement read operations (4 tools)
- [ ] Write unit tests for read operations
- [ ] Create basic MCP server wrapper

### Phase 2: Single-Cell Operations
- [ ] Implement cell modification operations (5 tools)
- [ ] Implement metadata operations (4 tools)
- [ ] Write unit tests for all operations
- [ ] Add error handling to MCP tools
- [ ] Test with FastMCP test client

### Phase 3: Batch Operations
- [ ] Implement multi-cell batch operations (6 tools)
- [ ] Implement multi-notebook batch operations (7 tools)
- [ ] Write integration tests
- [ ] Performance testing
- [ ] Test with Claude Desktop

### Phase 4: Validation & Polish
- [ ] Implement validation operations (3 tools)
- [ ] Complete documentation (README, docstrings)
- [ ] Code review and refactoring
- [ ] Final testing
- [ ] Prepare for release

## 11. Future Enhancements

### 11.1 Advanced Features
- **Output manipulation**: Edit cell outputs, not just clear them
- **Cell execution**: Execute cells via Jupyter kernel
- **Diff/merge**: Compare and merge notebooks
- **Templates**: Generate notebooks from templates
- **Git integration**: Better version control support

### 11.2 Performance Improvements
- **Lazy loading**: Load only required cells
- **Caching**: Cache notebook structure for repeated operations
- **Parallel processing**: Process multiple notebooks in parallel

### 11.3 Additional Tools
- **Export**: Convert notebooks to HTML, PDF, Markdown
- **Import**: Convert other formats to notebooks
- **Linting**: Check code quality in cells
- **Statistics**: Analyze notebook complexity, size, etc.
