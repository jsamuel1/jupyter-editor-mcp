# Jupyter Notebook Editor MCP Server - Requirements Document

## 1. Project Overview

### 1.1 Purpose
Build a Model Context Protocol (MCP) server that enables AI assistants to programmatically edit Jupyter notebooks while preserving their format and structure.

### 1.2 Goals
- Provide 29 specialized tools for notebook manipulation
- Ensure format preservation and validation
- Enable file-based operations without Jupyter server dependency
- Support both single-cell and batch operations
- Enable multi-notebook workflows

### 1.3 Target Users
- AI assistants (Claude, ChatGPT, etc.) via MCP protocol
- Developers using MCP-enabled IDEs
- Data scientists automating notebook workflows

## 2. User Stories

### 2.1 Read Operations

**US-001: Read Notebook Structure**
- As an AI assistant, I need to read a notebook's structure so I can understand its contents
- Acceptance Criteria:
  - Returns cell count, types, and metadata
  - Includes kernel information
  - Shows notebook format version
  - Test: Read sample notebook and verify all fields present

**US-002: List All Cells**
- As an AI assistant, I need to list all cells with previews so I can navigate the notebook
- Acceptance Criteria:
  - Returns cell index, type, and content preview (first 100 chars)
  - Preserves cell order
  - Test: List cells and verify indices match actual positions

**US-003: Get Specific Cell**
- As an AI assistant, I need to retrieve a specific cell's content so I can analyze or modify it
- Acceptance Criteria:
  - Returns full cell content by index
  - Handles negative indices (from end)
  - Returns error for invalid index
  - Test: Get cell at various indices including edge cases

**US-004: Search Cell Content**
- As an AI assistant, I need to search for patterns in cells so I can find relevant code
- Acceptance Criteria:
  - Supports case-sensitive and case-insensitive search
  - Returns cell indices and matching content
  - Supports regex patterns
  - Test: Search for various patterns and verify matches

### 2.2 Cell Modification

**US-005: Replace Cell Content**
- As an AI assistant, I need to replace entire cell content so I can update code or text
- Acceptance Criteria:
  - Replaces cell at specified index
  - Preserves cell type and metadata
  - Validates notebook after replacement
  - Test: Replace cell and verify content changed, format preserved

**US-006: Insert Cell**
- As an AI assistant, I need to insert cells at specific positions so I can add new content
- Acceptance Criteria:
  - Inserts at specified index (0 = beginning)
  - Supports code, markdown, and raw cell types
  - Shifts subsequent cells down
  - Test: Insert at various positions and verify order

**US-007: Append Cell**
- As an AI assistant, I need to append cells to the end so I can add new content quickly
- Acceptance Criteria:
  - Adds cell to end of notebook
  - Supports all cell types
  - Test: Append multiple cells and verify order

**US-008: Delete Cell**
- As an AI assistant, I need to delete cells so I can remove unwanted content
- Acceptance Criteria:
  - Removes cell at specified index
  - Adjusts subsequent indices
  - Returns error for invalid index
  - Test: Delete cells and verify removal, index adjustment

**US-009: String Replace in Cell**
- As an AI assistant, I need to replace substrings within cells so I can make targeted edits
- Acceptance Criteria:
  - Replaces exact string match within cell
  - Fails if old_str not found or not unique
  - Similar to fs_write str_replace behavior
  - Test: Replace strings and verify exact matching behavior

### 2.3 Metadata Operations

**US-010: Get Metadata**
- As an AI assistant, I need to read metadata so I can understand notebook configuration
- Acceptance Criteria:
  - Returns notebook-level metadata when cell_index is None
  - Returns cell-level metadata when cell_index provided
  - Test: Get both notebook and cell metadata

**US-011: Update Metadata**
- As an AI assistant, I need to update metadata so I can configure notebooks
- Acceptance Criteria:
  - Updates notebook or cell metadata
  - Merges with existing metadata
  - Validates metadata structure
  - Test: Update metadata and verify changes

**US-012: Set Kernel**
- As an AI assistant, I need to set kernel specifications so notebooks use correct environments
- Acceptance Criteria:
  - Sets kernel name, display name, and language
  - Updates kernelspec metadata
  - Test: Set kernel and verify metadata updated

**US-013: List Available Kernels**
- As an AI assistant, I need to see common kernel options so I can choose appropriate ones
- Acceptance Criteria:
  - Returns predefined list of common kernels (Python 3.x, R, Julia, etc.)
  - Includes name, display_name, and language for each
  - Test: Verify list contains expected kernels

### 2.4 Batch Operations - Multi-Cell

**US-014: Replace Multiple Cells**
- As an AI assistant, I need to replace multiple cells at once so I can make bulk updates efficiently
- Acceptance Criteria:
  - Accepts list of {cell_index, content} dicts
  - Applies all replacements in single operation
  - Validates notebook after all changes
  - Test: Replace multiple cells and verify all changes applied

**US-015: Delete Multiple Cells**
- As an AI assistant, I need to delete multiple cells at once so I can clean up notebooks efficiently
- Acceptance Criteria:
  - Accepts list of cell indices
  - Handles indices in any order
  - Adjusts for index shifts during deletion
  - Test: Delete multiple cells and verify correct removal

**US-016: Insert Multiple Cells**
- As an AI assistant, I need to insert multiple cells at once so I can add content efficiently
- Acceptance Criteria:
  - Accepts list of {cell_index, content, cell_type} dicts
  - Inserts in order to maintain correct positions
  - Test: Insert multiple cells and verify positions

**US-017: Search and Replace All**
- As an AI assistant, I need to search/replace across all cells so I can refactor code
- Acceptance Criteria:
  - Replaces all occurrences of pattern
  - Optionally filters by cell_type
  - Returns count of replacements made
  - Test: Replace across cells and verify all occurrences changed

**US-018: Reorder Cells**
- As an AI assistant, I need to reorder cells so I can reorganize notebook structure
- Acceptance Criteria:
  - Accepts new order as list of indices
  - Validates all indices present exactly once
  - Test: Reorder cells and verify new order

**US-019: Filter Cells**
- As an AI assistant, I need to filter cells by criteria so I can keep only relevant content
- Acceptance Criteria:
  - Filters by cell_type and/or pattern
  - Deletes non-matching cells
  - Test: Filter cells and verify only matches remain

### 2.5 Batch Operations - Multi-Notebook

**US-020: Merge Notebooks**
- As an AI assistant, I need to merge notebooks so I can combine related work
- Acceptance Criteria:
  - Combines cells from multiple notebooks in order
  - Optionally adds separator cells between notebooks
  - Uses first notebook's metadata as base
  - Test: Merge notebooks and verify cell order, metadata

**US-021: Split Notebook**
- As an AI assistant, I need to split notebooks so I can organize content into separate files
- Acceptance Criteria:
  - Splits by markdown headers, cell count, or explicit indices
  - Creates valid notebooks for each split
  - Test: Split notebook and verify all parts valid

**US-022: Apply to Multiple Notebooks**
- As an AI assistant, I need to apply operations to multiple notebooks so I can make project-wide changes
- Acceptance Criteria:
  - Accepts list of filepaths and operation name
  - Applies same operation with same parameters to all
  - Returns success/failure status for each
  - Test: Apply operation to multiple notebooks and verify all updated

**US-023: Search Multiple Notebooks**
- As an AI assistant, I need to search across notebooks so I can find content project-wide
- Acceptance Criteria:
  - Searches all specified notebooks
  - Returns filepath, cell_index, and matching content
  - Optionally includes context lines
  - Test: Search multiple notebooks and verify all matches found

**US-024: Sync Metadata**
- As an AI assistant, I need to synchronize metadata so notebooks have consistent configuration
- Acceptance Criteria:
  - Updates metadata across multiple notebooks
  - Optionally merges with existing metadata
  - Test: Sync metadata and verify all notebooks updated

**US-025: Extract Cells**
- As an AI assistant, I need to extract cells from multiple notebooks so I can create compilations
- Acceptance Criteria:
  - Extracts cells matching pattern and/or cell_type
  - Creates new notebook with extracted cells
  - Preserves cell metadata
  - Test: Extract cells and verify new notebook contains only matches

**US-026: Clear Outputs**
- As an AI assistant, I need to clear outputs from notebooks so I can reduce file size and prepare for version control
- Acceptance Criteria:
  - Clears all outputs from code cells
  - Resets execution_count to None
  - Accepts single filepath or list of filepaths
  - Preserves cell source and metadata
  - Test: Clear outputs and verify all outputs removed, execution counts reset

### 2.6 Validation

**US-027: Validate Notebook**
- As an AI assistant, I need to validate notebooks so I can ensure format correctness
- Acceptance Criteria:
  - Uses nbformat.validate()
  - Returns validation status and errors
  - Test: Validate valid and invalid notebooks

**US-028: Get Notebook Info**
- As an AI assistant, I need to get notebook summary so I can understand its structure quickly
- Acceptance Criteria:
  - Returns cell count by type
  - Returns kernel information
  - Returns format version
  - Test: Get info and verify all fields accurate

**US-029: Validate Multiple Notebooks**
- As an AI assistant, I need to validate multiple notebooks so I can check project health
- Acceptance Criteria:
  - Validates all specified notebooks
  - Returns status for each (valid/invalid with errors)
  - Test: Validate mix of valid and invalid notebooks

## 3. Functional Requirements

### 3.1 Core Operations Module

**FR-001: Notebook File I/O**
- Must read notebooks using `nbformat.read(f, as_version=4)`
- Must write notebooks using `nbformat.write(nb, f)`
- Must handle file not found errors gracefully
- Must handle permission errors gracefully

**FR-002: Cell Access**
- Must access cells via `nb['cells']` list
- Must support negative indexing
- Must validate cell indices before access
- Must preserve cell order

**FR-003: Cell Creation**
- Must create code cells using `nbformat.v4.new_code_cell()`
- Must create markdown cells using `nbformat.v4.new_markdown_cell()`
- Must create raw cells using `nbformat.v4.new_raw_cell()`

**FR-004: Metadata Management**
- Must access notebook metadata via `nb.metadata`
- Must access cell metadata via `cell.metadata`
- Must preserve existing metadata when updating

**FR-005: Format Validation**
- Must validate notebooks using `nbformat.validate()`
- Must validate after all modification operations
- Must return clear error messages for validation failures

### 3.2 MCP Server

**FR-006: Server Initialization**
- Must create FastMCP instance with name "Jupyter Notebook Editor"
- Must expose all 28 tools via `@mcp.tool` decorator
- Must run via `mcp.run()` in `__main__` block

**FR-007: Tool Definitions**
- Must use type hints for all parameters
- Must provide clear docstrings for LLM guidance
- Must return structured data (dict/list) or error strings
- Must handle all exceptions and return user-friendly errors

**FR-008: Parameter Validation**
- Must validate filepath exists before operations
- Must validate cell_index within range
- Must validate cell_type in ['code', 'markdown', 'raw']
- Must validate list parameters are non-empty

### 3.3 Error Handling

**FR-009: Error Types**
- Must handle FileNotFoundError with clear message
- Must handle IndexError with cell index information
- Must handle ValidationError with validation details
- Must handle PermissionError with file path

**FR-010: Error Format**
- Must return errors as strings starting with "Error: "
- Must include relevant context (filepath, index, etc.)
- Must not expose internal stack traces to LLM

## 4. Non-Functional Requirements

### 4.1 Performance

**NFR-001: Response Time**
- Single-cell operations must complete in < 1 second
- Batch operations must complete in < 5 seconds for 10 notebooks
- File I/O must be optimized (read once, write once per operation)

**NFR-002: Memory Usage**
- Must handle notebooks up to 10MB
- Must not load multiple notebooks simultaneously unless required
- Must clean up file handles properly

### 4.2 Reliability

**NFR-003: Data Integrity**
- Must never corrupt notebook files
- Must validate notebooks after modifications
- Must use atomic writes (write to temp, then rename)

**NFR-004: Error Recovery**
- Must not leave notebooks in invalid state
- Must provide clear error messages for recovery
- Must log errors for debugging

### 4.3 Security

**NFR-005: File Access**
- Must only access files explicitly provided in parameters
- Must not traverse directories without explicit permission
- Must validate file paths to prevent path traversal attacks

**NFR-006: Code Execution**
- Must never execute notebook code
- Must only manipulate notebook structure and content

### 4.4 Usability

**NFR-007: LLM-Friendly**
- Tool names must be clear and descriptive
- Docstrings must explain purpose and parameters
- Return values must be structured and parseable

**NFR-008: Documentation**
- Must provide README with installation and usage
- Must provide examples for common workflows
- Must document all 28 tools with parameters and return types

## 5. Technical Architecture

### 5.1 Project Structure

```
jupyter-editor/
├── src/
│   └── jupyter_editor/
│       ├── __init__.py
│       ├── server.py          # FastMCP server with @mcp.tool decorators
│       ├── operations.py      # Core notebook operations (nbformat logic)
│       └── utils.py           # Helper functions (validation, error handling)
├── tests/
│   ├── test_operations.py     # Unit tests for operations
│   ├── test_server.py         # Integration tests for MCP tools
│   └── fixtures/
│       ├── sample.ipynb       # Valid test notebook
│       └── invalid.ipynb      # Invalid test notebook
├── pyproject.toml
├── README.md
├── REQUIREMENTS.md
└── RESEARCH.md
```

### 5.2 Component Design

**operations.py**
- Pure Python functions for notebook manipulation
- No MCP dependencies
- Testable independently
- Returns Python objects or raises exceptions

**server.py**
- Thin wrapper around operations
- Converts operations to MCP tools
- Handles error conversion to strings
- Manages FastMCP lifecycle

**utils.py**
- Validation helpers
- Error formatting
- Common constants (kernel specs, etc.)

### 5.3 Data Flow

1. LLM calls MCP tool via STDIO transport
2. FastMCP deserializes request and calls tool function
3. Tool function calls operation in operations.py
4. Operation reads notebook, modifies, validates, writes
5. Operation returns result or raises exception
6. Tool function catches exception, formats error
7. FastMCP serializes response and returns to LLM

## 6. Dependencies & Environment

### 6.1 Python Version
- Requires Python >= 3.10 (for type hints with `|` syntax)

### 6.2 Core Dependencies
- `fastmcp >= 0.1.0` - MCP server framework
- `nbformat >= 5.10.0` - Jupyter notebook format library

### 6.3 Development Dependencies
- `pytest >= 7.0.0` - Testing framework
- `pytest-asyncio >= 0.21.0` - Async test support

### 6.4 Installation

```bash
# Create virtual environment
uv venv

# Install in development mode
uv pip install -e ".[dev]"

# Run tests
pytest
```

### 6.5 MCP Client Configuration

**Claude Desktop** (`claude_desktop_config.json`):
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

## 7. Testing Strategy (TDD Approach)

### 7.1 Test Categories

**Unit Tests** (test_operations.py)
- Test each operation function independently
- Use temporary files for I/O operations
- Test edge cases (empty notebooks, invalid indices, etc.)
- Test error conditions

**Integration Tests** (test_server.py)
- Test MCP tools end-to-end
- Use FastMCP test client
- Verify tool responses match expected format
- Test error handling through MCP layer

**Fixture Tests**
- Create sample notebooks with various structures
- Test operations preserve format
- Test validation catches corruption

### 7.2 Test-Driven Development Process

For each tool:
1. Write test defining expected behavior
2. Run test (should fail)
3. Implement minimal code to pass test
4. Refactor while keeping tests green
5. Add edge case tests
6. Repeat until all acceptance criteria met

### 7.3 Sample Test Structure

```python
def test_insert_cell():
    """Test US-006: Insert Cell"""
    # Arrange
    nb = create_test_notebook(cells=["print('first')"])
    filepath = save_temp_notebook(nb)
    
    # Act
    result = operations.insert_cell(
        filepath, 
        cell_index=1, 
        content="print('second')", 
        cell_type="code"
    )
    
    # Assert
    nb = operations.read_notebook_file(filepath)
    assert len(nb['cells']) == 2
    assert nb['cells'][1]['source'] == "print('second')"
    assert nb['cells'][1]['cell_type'] == 'code'
    
    # Validate format preserved
    nbformat.validate(nb)
```

### 7.4 Coverage Requirements
- Minimum 90% code coverage
- 100% coverage for critical paths (file I/O, validation)
- All error paths must be tested

## 8. Success Criteria

### 8.1 Functional Success
- ✅ All 29 tools implemented and working
- ✅ All user stories have passing tests
- ✅ All acceptance criteria met
- ✅ No notebook corruption in any test case

### 8.2 Quality Success
- ✅ 90%+ test coverage
- ✅ All tests passing
- ✅ No critical bugs
- ✅ Documentation complete

### 8.3 Integration Success
- ✅ Works with Claude Desktop
- ✅ Works with other MCP clients
- ✅ LLM can successfully use all tools
- ✅ Error messages are clear and actionable

### 8.4 Performance Success
- ✅ Single operations < 1s
- ✅ Batch operations < 5s for 10 notebooks
- ✅ No memory leaks
- ✅ Handles 10MB notebooks

## 9. Implementation Phases

### Phase 1: Foundation (Week 1)
- Set up project structure
- Implement core operations module
- Implement read operations (US-001 to US-004)
- Write unit tests for read operations

### Phase 2: Single-Cell Modifications (Week 2)
- Implement cell modification operations (US-005 to US-009)
- Implement metadata operations (US-010 to US-013)
- Write unit tests for all operations
- Implement MCP server wrapper

### Phase 3: Batch Operations (Week 3)
- Implement multi-cell batch operations (US-014 to US-019)
- Implement multi-notebook batch operations (US-020 to US-025)
- Write integration tests
- Test with Claude Desktop

### Phase 4: Validation & Polish (Week 4)
- Implement validation operations (US-026 to US-028)
- Complete documentation
- Performance optimization
- Final testing and bug fixes

## 10. Risks & Mitigations

### Risk 1: Notebook Corruption
- **Impact**: High - Data loss
- **Probability**: Medium
- **Mitigation**: Comprehensive validation, atomic writes, extensive testing

### Risk 2: Performance Issues with Large Notebooks
- **Impact**: Medium - Poor UX
- **Probability**: Medium
- **Mitigation**: Optimize I/O, lazy loading, performance testing

### Risk 3: MCP Protocol Changes
- **Impact**: High - Breaking changes
- **Probability**: Low
- **Mitigation**: Pin FastMCP version, monitor updates

### Risk 4: Complex Batch Operations
- **Impact**: Medium - Implementation complexity
- **Probability**: High
- **Mitigation**: Incremental implementation, thorough testing

## 11. Future Enhancements

- Support for notebook outputs manipulation
- Cell execution via Jupyter kernel
- Diff/merge capabilities for notebooks
- Template-based notebook generation
- Integration with version control systems
- Support for JupyterLab extensions metadata
