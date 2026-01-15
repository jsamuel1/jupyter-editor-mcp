---
name: "jupyter-editor"
displayName: "Jupyter Editor MCP"
description: "Edit Jupyter Notebooks (.ipynb) programmatically. Read, create, modify cells, manage metadata, and perform batch operations on notebooks without needing a running Jupyter server."
keywords: ["jupyter", "notebook", "ipynb", "python", "data-science", "cells", "markdown"]
author: "Kiro User"
---

# Jupyter Editor MCP

## Overview

The Jupyter Editor MCP server enables AI coding assistants to work directly with Jupyter Notebook files (.ipynb) without requiring a running Jupyter server. This is essential for programmatic notebook manipulation, batch operations, and integrating notebook editing into automated workflows.

## When to Use This Power

Use the Jupyter Editor MCP when you need to:

- Read, analyze, or search notebook content
- Add, modify, or delete cells in notebooks
- Perform batch operations across multiple cells or notebooks
- Manage notebook metadata and kernel specifications
- Merge, split, or reorganize notebooks
- Validate notebook structure and format

Do NOT use this power for:

- Executing notebook cells (use a Jupyter kernel instead)
- Real-time collaborative editing
- Working with non-.ipynb files

## Onboarding

### Prerequisites

- Python 3.10+ with `uv` or `uvx` available
- Kiro with MCP Support configured

### Installation

The Jupyter Editor MCP server is automatically configured when you install this power. No additional installation steps required.

## Tool Selection Guide

### When to Use Each Tool

| Task | Tool | Example |
|------|------|---------|
| Understand notebook structure | `ipynb_read_notebook` | "What's in this notebook?" |
| See all cells at a glance | `ipynb_list_cells` | "Show me all cells" |
| Read specific cell content | `ipynb_get_cell` | "What's in cell 3?" |
| Find content in notebook | `ipynb_search_cells` | "Find cells with pandas" |
| Replace entire cell | `ipynb_replace_cell` | "Rewrite cell 5" |
| Add cell at position | `ipynb_insert_cell` | "Add imports at the top" |
| Add cell at end | `ipynb_append_cell` | "Add a conclusion section" |
| Remove a cell | `ipynb_delete_cell` | "Delete cell 2" |
| Fix text in cell | `ipynb_str_replace_in_cell` | "Change function name in cell 4" |
| Find/replace everywhere | `ipynb_search_replace_all` | "Rename variable across all cells" |
| Modify multiple cells | `ipynb_replace_cells_batch` | "Update cells 1, 3, and 5" |
| Delete multiple cells | `ipynb_delete_cells_batch` | "Remove cells 2, 4, 6" |
| Combine notebooks | `ipynb_merge_notebooks` | "Merge analysis and viz notebooks" |
| Search multiple notebooks | `ipynb_search_notebooks` | "Find 'deprecated' in all notebooks" |
| Clean for git commit | `ipynb_clear_outputs` | "Clear all outputs" |
| Check notebook validity | `ipynb_validate_notebook` | "Is this notebook valid?" |

### Tool Categories

**Read Operations** - Use first to understand notebook structure:

- `ipynb_read_notebook` - Get notebook summary (cell count, types, kernel)
- `ipynb_list_cells` - List all cells with indices and previews
- `ipynb_get_cell` - Get specific cell content by index
- `ipynb_search_cells` - Search for patterns in cell content

**Single Cell Modification** - For targeted edits:

- `ipynb_replace_cell` - Replace entire cell content
- `ipynb_insert_cell` - Insert new cell at position
- `ipynb_append_cell` - Add cell to end of notebook
- `ipynb_delete_cell` - Remove cell by index
- `ipynb_str_replace_in_cell` - Replace substring within cell (like str_replace)

**Batch Cell Operations** - For multiple changes in one operation:

- `ipynb_replace_cells_batch` - Replace multiple cells at once
- `ipynb_delete_cells_batch` - Delete multiple cells
- `ipynb_insert_cells_batch` - Insert multiple cells
- `ipynb_search_replace_all` - Find/replace across all cells
- `ipynb_reorder_cells` - Reorder cells by new index mapping
- `ipynb_filter_cells` - Keep only cells matching criteria

**Multi-Notebook Operations** - For project-wide changes:

- `ipynb_merge_notebooks` - Combine multiple notebooks into one
- `ipynb_split_notebook` - Split notebook by headers or cell count
- `ipynb_apply_to_notebooks` - Apply operation to multiple notebooks
- `ipynb_search_notebooks` - Search across multiple notebooks
- `ipynb_sync_metadata` - Sync metadata across notebooks
- `ipynb_extract_cells` - Extract matching cells to new notebook
- `ipynb_clear_outputs` - Clear all cell outputs

**Metadata Operations** - For notebook configuration:

- `ipynb_get_metadata` - Get notebook or cell metadata
- `ipynb_update_metadata` - Update notebook or cell metadata
- `ipynb_set_kernel` - Set kernel specification
- `ipynb_list_available_kernels` - List common kernel configs

**Validation** - For quality assurance:

- `ipynb_validate_notebook` - Validate notebook structure
- `ipynb_get_notebook_info` - Get comprehensive notebook info
- `ipynb_validate_notebooks_batch` - Validate multiple notebooks

## Common Workflows

### Workflow 1: Read and Analyze Notebook

**Goal**: Understand the structure and content of an existing notebook.

**Steps**:

1. Use `ipynb_read_notebook` to get notebook summary
2. Use `ipynb_list_cells` to see all cells with previews
3. Use `ipynb_get_cell` to read specific cell content

**Example Prompt**: "Show me the structure of my analysis.ipynb notebook"

### Workflow 2: Add New Cells

**Goal**: Add code or markdown cells to an existing notebook.

**Steps**:

1. Use `ipynb_list_cells` to understand current structure
2. Use `ipynb_insert_cell` or `ipynb_append_cell` to add new cells
3. Specify cell type as 'code', 'markdown', or 'raw'

**Example Prompt**: "Add a markdown cell with a summary section at the end of my notebook"

### Workflow 3: Modify Existing Cells

**Goal**: Update or fix content in existing notebook cells.

**Steps**:

1. Use `ipynb_get_cell` to read current content
2. Use `ipynb_replace_cell` to replace entire cell content
3. Or use `ipynb_str_replace_in_cell` for targeted string replacements

**Example Prompt**: "Fix the import statement in cell 3 of my notebook"

### Workflow 4: Refactoring Across Cells

**Goal**: Rename variables, functions, or update patterns across the notebook.

**Steps**:

1. Use `ipynb_search_cells` to find all occurrences
2. Use `ipynb_search_replace_all` for find/replace across all cells
3. Optionally filter by cell_type to only affect code cells

**Example Prompt**: "Replace all occurrences of 'old_function' with 'new_function' in my notebook"

### Workflow 5: Batch Cell Operations

**Goal**: Modify multiple cells efficiently.

**Steps**:

1. Use `ipynb_replace_cells_batch` for multiple cell updates
2. Use `ipynb_delete_cells_batch` to remove multiple cells
3. Use `ipynb_insert_cells_batch` to add multiple cells at once

**Example Prompt**: "Delete cells 2, 5, and 7 from my notebook"

### Workflow 6: Notebook Management

**Goal**: Organize, merge, or split notebooks.

**Steps**:

1. Use `ipynb_merge_notebooks` to combine multiple notebooks
2. Use `ipynb_split_notebook` to split by headers or cell count
3. Use `ipynb_reorder_cells` to reorganize cell order

**Example Prompt**: "Merge my data_prep.ipynb and analysis.ipynb into a single notebook"

### Workflow 7: Project-Wide Updates

**Goal**: Apply changes across all notebooks in a project.

**Steps**:

1. Use `ipynb_search_notebooks` to find notebooks with specific content
2. Use `ipynb_apply_to_notebooks` to apply operations to multiple notebooks
3. Use `ipynb_sync_metadata` to standardize metadata

**Example Prompt**: "Set all notebooks in this directory to use the Python 3.11 kernel"

### Workflow 8: Prepare for Version Control

**Goal**: Clean notebooks before committing to git.

**Steps**:

1. Use `ipynb_clear_outputs` to remove all cell outputs
2. Use `ipynb_validate_notebook` to ensure valid structure

**Example Prompt**: "Clear all outputs from my notebook before committing to git"

## Best Practices

- **Use absolute paths**: Prevents path resolution issues
- **Read before modifying**: Use `ipynb_list_cells` or `ipynb_get_cell` to understand structure first
- **Validate after batch operations**: Use `ipynb_validate_notebook` to check structure
- **Backup important notebooks**: Before major modifications
- **Use batch operations**: More efficient than individual calls for multiple changes
- **Clear outputs when sharing**: Use `ipynb_clear_outputs` for cleaner notebooks
- **Cell indices are 0-based**: First cell is index 0
- **Negative indices work**: -1 refers to the last cell

## Troubleshooting

### File Not Found

**Problem**: "File not found" or path errors
**Solution**:

1. Use absolute paths when possible
2. Verify the file exists and has .ipynb extension
3. Check file permissions

### Invalid Notebook Format

**Problem**: "Invalid notebook" or JSON parsing errors
**Solution**:

1. Use `ipynb_validate_notebook` to check structure
2. Ensure the file is a valid Jupyter notebook format
3. Check for corrupted or incomplete JSON

### Cell Index Out of Range

**Problem**: "Index out of range" errors
**Solution**:

1. Use `ipynb_list_cells` to see valid indices
2. Remember indices are 0-based
3. Negative indices count from the end

### Content Escaping Issues

**Problem**: Special characters causing issues
**Solution**:

1. Provide content as raw strings
2. No additional escaping needed for tool parameters
3. The tool handles JSON escaping internally

## Example Use Cases

### Data Science Workflow

```text
"Add a new analysis section to my notebook with imports for pandas and matplotlib"
```

### Documentation

```text
"Add markdown documentation cells explaining each code section"
```

### Cleanup

```text
"Clear all outputs and remove empty cells from my notebook"
```

### Standardization

```text
"Set the kernel to python3 and update metadata for all notebooks in this folder"
```

### Refactoring

```text
"Rename the variable 'df' to 'data_frame' in all code cells"
```

### Content Extraction

```text
"Extract all markdown cells from these notebooks into a documentation notebook"
```

## Additional Resources

- [Jupyter Notebook Format](https://nbformat.readthedocs.io/)
- [jupyter-editor-mcp PyPI](https://pypi.org/project/jupyter-editor-mcp/)
- [GitHub Repository](https://github.com/jsamuel1/jupyter-editor-mcp)

---

**Package**: `jupyter-editor-mcp`
**MCP Server**: `jupyter-editor`
