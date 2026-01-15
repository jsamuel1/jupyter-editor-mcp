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

## Onboarding

### Prerequisites

- **Python**: Python 3.8+ with `uv` or `uvx` available
- **Kiro with MCP Support**: Ensure Kiro is configured with MCP capabilities

### Installation

The Jupyter Editor MCP server is automatically configured when you install this power. No additional installation steps required.

### Configuration

**No additional configuration required** - works after the power is installed in Kiro.

## Common Workflows

### Workflow 1: Read and Analyze Notebook Structure

**Goal**: Understand the structure and content of an existing notebook.

**Steps**:
1. Use `ipynb_read_notebook` to get notebook summary
2. Use `ipynb_list_cells` to see all cells with previews
3. Use `ipynb_get_cell` to read specific cell content

**Example Prompt**:
```
"Show me the structure of my analysis.ipynb notebook"
```

### Workflow 2: Add New Cells to a Notebook

**Goal**: Add code or markdown cells to an existing notebook.

**Steps**:
1. Use `ipynb_list_cells` to understand current structure
2. Use `ipynb_insert_cell` or `ipynb_append_cell` to add new cells
3. Specify cell type as 'code', 'markdown', or 'raw'

**Example Prompt**:
```
"Add a markdown cell with a summary section at the end of my notebook"
```

### Workflow 3: Modify Existing Cells

**Goal**: Update or fix content in existing notebook cells.

**Steps**:
1. Use `ipynb_get_cell` to read current content
2. Use `ipynb_replace_cell` to replace entire cell content
3. Or use `ipynb_str_replace_in_cell` for targeted string replacements

**Example Prompt**:
```
"Fix the import statement in cell 3 of my notebook"
```

### Workflow 4: Batch Operations

**Goal**: Perform operations across multiple cells or notebooks.

**Steps**:
1. Use `ipynb_replace_cells_batch` for multiple cell updates
2. Use `ipynb_search_replace_all` for find/replace across all cells
3. Use `ipynb_apply_to_notebooks` for operations across multiple notebooks

**Example Prompt**:
```
"Replace all occurrences of 'old_function' with 'new_function' in my notebook"
```

### Workflow 5: Notebook Management

**Goal**: Organize, merge, or split notebooks.

**Steps**:
1. Use `ipynb_merge_notebooks` to combine multiple notebooks
2. Use `ipynb_split_notebook` to split by headers or cell count
3. Use `ipynb_reorder_cells` to reorganize cell order

**Example Prompt**:
```
"Merge my data_prep.ipynb and analysis.ipynb into a single notebook"
```

## Available Tools

### Reading Operations
- **ipynb_read_notebook**: Get notebook structure summary
- **ipynb_list_cells**: List all cells with indices and previews
- **ipynb_get_cell**: Get specific cell content by index
- **ipynb_search_cells**: Search for patterns in cell content
- **ipynb_get_metadata**: Get notebook or cell metadata
- **ipynb_get_notebook_info**: Get comprehensive notebook info

### Writing Operations
- **ipynb_replace_cell**: Replace entire cell content
- **ipynb_insert_cell**: Insert new cell at position
- **ipynb_append_cell**: Add cell to end of notebook
- **ipynb_delete_cell**: Remove cell by index
- **ipynb_str_replace_in_cell**: Replace substring within cell

### Batch Operations
- **ipynb_replace_cells_batch**: Replace multiple cells at once
- **ipynb_delete_cells_batch**: Delete multiple cells
- **ipynb_insert_cells_batch**: Insert multiple cells
- **ipynb_search_replace_all**: Find/replace across all cells

### Notebook Management
- **ipynb_reorder_cells**: Reorder cells by new index mapping
- **ipynb_filter_cells**: Keep only cells matching criteria
- **ipynb_merge_notebooks**: Combine multiple notebooks
- **ipynb_split_notebook**: Split notebook by criteria
- **ipynb_clear_outputs**: Clear all cell outputs

### Metadata Operations
- **ipynb_update_metadata**: Update notebook or cell metadata
- **ipynb_set_kernel**: Set kernel specification
- **ipynb_list_available_kernels**: List common kernel configs
- **ipynb_sync_metadata**: Sync metadata across notebooks

### Multi-Notebook Operations
- **ipynb_search_notebooks**: Search across multiple notebooks
- **ipynb_apply_to_notebooks**: Apply operation to multiple notebooks
- **ipynb_extract_cells**: Extract matching cells to new notebook
- **ipynb_validate_notebook**: Validate notebook structure
- **ipynb_validate_notebooks_batch**: Validate multiple notebooks

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

## Best Practices

- **Use absolute paths**: Prevents path resolution issues
- **Validate before batch operations**: Check notebook structure first
- **Backup important notebooks**: Before major modifications
- **Use batch operations**: More efficient than individual calls
- **Clear outputs when sharing**: Use `ipynb_clear_outputs` for cleaner notebooks

## Example Use Cases

### Data Science Workflow
```
"Add a new analysis section to my notebook with imports for pandas and matplotlib"
```

### Documentation
```
"Add markdown documentation cells explaining each code section"
```

### Cleanup
```
"Clear all outputs and remove empty cells from my notebook"
```

### Standardization
```
"Set the kernel to python3 and update metadata for all notebooks in this folder"
```

## Additional Resources

- **Jupyter Notebook Format**: https://nbformat.readthedocs.io/
- **jupyter-editor-mcp PyPI**: https://pypi.org/project/jupyter-editor-mcp/

---

**Package**: `jupyter-editor-mcp`
**MCP Server**: `jupyter-editor`
