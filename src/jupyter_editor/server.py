"""MCP server for Jupyter Notebook editing.

This MCP server provides 29 specialized tools for programmatic Jupyter notebook (.ipynb) editing.
All tools preserve notebook structure and format integrity using nbformat validation.

Tool Naming Convention:
- All tools are prefixed with 'ipynb_' to clearly indicate Jupyter notebook operations
- Action-oriented naming: verb_noun pattern (e.g., ipynb_read_notebook, ipynb_insert_cell)
- Batch operations end with '_batch' or include 'notebooks' for multi-file operations

Tool Categories:
- Read Operations (4): Read, list, get, search cells/notebooks
- Cell Modification (5): Replace, insert, append, delete, str_replace in cells
- Metadata Operations (4): Get/update metadata, set kernel, list kernels
- Batch Multi-Cell (6): Batch replace/delete/insert, search-replace-all, reorder, filter
- Batch Multi-Notebook (7): Merge, split, apply, search, sync, extract, clear outputs
- Validation (3): Validate single/batch notebooks, get notebook info

Best Practices:
- Use absolute file paths for reliability
- Clear notebook outputs before git commits using ipynb_clear_outputs
- Use batch operations for multiple changes to improve performance
- Always validate notebooks after complex operations
"""

import sys
import argparse
from importlib.metadata import metadata
from fastmcp import FastMCP
from . import operations
from .utils import COMMON_KERNELS

# Get package metadata
pkg_metadata = metadata("jupyter-editor-mcp")
__version__ = pkg_metadata["Version"]

# Extract URLs from metadata
_project_urls = {
    url.split(", ")[0]: url.split(", ")[1]
    for url in pkg_metadata.get_all("Project-URL") or []
}
__github_url__ = _project_urls.get("Repository", "")
__pypi_url__ = _project_urls.get("PyPI", "")

mcp = FastMCP(
    name="Jupyter Notebook Editor",
    version=__version__,
    website_url=__github_url__,
    instructions="""This MCP server provides specialized tools for programmatic Jupyter notebook (.ipynb) editing while preserving structure and format integrity.

WHEN TO USE: Use these tools instead of generic file read/write or JSON manipulation when working with .ipynb files. These tools ensure notebook format is preserved and validated.

CAPABILITIES:
- Read: Inspect notebook structure, list cells, get cell content, search content
- Modify: Replace, insert, append, delete cells; string replacement within cells  
- Metadata: Get/update notebook and cell metadata, configure kernel settings
- Batch: Perform operations across multiple cells or multiple notebooks efficiently
- Validate: Verify notebook structure and format integrity

IMPORTANT GUIDELINES:
1. Always use absolute file paths for reliable file access
2. Cell indices are 0-based; negative indices are supported (e.g., -1 for last cell)
3. Valid cell types are: 'code', 'markdown', 'raw'
4. Use batch operations when modifying multiple cells/notebooks for better performance
5. Call ipynb_clear_outputs before git commits to prevent information leakage and reduce merge conflicts

ERROR HANDLING: All tools return a dict with 'error' key on failure. Check for this key to determine success."""
)


# Read Operations

@mcp.tool
def ipynb_read_notebook(ipynb_filepath: str) -> dict:
    """Read a Jupyter notebook and return its structure summary.
    
    Use this tool to get an overview of a notebook's structure before performing
    operations. This is the recommended first step when working with an unfamiliar
    notebook.
    
    DO NOT use this tool to get actual cell content - use ipynb_get_cell or 
    ipynb_list_cells for that purpose.
    
    Args:
        ipynb_filepath: Absolute path to the .ipynb file. Relative paths may fail
                        depending on the MCP client's working directory.
    
    Returns:
        On success: {
            'cell_count': int,           # Total number of cells
            'cell_types': {'code': N, 'markdown': M, 'raw': R},  # Cell type counts
            'kernel_info': {'name': str, 'display_name': str, 'language': str},
            'format_version': str        # e.g., '4.5'
        }
        On error: {'error': str}         # Error description
    
    Example:
        ipynb_read_notebook('/path/to/notebook.ipynb')
        → {'cell_count': 10, 'cell_types': {'code': 7, 'markdown': 3}, ...}
    """
    try:
        return operations.get_notebook_summary(ipynb_filepath)
    except FileNotFoundError:
        return {"error": f"Notebook not found: {ipynb_filepath}"}
    except Exception as e:
        return {"error": f"Failed to read notebook: {str(e)}"}


@mcp.tool
def ipynb_list_cells(ipynb_filepath: str) -> dict:
    """List all cells in a notebook with their indices, types, and content previews.
    
    Use this tool to get a quick overview of all cells and their positions before
    performing targeted operations. The preview shows the first 100 characters
    of each cell's content.
    
    PREFER this over ipynb_get_cell when you need to scan multiple cells or
    find a specific cell's index.
    
    Args:
        ipynb_filepath: Absolute path to the .ipynb file.
    
    Returns:
        On success: {
            'cells': [
                {
                    'index': int,           # 0-based cell index
                    'type': str,            # 'code', 'markdown', or 'raw'
                    'preview': str,         # First 100 chars (truncated with '...')
                    'execution_count': int | None  # For code cells only
                },
                ...
            ]
        }
        On error: {'error': str}
    
    Example:
        ipynb_list_cells('/path/to/notebook.ipynb')
        → {'cells': [{'index': 0, 'type': 'code', 'preview': 'import pandas...', ...}]}
    """
    try:
        cells = operations.list_all_cells(ipynb_filepath)
        return {"cells": cells}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {ipynb_filepath}"}
    except Exception as e:
        return {"error": f"Failed to list cells: {str(e)}"}


@mcp.tool
def ipynb_get_cell(ipynb_filepath: str, cell_index: int) -> dict:
    """Get the complete content of a specific cell by its index.
    
    Use this tool when you need the full content of a known cell. Supports
    negative indexing (e.g., -1 for the last cell, -2 for second-to-last).
    
    USE ipynb_list_cells first if you don't know the cell index.
    USE ipynb_search_cells if you're looking for cells containing specific content.
    
    Args:
        ipynb_filepath: Absolute path to the .ipynb file.
        cell_index: 0-based index of the cell. Supports negative indexing.
                    Valid range: -cell_count to cell_count-1.
    
    Returns:
        On success: {'content': str}     # Complete cell source content
        On error: {'error': str}         # e.g., "Cell index 5 out of range"
    
    Example:
        ipynb_get_cell('/path/to/notebook.ipynb', 0)    # First cell
        ipynb_get_cell('/path/to/notebook.ipynb', -1)   # Last cell
    """
    try:
        content = operations.get_cell_content(ipynb_filepath, cell_index)
        return {"content": content}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {ipynb_filepath}"}
    except IndexError:
        return {"error": f"Cell index {cell_index} out of range"}
    except Exception as e:
        return {"error": f"Failed to get cell: {str(e)}"}


@mcp.tool
def ipynb_search_cells(ipynb_filepath: str, pattern: str, case_sensitive: bool = False) -> dict:
    """Search for a pattern across all cells in a notebook.
    
    Use this tool to find cells containing specific text or patterns. Supports
    regular expressions for advanced pattern matching.
    
    PREFER this over manually iterating with ipynb_get_cell when searching
    for content across the notebook.
    
    Args:
        ipynb_filepath: Absolute path to the .ipynb file.
        pattern: Search pattern. Supports regex (e.g., r'def\\s+\\w+' for function defs).
                 Special regex characters need escaping if literal match needed.
        case_sensitive: If False (default), search is case-insensitive.
    
    Returns:
        On success: {
            'results': [
                {
                    'cell_index': int,   # Index of matching cell
                    'cell_type': str,    # 'code', 'markdown', or 'raw'
                    'match': str,        # The matched text
                    'context': str       # Surrounding context for the match
                },
                ...
            ],
            'match_count': int           # Total number of matches
        }
        On error: {'error': str}
    
    Example:
        ipynb_search_cells('/path/to/notebook.ipynb', 'import pandas')
        ipynb_search_cells('/path/to/notebook.ipynb', r'def\\s+test_', case_sensitive=True)
    """
    try:
        results = operations.search_cells(ipynb_filepath, pattern, case_sensitive)
        return {"results": results, "match_count": len(results)}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {ipynb_filepath}"}
    except Exception as e:
        return {"error": f"Failed to search cells: {str(e)}"}


# Cell Modification Operations

@mcp.tool
def ipynb_replace_cell(ipynb_filepath: str, cell_index: int, new_content: str) -> dict:
    """Replace the entire content of a specific cell.
    
    Use this tool when you need to completely replace a cell's content. The cell
    type (code/markdown/raw) is preserved - only the content is replaced.
    
    USE ipynb_str_replace_in_cell for partial text replacement within a cell.
    USE ipynb_replace_cells_batch for replacing multiple cells in one operation.
    
    Args:
        ipynb_filepath: Absolute path to the .ipynb file.
        cell_index: 0-based index of the cell to replace. Supports negative indexing.
        new_content: New content for the cell. Provide as plain text - no JSON
                     escaping needed. Newlines, quotes, and special characters
                     are handled automatically.
    
    Returns:
        On success: {'success': True}
        On error: {'error': str}
    
    Example:
        ipynb_replace_cell('/path/to/notebook.ipynb', 0, 'print("Hello, World!")')
    """
    try:
        operations.replace_cell_content(ipynb_filepath, cell_index, new_content)
        return {"success": True}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {ipynb_filepath}"}
    except IndexError:
        return {"error": f"Cell index {cell_index} out of range"}
    except Exception as e:
        return {"error": f"Failed to replace cell: {str(e)}"}


@mcp.tool
def ipynb_insert_cell(ipynb_filepath: str, cell_index: int, content: str, cell_type: str = "code") -> dict:
    """Insert a new cell at the specified position.
    
    Use this tool to add a new cell at a specific position in the notebook.
    Existing cells at and after the insertion point will shift down by one index.
    
    USE ipynb_append_cell to add a cell at the end of the notebook.
    USE ipynb_insert_cells_batch for inserting multiple cells in one operation.
    
    Args:
        ipynb_filepath: Absolute path to the .ipynb file.
        cell_index: Position to insert the new cell (0-based). The new cell will
                    be at this index after insertion. Use len(cells) to append.
        content: Cell content. Provide as plain text - no JSON escaping needed.
        cell_type: Type of cell to create. Must be one of:
                   - 'code' (default): Executable Python/language code
                   - 'markdown': Rich text with Markdown formatting
                   - 'raw': Unformatted text, not rendered
    
    Returns:
        On success: {
            'success': True,
            'new_cell_count': int    # Total cells after insertion
        }
        On error: {'error': str}
    
    Note:
        After insertion, all cells originally at index >= cell_index will have
        their indices increased by 1.
    
    Example:
        ipynb_insert_cell('/path/to/notebook.ipynb', 0, '# Introduction', 'markdown')
    """
    try:
        operations.insert_cell(ipynb_filepath, cell_index, content, cell_type)
        nb = operations.read_notebook_file(ipynb_filepath)
        return {"success": True, "new_cell_count": len(nb['cells'])}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {ipynb_filepath}"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to insert cell: {str(e)}"}


@mcp.tool
def ipynb_append_cell(ipynb_filepath: str, content: str, cell_type: str = "code") -> dict:
    """Append a new cell to the end of the notebook.
    
    Use this tool when you want to add a new cell at the end without specifying
    an index. This is simpler than ipynb_insert_cell when position doesn't matter.
    
    USE ipynb_insert_cell if you need to insert at a specific position.
    
    Args:
        ipynb_filepath: Absolute path to the .ipynb file.
        content: Cell content. Provide as plain text - no JSON escaping needed.
        cell_type: Type of cell to create. Must be one of:
                   - 'code' (default): Executable Python/language code
                   - 'markdown': Rich text with Markdown formatting
                   - 'raw': Unformatted text, not rendered
    
    Returns:
        On success: {
            'success': True,
            'cell_index': int    # 0-based index of the newly appended cell
        }
        On error: {'error': str}
    
    Example:
        ipynb_append_cell('/path/to/notebook.ipynb', 'print("The End")')
        → {'success': True, 'cell_index': 15}
    """
    try:
        nb = operations.read_notebook_file(ipynb_filepath)
        cell_index = len(nb['cells'])
        operations.append_cell(ipynb_filepath, content, cell_type)
        return {"success": True, "cell_index": cell_index}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {ipynb_filepath}"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to append cell: {str(e)}"}


@mcp.tool
def ipynb_delete_cell(ipynb_filepath: str, cell_index: int) -> dict:
    """Delete a cell at the specified index.
    
    Use this tool to remove a single cell from the notebook. Cells after the
    deleted cell will shift up by one index.
    
    USE ipynb_delete_cells_batch for deleting multiple cells in one operation.
    USE ipynb_filter_cells to delete cells matching specific criteria.
    
    WARNING: This operation cannot be undone. Ensure you have the correct index.
    
    Args:
        ipynb_filepath: Absolute path to the .ipynb file.
        cell_index: 0-based index of the cell to delete. Supports negative indexing.
    
    Returns:
        On success: {
            'success': True,
            'new_cell_count': int    # Total cells remaining after deletion
        }
        On error: {'error': str}
    
    Note:
        After deletion, all cells originally at index > cell_index will have
        their indices decreased by 1.
    
    Example:
        ipynb_delete_cell('/path/to/notebook.ipynb', 2)  # Delete third cell
    """
    try:
        operations.delete_cell(ipynb_filepath, cell_index)
        nb = operations.read_notebook_file(ipynb_filepath)
        return {"success": True, "new_cell_count": len(nb['cells'])}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {ipynb_filepath}"}
    except IndexError:
        return {"error": f"Cell index {cell_index} out of range"}
    except Exception as e:
        return {"error": f"Failed to delete cell: {str(e)}"}


@mcp.tool
def ipynb_str_replace_in_cell(ipynb_filepath: str, cell_index: int, old_str: str, new_str: str) -> dict:
    """Replace a specific substring within a cell's content.
    
    Use this tool for targeted text replacement within a cell without affecting
    the rest of the content. This is safer than ipynb_replace_cell when you only
    need to change a small portion of the cell.
    
    USE ipynb_replace_cell if you need to replace the entire cell content.
    USE ipynb_search_replace_all for replacing text across all cells.
    
    Args:
        ipynb_filepath: Absolute path to the .ipynb file.
        cell_index: 0-based index of the cell. Supports negative indexing.
        old_str: The exact string to find and replace. Must exist in the cell
                 exactly once. Provide as plain text - no JSON escaping needed.
        new_str: The replacement string. Provide as plain text.
    
    Returns:
        On success: {'success': True}
        On error: {'error': str}  # e.g., "String not found" or "Multiple matches"
    
    Example:
        ipynb_str_replace_in_cell('/path/to/notebook.ipynb', 0, 'old_var', 'new_var')
    """
    try:
        operations.str_replace_in_cell(ipynb_filepath, cell_index, old_str, new_str)
        return {"success": True}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {ipynb_filepath}"}
    except IndexError:
        return {"error": f"Cell index {cell_index} out of range"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to replace string: {str(e)}"}


# Metadata Operations

@mcp.tool
def ipynb_get_metadata(ipynb_filepath: str, cell_index: int | None = None) -> dict:
    """Get metadata from a notebook or a specific cell.
    
    Use this tool to inspect notebook-level settings (kernel, language info) or
    cell-level metadata (tags, custom properties). Useful for understanding
    notebook configuration before modifications.
    
    Args:
        ipynb_filepath: Absolute path to the .ipynb file.
        cell_index: If None (default), returns notebook-level metadata.
                    If specified, returns metadata for that specific cell.
    
    Returns:
        On success: Metadata dictionary. Structure varies by level:
            Notebook-level: {
                'kernelspec': {'name': str, 'display_name': str, 'language': str},
                'language_info': {...},
                ... custom metadata
            }
            Cell-level: {
                'tags': [...],
                ... custom cell metadata
            }
        On error: {'error': str}
    
    Example:
        ipynb_get_metadata('/path/to/notebook.ipynb')        # Notebook metadata
        ipynb_get_metadata('/path/to/notebook.ipynb', 0)     # First cell metadata
    """
    try:
        return operations.get_metadata(ipynb_filepath, cell_index)
    except FileNotFoundError:
        return {"error": f"Notebook not found: {ipynb_filepath}"}
    except IndexError:
        return {"error": f"Cell index {cell_index} out of range"}
    except Exception as e:
        return {"error": f"Failed to get metadata: {str(e)}"}


@mcp.tool
def ipynb_update_metadata(ipynb_filepath: str, metadata: dict, cell_index: int | None = None) -> dict:
    """Update metadata for a notebook or a specific cell.
    
    Use this tool to modify notebook configuration (kernel, custom settings) or
    cell metadata (tags, properties). The provided metadata is merged with existing
    metadata - existing keys not in the update are preserved.
    
    USE ipynb_set_kernel for a simpler interface to change kernel settings.
    USE ipynb_sync_metadata to update metadata across multiple notebooks.
    
    Args:
        ipynb_filepath: Absolute path to the .ipynb file.
        metadata: Dictionary of metadata to merge. Keys in this dict will
                  overwrite existing keys; other existing keys are preserved.
        cell_index: If None (default), updates notebook-level metadata.
                    If specified, updates metadata for that specific cell.
    
    Returns:
        On success: {'success': True}
        On error: {'error': str}
    
    Example:
        # Add custom notebook metadata
        ipynb_update_metadata('/path/to/nb.ipynb', {'author': 'John', 'version': '1.0'})
        
        # Add tags to a cell
        ipynb_update_metadata('/path/to/nb.ipynb', {'tags': ['skip', 'test']}, cell_index=2)
    """
    try:
        operations.update_metadata(ipynb_filepath, metadata, cell_index)
        return {"success": True}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {ipynb_filepath}"}
    except IndexError:
        return {"error": f"Cell index {cell_index} out of range"}
    except Exception as e:
        return {"error": f"Failed to update metadata: {str(e)}"}


@mcp.tool
def ipynb_set_kernel(ipynb_filepath: str, kernel_name: str, display_name: str, language: str = "python") -> dict:
    """Set the kernel specification for a notebook.
    
    Use this tool to configure which kernel should execute the notebook's code
    cells. This is commonly needed when creating new notebooks or converting
    notebooks between languages.
    
    USE ipynb_list_available_kernels to see common kernel configurations.
    
    Args:
        ipynb_filepath: Absolute path to the .ipynb file.
        kernel_name: Internal kernel identifier (e.g., 'python3', 'ir', 'julia-1.9').
                     Must match an installed kernel on the target system.
        display_name: Human-readable name shown in Jupyter UI (e.g., 'Python 3').
        language: Programming language (default: 'python'). Common values:
                  'python', 'R', 'julia', 'javascript', 'scala'.
    
    Returns:
        On success: {'success': True}
        On error: {'error': str}
    
    Example:
        ipynb_set_kernel('/path/to/notebook.ipynb', 'python3', 'Python 3', 'python')
        ipynb_set_kernel('/path/to/notebook.ipynb', 'ir', 'R', 'R')
    """
    try:
        operations.set_kernel_spec(ipynb_filepath, kernel_name, display_name, language)
        return {"success": True}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {ipynb_filepath}"}
    except Exception as e:
        return {"error": f"Failed to set kernel: {str(e)}"}


@mcp.tool
def ipynb_list_available_kernels() -> dict:
    """List common Jupyter kernel configurations.
    
    Use this tool to get reference kernel configurations when setting up notebooks.
    These are common kernel specifications - actual availability depends on what's
    installed on the target system.
    
    This tool always succeeds as it returns static configuration data. There are
    no error conditions since it doesn't access the filesystem.
    
    USE ipynb_set_kernel with values from this list to configure a notebook's kernel.
    
    Returns:
        {'kernels': [
            {'name': str, 'display_name': str, 'language': str},
            ...
        ]}
    
    Example kernels returned:
        - {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
        - {'name': 'ir', 'display_name': 'R', 'language': 'R'}
        - {'name': 'julia-1.9', 'display_name': 'Julia 1.9', 'language': 'julia'}
    """
    return {"kernels": COMMON_KERNELS}


# Batch Operations - Multi-Cell

@mcp.tool
def ipynb_replace_cells_batch(ipynb_filepath: str, replacements: list[dict]) -> dict:
    """Replace content of multiple cells in a single operation.
    
    Use this tool when you need to update several cells at once. This is more
    efficient than calling ipynb_replace_cell multiple times as it performs a
    single read-modify-write cycle.
    
    PREFER this over multiple ipynb_replace_cell calls for better performance.
    
    Args:
        ipynb_filepath: Absolute path to the .ipynb file.
        replacements: List of replacement specifications. Each dict must contain:
                      - 'cell_index': int - The cell index to replace
                      - 'content': str - New content (plain text, no escaping needed)
                      Example: [
                          {'cell_index': 0, 'content': 'import pandas as pd'},
                          {'cell_index': 2, 'content': 'df.head()'}
                      ]
    
    Returns:
        On success: {
            'success': True,
            'cells_modified': int    # Number of cells that were replaced
        }
        On error: {'error': str}
    
    Example:
        ipynb_replace_cells_batch('/path/to/nb.ipynb', [
            {'cell_index': 0, 'content': '# Updated imports\\nimport numpy as np'},
            {'cell_index': 3, 'content': 'result = compute()'}
        ])
    """
    try:
        operations.replace_cells_batch(ipynb_filepath, replacements)
        return {"success": True, "cells_modified": len(replacements)}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {ipynb_filepath}"}
    except IndexError as e:
        return {"error": f"Cell index out of range: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to replace cells: {str(e)}"}


@mcp.tool
def ipynb_delete_cells_batch(ipynb_filepath: str, cell_indices: list[int]) -> dict:
    """Delete multiple cells in a single operation.
    
    Use this tool to remove several cells at once. More efficient than calling
    ipynb_delete_cell multiple times. Deletions are processed in descending
    order to maintain correct index references.
    
    IMPORTANT: Provide indices as they appear BEFORE any deletions. The tool
    handles index adjustment internally.
    
    Args:
        ipynb_filepath: Absolute path to the .ipynb file.
        cell_indices: List of cell indices to delete (0-based).
                      Example: [0, 2, 5] - deletes cells at positions 0, 2, and 5.
    
    Returns:
        On success: {
            'success': True,
            'cells_deleted': int,    # Number of cells removed
            'new_cell_count': int    # Remaining cells in notebook
        }
        On error: {'error': str}
    
    Example:
        ipynb_delete_cells_batch('/path/to/nb.ipynb', [0, 3, 7])  # Delete 3 cells
    """
    try:
        operations.delete_cells_batch(ipynb_filepath, cell_indices)
        nb = operations.read_notebook_file(ipynb_filepath)
        return {"success": True, "cells_deleted": len(cell_indices), "new_cell_count": len(nb['cells'])}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {ipynb_filepath}"}
    except IndexError as e:
        return {"error": f"Cell index out of range: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to delete cells: {str(e)}"}


@mcp.tool
def ipynb_insert_cells_batch(ipynb_filepath: str, insertions: list[dict]) -> dict:
    """Insert multiple new cells in a single operation.
    
    Use this tool to add several cells at once. Insertions are processed in
    the order provided, so earlier insertions affect the positions of later ones.
    
    TIP: To insert cells at specific final positions without worrying about
    shifting, sort insertions by index in descending order.
    
    Args:
        ipynb_filepath: Absolute path to the .ipynb file.
        insertions: List of insertion specifications. Each dict must contain:
                    - 'cell_index': int - Position to insert (0-based)
                    - 'content': str - Cell content (plain text)
                    - 'cell_type': str - 'code', 'markdown', or 'raw'
                    Example: [
                        {'cell_index': 0, 'content': '# Title', 'cell_type': 'markdown'},
                        {'cell_index': 2, 'content': 'x = 1', 'cell_type': 'code'}
                    ]
    
    Returns:
        On success: {
            'success': True,
            'cells_inserted': int    # Number of cells added
        }
        On error: {'error': str}
    
    Note:
        Insertions are processed sequentially. If you insert at index 0 first,
        then insert at index 2, the second insertion will be after the first
        inserted cell plus the original cell at index 0 and 1.
    """
    try:
        operations.insert_cells_batch(ipynb_filepath, insertions)
        return {"success": True, "cells_inserted": len(insertions)}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {ipynb_filepath}"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to insert cells: {str(e)}"}


@mcp.tool
def ipynb_search_replace_all(ipynb_filepath: str, pattern: str, replacement: str, cell_type: str | None = None) -> dict:
    """Search and replace text across all cells in a notebook.
    
    Use this tool for global find-and-replace operations. Supports regex patterns
    for complex replacements. Optionally filter to specific cell types.
    
    USE ipynb_str_replace_in_cell for replacing in a single specific cell.
    
    Args:
        ipynb_filepath: Absolute path to the .ipynb file.
        pattern: Search pattern (supports regex). Examples:
                 - 'old_name' - literal string
                 - r'\\bfoo\\b' - word boundary (matches 'foo' not 'foobar')
                 - r'v\\d+' - pattern with digits
        replacement: Replacement string. Can include regex backreferences:
                     - r'\\1' - first captured group
                     - r'\\g<name>' - named group
        cell_type: Optional filter. If specified, only cells of this type
                   are searched. Values: 'code', 'markdown', 'raw', or None (all).
    
    Returns:
        On success: {
            'success': True,
            'replacements_made': int    # Total number of replacements
        }
        On error: {'error': str}
    
    Example:
        # Replace all occurrences of 'pd' with 'pandas'
        ipynb_search_replace_all('/path/to/nb.ipynb', r'\\bpd\\b', 'pandas', 'code')
    """
    try:
        count = operations.search_replace_all(ipynb_filepath, pattern, replacement, cell_type)
        return {"success": True, "replacements_made": count}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {ipynb_filepath}"}
    except Exception as e:
        return {"error": f"Failed to search/replace: {str(e)}"}


@mcp.tool
def ipynb_reorder_cells(ipynb_filepath: str, new_order: list[int]) -> dict:
    """Reorder cells in a notebook by specifying a new arrangement.
    
    Use this tool to rearrange the order of cells without modifying their content.
    Provide a complete list of current indices in the desired new order.
    
    Args:
        ipynb_filepath: Absolute path to the .ipynb file.
        new_order: List of current cell indices in desired new order.
                   Must include ALL cell indices exactly once.
                   Example for 5 cells: [0, 2, 1, 4, 3] moves:
                   - Original cell 0 stays at position 0
                   - Original cell 2 moves to position 1
                   - Original cell 1 moves to position 2
                   - Original cell 4 moves to position 3
                   - Original cell 3 moves to position 4
    
    Returns:
        On success: {'success': True}
        On error: {'error': str}  # e.g., "Invalid order: missing index 2"
    
    Example:
        # Move first cell to the end in a 4-cell notebook
        ipynb_reorder_cells('/path/to/nb.ipynb', [1, 2, 3, 0])
    """
    try:
        operations.reorder_cells(ipynb_filepath, new_order)
        return {"success": True}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {ipynb_filepath}"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to reorder cells: {str(e)}"}


@mcp.tool
def ipynb_filter_cells(ipynb_filepath: str, cell_type: str | None = None, pattern: str | None = None) -> dict:
    """Keep only cells matching criteria, delete all others.
    
    Use this tool to remove cells that don't match specified criteria. Useful
    for cleaning up notebooks by removing certain cell types or non-matching content.
    
    WARNING: This deletes cells that DON'T match. Cannot be undone.
    
    Args:
        ipynb_filepath: Absolute path to the .ipynb file.
        cell_type: If specified, keep only cells of this type.
                   Values: 'code', 'markdown', 'raw'.
        pattern: If specified, keep only cells whose content matches this
                 regex pattern.
    
    Note: If both cell_type and pattern are specified, cells must match BOTH
    criteria to be kept.
    
    Returns:
        On success: {
            'success': True,
            'cells_kept': int,       # Number of cells remaining
            'cells_deleted': int     # Number of cells removed
        }
        On error: {'error': str}
    
    Example:
        # Keep only code cells
        ipynb_filter_cells('/path/to/nb.ipynb', cell_type='code')
        
        # Keep only cells containing 'import'
        ipynb_filter_cells('/path/to/nb.ipynb', pattern='import')
    """
    try:
        nb_before = operations.read_notebook_file(ipynb_filepath)
        cells_before = len(nb_before['cells'])
        
        operations.filter_cells(ipynb_filepath, cell_type, pattern)
        
        nb_after = operations.read_notebook_file(ipynb_filepath)
        cells_after = len(nb_after['cells'])
        
        return {"success": True, "cells_kept": cells_after, "cells_deleted": cells_before - cells_after}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {ipynb_filepath}"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to filter cells: {str(e)}"}


# Batch Operations - Multi-Notebook

@mcp.tool
def ipynb_merge_notebooks(output_ipynb_filepath: str, input_ipynb_filepaths: list[str], add_separators: bool = True) -> dict:
    """Merge multiple notebooks into a single combined notebook.
    
    Use this tool to combine content from several notebooks into one. Cells
    from each input notebook are appended in order. Optionally adds markdown
    separator cells between notebooks for clarity.
    
    Args:
        output_ipynb_filepath: Path for the merged output notebook (will be created
                               or overwritten). Use absolute path.
        input_ipynb_filepaths: List of notebook paths to merge, in order.
                               First notebook's metadata is used for the output.
        add_separators: If True (default), adds a markdown cell with the source
                        notebook filename between each merged notebook's content.
    
    Returns:
        On success: {
            'success': True,
            'total_cells': int,        # Total cells in merged notebook
            'notebooks_merged': int    # Number of input notebooks processed
        }
        On error: {'error': str}
    
    Example:
        ipynb_merge_notebooks(
            '/path/to/combined.ipynb',
            ['/path/to/part1.ipynb', '/path/to/part2.ipynb'],
            add_separators=True
        )
    """
    try:
        operations.merge_notebooks(output_ipynb_filepath, input_ipynb_filepaths, add_separators)
        nb = operations.read_notebook_file(output_ipynb_filepath)
        return {"success": True, "total_cells": len(nb['cells']), "notebooks_merged": len(input_ipynb_filepaths)}
    except FileNotFoundError as e:
        return {"error": f"Notebook not found: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to merge notebooks: {str(e)}"}


@mcp.tool
def ipynb_split_notebook(ipynb_filepath: str, output_dir: str, split_by: str = "markdown_headers") -> dict:
    """Split a notebook into multiple smaller notebooks.
    
    Use this tool to break up a large notebook into logical sections. Useful
    for organizing content or creating modular notebook collections.
    
    Args:
        ipynb_filepath: Absolute path to the source notebook.
        output_dir: Directory where split notebooks will be created.
                    Will be created if it doesn't exist.
        split_by: Splitting strategy:
                  - 'markdown_headers' (default): Split at markdown cells starting
                    with '#' (any header level). Each section becomes a notebook.
                  - 'cell_count': Split into fixed-size chunks (future feature).
    
    Returns:
        On success: {
            'success': True,
            'files_created': [str, ...]  # List of created notebook paths
        }
        On error: {'error': str}
    
    Example:
        ipynb_split_notebook('/path/to/large.ipynb', '/path/to/output/')
        → {'success': True, 'files_created': ['output/section_1.ipynb', ...]}
    """
    try:
        files = operations.split_notebook(ipynb_filepath, output_dir, split_by)
        return {"success": True, "files_created": files}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {ipynb_filepath}"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to split notebook: {str(e)}"}


@mcp.tool
def ipynb_apply_to_notebooks(ipynb_filepaths: list[str], operation: str, operation_params: dict | None = None) -> dict:
    """Apply the same operation to multiple notebooks.
    
    Use this tool to perform bulk operations across a collection of notebooks.
    More efficient than calling individual tools repeatedly.
    
    Args:
        ipynb_filepaths: List of notebook paths to process.
        operation: Operation to apply. Supported operations:
                   - 'set_kernel': Set kernel spec (requires kernel_name, display_name)
                   - 'clear_outputs': Remove all cell outputs
                   - 'update_metadata': Update notebook metadata (requires metadata dict)
        operation_params: Parameters for the operation as a dictionary.
                          Example for set_kernel: 
                          {'kernel_name': 'python3', 'display_name': 'Python 3'}
    
    Returns:
        On success: {
            'success': True,
            'results': {filepath: True/False, ...},  # Per-notebook results
            'successful': int,    # Count of successful operations
            'failed': int         # Count of failed operations
        }
        On error: {'error': str}
    
    Example:
        ipynb_apply_to_notebooks(
            ['/path/to/nb1.ipynb', '/path/to/nb2.ipynb'],
            'clear_outputs'
        )
    """
    try:
        params = operation_params or {}
        results = operations.apply_operation_to_notebooks(ipynb_filepaths, operation, **params)
        success_count = sum(1 for v in results.values() if v)
        return {"success": True, "results": results, "successful": success_count, "failed": len(results) - success_count}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to apply operation: {str(e)}"}


@mcp.tool
def ipynb_search_notebooks(ipynb_filepaths: list[str], pattern: str, return_context: bool = True) -> dict:
    """Search for a pattern across multiple notebooks.
    
    Use this tool to find content across a collection of notebooks. Returns
    matches with file locations, cell indices, and optional context.
    
    Args:
        ipynb_filepaths: List of notebook paths to search.
        pattern: Search pattern (regex supported).
        return_context: If True (default), includes surrounding text context
                        for each match.
    
    Returns:
        On success: {
            'results': [
                {
                    'filepath': str,     # Which notebook
                    'cell_index': int,   # Which cell
                    'cell_type': str,    # 'code', 'markdown', 'raw'
                    'match': str,        # Matched text
                    'context': str       # Surrounding context (if requested)
                },
                ...
            ],
            'match_count': int           # Total matches found
        }
        On error: {'error': str}
    
    Example:
        ipynb_search_notebooks(
            ['/path/to/nb1.ipynb', '/path/to/nb2.ipynb'],
            'TODO|FIXME',
            return_context=True
        )
    """
    try:
        results = operations.search_across_notebooks(ipynb_filepaths, pattern, return_context)
        return {"results": results, "match_count": len(results)}
    except Exception as e:
        return {"error": f"Failed to search notebooks: {str(e)}"}


@mcp.tool
def ipynb_sync_metadata(ipynb_filepaths: list[str], metadata: dict, merge: bool = False) -> dict:
    """Synchronize metadata across multiple notebooks.
    
    Use this tool to ensure consistent metadata across a collection of notebooks.
    Useful for standardizing author info, version numbers, or custom properties.
    
    Args:
        ipynb_filepaths: List of notebook paths to update.
        metadata: Metadata dictionary to apply to all notebooks.
        merge: If False (default), replaces entire metadata with provided dict.
               If True, merges provided metadata with existing (provided keys
               overwrite, existing keys not in provided dict are preserved).
    
    Returns:
        On success: {
            'success': True,
            'notebooks_updated': int    # Number of notebooks processed
        }
        On error: {'error': str}
    
    Example:
        ipynb_sync_metadata(
            ['/path/to/nb1.ipynb', '/path/to/nb2.ipynb'],
            {'author': 'Team', 'project': 'Analysis'},
            merge=True
        )
    """
    try:
        operations.sync_metadata_across_notebooks(ipynb_filepaths, metadata, merge)
        return {"success": True, "notebooks_updated": len(ipynb_filepaths)}
    except Exception as e:
        return {"error": f"Failed to sync metadata: {str(e)}"}


@mcp.tool
def ipynb_extract_cells(output_ipynb_filepath: str, input_ipynb_filepaths: list[str], 
                  pattern: str | None = None, cell_type: str | None = None) -> dict:
    """Extract matching cells from multiple notebooks into a new notebook.
    
    Use this tool to collect specific cells from across multiple notebooks
    based on content pattern or cell type. Creates a new notebook containing
    only the matching cells.
    
    Args:
        output_ipynb_filepath: Path for the new notebook containing extracted cells.
        input_ipynb_filepaths: List of source notebook paths to extract from.
        pattern: Optional regex pattern. If specified, only cells matching this
                 pattern are extracted.
        cell_type: Optional cell type filter ('code', 'markdown', 'raw').
                   If specified, only cells of this type are extracted.
    
    Note: If both pattern and cell_type are specified, cells must match BOTH
    criteria to be extracted.
    
    Returns:
        On success: {
            'success': True,
            'cells_extracted': int,      # Number of cells in output notebook
            'source_notebooks': int      # Number of notebooks searched
        }
        On error: {'error': str}
    
    Example:
        # Extract all markdown cells containing 'Summary'
        ipynb_extract_cells(
            '/path/to/summaries.ipynb',
            ['/path/to/nb1.ipynb', '/path/to/nb2.ipynb'],
            pattern='Summary',
            cell_type='markdown'
        )
    """
    try:
        operations.extract_cells_from_notebooks(output_ipynb_filepath, input_ipynb_filepaths, pattern, cell_type)
        nb = operations.read_notebook_file(output_ipynb_filepath)
        return {"success": True, "cells_extracted": len(nb['cells']), "source_notebooks": len(input_ipynb_filepaths)}
    except Exception as e:
        return {"error": f"Failed to extract cells: {str(e)}"}


@mcp.tool
def ipynb_clear_outputs(ipynb_filepaths: str | list[str]) -> dict:
    """Clear all execution outputs from code cells in one or more notebooks.
    
    Use this tool before committing notebooks to git to:
    - Prevent information leakage (sensitive data in outputs)
    - Reduce file size significantly
    - Avoid merge conflicts in version control
    
    This is a RECOMMENDED best practice before git commits.
    
    Args:
        ipynb_filepaths: Single path or list of paths to notebooks.
                         Accepts both string and list[str] for convenience.
    
    Returns:
        On success: {
            'success': True,
            'notebooks_processed': int    # Number of notebooks cleared
        }
        On error: {'error': str}
    
    Example:
        # Single notebook
        ipynb_clear_outputs('/path/to/notebook.ipynb')
        
        # Multiple notebooks
        ipynb_clear_outputs(['/path/to/nb1.ipynb', '/path/to/nb2.ipynb'])
    """
    try:
        operations.clear_outputs(ipynb_filepaths)
        count = 1 if isinstance(ipynb_filepaths, str) else len(ipynb_filepaths)
        return {"success": True, "notebooks_processed": count}
    except FileNotFoundError as e:
        return {"error": f"Notebook not found: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to clear outputs: {str(e)}"}


# Validation Operations

@mcp.tool
def ipynb_validate_notebook(ipynb_filepath: str) -> dict:
    """Validate a notebook's structure and format integrity.
    
    Use this tool to check if a notebook conforms to the Jupyter notebook format
    specification. Useful after manual edits or to verify notebooks before processing.
    
    This performs nbformat validation which checks:
    - Required fields are present
    - Cell types are valid
    - Metadata structure is correct
    - Format version is supported
    
    Args:
        ipynb_filepath: Absolute path to the .ipynb file.
    
    Returns:
        On valid: {'valid': True}
        On invalid: {'valid': False, 'errors': [str, ...]}
        On read error: {'error': str}
    
    Example:
        ipynb_validate_notebook('/path/to/notebook.ipynb')
        → {'valid': True}
        
        ipynb_validate_notebook('/path/to/corrupted.ipynb')
        → {'valid': False, 'errors': ['Missing required field: cells']}
    """
    try:
        is_valid, error = operations.validate_notebook_file(ipynb_filepath)
        if is_valid:
            return {"valid": True}
        else:
            return {"valid": False, "errors": [error]}
    except Exception as e:
        return {"error": f"Failed to validate notebook: {str(e)}"}


@mcp.tool
def ipynb_get_notebook_info(ipynb_filepath: str) -> dict:
    """Get comprehensive information about a notebook.
    
    Use this tool to get detailed information including structure summary,
    kernel configuration, and file size. More comprehensive than ipynb_read_notebook.
    
    Args:
        ipynb_filepath: Absolute path to the .ipynb file.
    
    Returns:
        On success: {
            'cell_count': int,
            'cell_types': {'code': N, 'markdown': M, 'raw': R},
            'kernel': {
                'name': str,
                'display_name': str,
                'language': str
            },
            'format_version': str,   # e.g., '4.5'
            'file_size': int         # Size in bytes
        }
        On error: {'error': str}
    
    Example:
        ipynb_get_notebook_info('/path/to/notebook.ipynb')
    """
    try:
        return operations.get_notebook_info(ipynb_filepath)
    except FileNotFoundError:
        return {"error": f"Notebook not found: {ipynb_filepath}"}
    except Exception as e:
        return {"error": f"Failed to get notebook info: {str(e)}"}


@mcp.tool
def ipynb_validate_notebooks_batch(ipynb_filepaths: list[str]) -> dict:
    """Validate multiple notebooks in a single operation.
    
    Use this tool to check validity of several notebooks at once. More efficient
    than calling ipynb_validate_notebook repeatedly.
    
    Args:
        ipynb_filepaths: List of notebook paths to validate.
    
    Returns:
        On success: {
            'results': {
                '/path/to/nb1.ipynb': {'valid': True},
                '/path/to/nb2.ipynb': {'valid': False, 'errors': [...]},
                ...
            },
            'total': int,     # Total notebooks checked
            'valid': int,     # Count of valid notebooks
            'invalid': int    # Count of invalid notebooks
        }
        On error: {'error': str}
    
    Example:
        ipynb_validate_notebooks_batch(['/path/to/nb1.ipynb', '/path/to/nb2.ipynb'])
        → {'results': {...}, 'total': 2, 'valid': 1, 'invalid': 1}
    """
    try:
        raw_results = operations.validate_multiple_notebooks(ipynb_filepaths)
        
        # Format results for better readability
        results = {}
        for filepath, (is_valid, error) in raw_results.items():
            if is_valid:
                results[filepath] = {"valid": True}
            else:
                results[filepath] = {"valid": False, "errors": [error]}
        
        valid_count = sum(1 for r in results.values() if r["valid"])
        
        return {
            "results": results,
            "total": len(ipynb_filepaths),
            "valid": valid_count,
            "invalid": len(ipynb_filepaths) - valid_count
        }
    except Exception as e:
        return {"error": f"Failed to validate notebooks: {str(e)}"}


def main():
    """Entry point for the MCP server."""
    parser = argparse.ArgumentParser(
        prog="jupyter-editor-mcp",
        description="MCP server for programmatic Jupyter notebook editing",
        epilog=f"GitHub: {__github_url__}\nPyPI: {__pypi_url__}",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}\nGitHub: {__github_url__}\nPyPI: {__pypi_url__}"
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport protocol (default: stdio)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for HTTP transport (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP transport (default: 8000)"
    )
    parser.add_argument(
        "--path",
        default="/mcp",
        help="Path for HTTP transport (default: /mcp)"
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Disable startup banner"
    )
    parser.add_argument(
        "--project",
        help="Project directory to scope file operations"
    )
    
    args = parser.parse_args()
    
    # Set project scope if provided
    if args.project:
        operations.set_project_scope(args.project)
    
    # Run with appropriate transport
    if args.transport == "stdio":
        mcp.run(transport="stdio", show_banner=not args.no_banner)
    else:
        mcp.run(
            transport="http",
            host=args.host,
            port=args.port,
            path=args.path,
            show_banner=not args.no_banner
        )


if __name__ == "__main__":
    main()
