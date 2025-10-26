"""MCP server for Jupyter Notebook editing."""

from fastmcp import FastMCP
from . import operations
from .utils import COMMON_KERNELS

mcp = FastMCP(name="Jupyter Notebook Editor")


# Read Operations

@mcp.tool
def read_notebook(filepath: str) -> dict:
    """Read Jupyter notebook and return structure summary.
    
    Args:
        filepath: Path to .ipynb file
        
    Returns:
        Dictionary with cell_count, cell_types, kernel_info, format_version
        or dict with 'error' key on failure
    """
    try:
        return operations.get_notebook_summary(filepath)
    except FileNotFoundError:
        return {"error": f"Notebook not found: {filepath}"}
    except Exception as e:
        return {"error": f"Failed to read notebook: {str(e)}"}


@mcp.tool
def list_cells(filepath: str) -> dict:
    """List all cells with indices, types, and content previews.
    
    Args:
        filepath: Path to .ipynb file
        
    Returns:
        Dict with 'cells' list or 'error' key on failure
    """
    try:
        cells = operations.list_all_cells(filepath)
        return {"cells": cells}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {filepath}"}
    except Exception as e:
        return {"error": f"Failed to list cells: {str(e)}"}


@mcp.tool
def get_cell(filepath: str, cell_index: int) -> dict:
    """Get content of specific cell by index.
    
    Args:
        filepath: Path to .ipynb file
        cell_index: Index of cell (supports negative indexing)
        
    Returns:
        Dict with 'content' or 'error' key
    """
    try:
        content = operations.get_cell_content(filepath, cell_index)
        return {"content": content}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {filepath}"}
    except IndexError:
        return {"error": f"Cell index {cell_index} out of range"}
    except Exception as e:
        return {"error": f"Failed to get cell: {str(e)}"}


@mcp.tool
def search_cells(filepath: str, pattern: str, case_sensitive: bool = False) -> dict:
    """Search for pattern in cell content.
    
    Args:
        filepath: Path to .ipynb file
        pattern: Search pattern (regex supported)
        case_sensitive: Whether search is case-sensitive (default: False)
        
    Returns:
        Dict with 'results' list or 'error' key
    """
    try:
        results = operations.search_cells(filepath, pattern, case_sensitive)
        return {"results": results, "match_count": len(results)}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {filepath}"}
    except Exception as e:
        return {"error": f"Failed to search cells: {str(e)}"}


# Cell Modification Operations

@mcp.tool
def replace_cell(filepath: str, cell_index: int, new_content: str) -> dict:
    """Replace entire cell content.
    
    Args:
        filepath: Path to .ipynb file
        cell_index: Index of cell to replace
        new_content: New content for cell
        
    Returns:
        Dict with 'success' or 'error' key
    """
    try:
        operations.replace_cell_content(filepath, cell_index, new_content)
        return {"success": True}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {filepath}"}
    except IndexError:
        return {"error": f"Cell index {cell_index} out of range"}
    except Exception as e:
        return {"error": f"Failed to replace cell: {str(e)}"}


@mcp.tool
def insert_cell(filepath: str, cell_index: int, content: str, cell_type: str = "code") -> dict:
    """Insert new cell at specified position.
    
    Args:
        filepath: Path to .ipynb file
        cell_index: Position to insert cell
        content: Cell content
        cell_type: Type of cell ('code', 'markdown', 'raw')
        
    Returns:
        Dict with 'success' and 'new_cell_count' or 'error' key
    """
    try:
        operations.insert_cell(filepath, cell_index, content, cell_type)
        nb = operations.read_notebook_file(filepath)
        return {"success": True, "new_cell_count": len(nb['cells'])}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {filepath}"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to insert cell: {str(e)}"}


@mcp.tool
def append_cell(filepath: str, content: str, cell_type: str = "code") -> dict:
    """Append cell to end of notebook.
    
    Args:
        filepath: Path to .ipynb file
        content: Cell content
        cell_type: Type of cell ('code', 'markdown', 'raw')
        
    Returns:
        Dict with 'success' and 'cell_index' or 'error' key
    """
    try:
        nb = operations.read_notebook_file(filepath)
        cell_index = len(nb['cells'])
        operations.append_cell(filepath, content, cell_type)
        return {"success": True, "cell_index": cell_index}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {filepath}"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to append cell: {str(e)}"}


@mcp.tool
def delete_cell(filepath: str, cell_index: int) -> dict:
    """Delete cell at specified index.
    
    Args:
        filepath: Path to .ipynb file
        cell_index: Index of cell to delete
        
    Returns:
        Dict with 'success' and 'new_cell_count' or 'error' key
    """
    try:
        operations.delete_cell(filepath, cell_index)
        nb = operations.read_notebook_file(filepath)
        return {"success": True, "new_cell_count": len(nb['cells'])}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {filepath}"}
    except IndexError:
        return {"error": f"Cell index {cell_index} out of range"}
    except Exception as e:
        return {"error": f"Failed to delete cell: {str(e)}"}


@mcp.tool
def str_replace_in_cell(filepath: str, cell_index: int, old_str: str, new_str: str) -> dict:
    """Replace substring within cell content.
    
    Args:
        filepath: Path to .ipynb file
        cell_index: Index of cell
        old_str: String to replace
        new_str: Replacement string
        
    Returns:
        Dict with 'success' or 'error' key
    """
    try:
        operations.str_replace_in_cell(filepath, cell_index, old_str, new_str)
        return {"success": True}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {filepath}"}
    except IndexError:
        return {"error": f"Cell index {cell_index} out of range"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to replace string: {str(e)}"}


# Metadata Operations

@mcp.tool
def get_metadata(filepath: str, cell_index: int | None = None) -> dict:
    """Get notebook or cell metadata.
    
    Args:
        filepath: Path to .ipynb file
        cell_index: Index of cell (None for notebook metadata)
        
    Returns:
        Metadata dictionary or dict with 'error' key
    """
    try:
        return operations.get_metadata(filepath, cell_index)
    except FileNotFoundError:
        return {"error": f"Notebook not found: {filepath}"}
    except IndexError:
        return {"error": f"Cell index {cell_index} out of range"}
    except Exception as e:
        return {"error": f"Failed to get metadata: {str(e)}"}


@mcp.tool
def update_metadata(filepath: str, metadata: dict, cell_index: int | None = None) -> dict:
    """Update notebook or cell metadata.
    
    Args:
        filepath: Path to .ipynb file
        metadata: Metadata dictionary to merge
        cell_index: Index of cell (None for notebook metadata)
        
    Returns:
        Dict with 'success' or 'error' key
    """
    try:
        operations.update_metadata(filepath, metadata, cell_index)
        return {"success": True}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {filepath}"}
    except IndexError:
        return {"error": f"Cell index {cell_index} out of range"}
    except Exception as e:
        return {"error": f"Failed to update metadata: {str(e)}"}


@mcp.tool
def set_kernel(filepath: str, kernel_name: str, display_name: str, language: str = "python") -> dict:
    """Set kernel specification.
    
    Args:
        filepath: Path to .ipynb file
        kernel_name: Kernel name (e.g., 'python3')
        display_name: Display name (e.g., 'Python 3')
        language: Programming language (default: 'python')
        
    Returns:
        Dict with 'success' or 'error' key
    """
    try:
        operations.set_kernel_spec(filepath, kernel_name, display_name, language)
        return {"success": True}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {filepath}"}
    except Exception as e:
        return {"error": f"Failed to set kernel: {str(e)}"}


@mcp.tool
def list_available_kernels() -> dict:
    """List common kernel configurations.
    
    Returns:
        Dict with 'kernels' list
    """
    return {"kernels": COMMON_KERNELS}


# Batch Operations - Multi-Cell

@mcp.tool
def replace_cells_batch(filepath: str, replacements: list[dict]) -> dict:
    """Replace multiple cells in one operation.
    
    Args:
        filepath: Path to .ipynb file
        replacements: List of dicts with 'cell_index' and 'content' keys
        
    Returns:
        Dict with 'success' and 'cells_modified' or 'error' key
    """
    try:
        operations.replace_cells_batch(filepath, replacements)
        return {"success": True, "cells_modified": len(replacements)}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {filepath}"}
    except IndexError as e:
        return {"error": f"Cell index out of range: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to replace cells: {str(e)}"}


@mcp.tool
def delete_cells_batch(filepath: str, cell_indices: list[int]) -> dict:
    """Delete multiple cells by indices.
    
    Args:
        filepath: Path to .ipynb file
        cell_indices: List of cell indices to delete
        
    Returns:
        Dict with 'success', 'cells_deleted', 'new_cell_count' or 'error' key
    """
    try:
        operations.delete_cells_batch(filepath, cell_indices)
        nb = operations.read_notebook_file(filepath)
        return {"success": True, "cells_deleted": len(cell_indices), "new_cell_count": len(nb['cells'])}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {filepath}"}
    except IndexError as e:
        return {"error": f"Cell index out of range: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to delete cells: {str(e)}"}


@mcp.tool
def insert_cells_batch(filepath: str, insertions: list[dict]) -> dict:
    """Insert multiple cells at specified positions.
    
    Args:
        filepath: Path to .ipynb file
        insertions: List of dicts with 'cell_index', 'content', 'cell_type' keys
        
    Returns:
        Dict with 'success' and 'cells_inserted' or 'error' key
    """
    try:
        operations.insert_cells_batch(filepath, insertions)
        return {"success": True, "cells_inserted": len(insertions)}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {filepath}"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to insert cells: {str(e)}"}


@mcp.tool
def search_replace_all(filepath: str, pattern: str, replacement: str, cell_type: str | None = None) -> dict:
    """Search and replace across all cells.
    
    Args:
        filepath: Path to .ipynb file
        pattern: Pattern to search for (regex)
        replacement: Replacement string
        cell_type: Optional filter by cell type
        
    Returns:
        Dict with 'success' and 'replacements_made' or 'error' key
    """
    try:
        count = operations.search_replace_all(filepath, pattern, replacement, cell_type)
        return {"success": True, "replacements_made": count}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {filepath}"}
    except Exception as e:
        return {"error": f"Failed to search/replace: {str(e)}"}


@mcp.tool
def reorder_cells(filepath: str, new_order: list[int]) -> dict:
    """Reorder cells by providing new index mapping.
    
    Args:
        filepath: Path to .ipynb file
        new_order: List of indices in desired order
        
    Returns:
        Dict with 'success' or 'error' key
    """
    try:
        operations.reorder_cells(filepath, new_order)
        return {"success": True}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {filepath}"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to reorder cells: {str(e)}"}


@mcp.tool
def filter_cells(filepath: str, cell_type: str | None = None, pattern: str | None = None) -> dict:
    """Keep only cells matching criteria, delete others.
    
    Args:
        filepath: Path to .ipynb file
        cell_type: Optional filter by cell type
        pattern: Optional regex pattern to match
        
    Returns:
        Dict with 'success', 'cells_kept' or 'error' key
    """
    try:
        nb_before = operations.read_notebook_file(filepath)
        cells_before = len(nb_before['cells'])
        
        operations.filter_cells(filepath, cell_type, pattern)
        
        nb_after = operations.read_notebook_file(filepath)
        cells_after = len(nb_after['cells'])
        
        return {"success": True, "cells_kept": cells_after, "cells_deleted": cells_before - cells_after}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {filepath}"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to filter cells: {str(e)}"}


# Batch Operations - Multi-Notebook

@mcp.tool
def merge_notebooks(output_filepath: str, input_filepaths: list[str], add_separators: bool = True) -> dict:
    """Merge multiple notebooks into one.
    
    Args:
        output_filepath: Path for merged notebook
        input_filepaths: List of notebook paths to merge
        add_separators: Whether to add separator cells between notebooks
        
    Returns:
        Dict with 'success', 'total_cells', 'notebooks_merged' or 'error' key
    """
    try:
        operations.merge_notebooks(output_filepath, input_filepaths, add_separators)
        nb = operations.read_notebook_file(output_filepath)
        return {"success": True, "total_cells": len(nb['cells']), "notebooks_merged": len(input_filepaths)}
    except FileNotFoundError as e:
        return {"error": f"Notebook not found: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to merge notebooks: {str(e)}"}


@mcp.tool
def split_notebook(filepath: str, output_dir: str, split_by: str = "markdown_headers") -> dict:
    """Split notebook into multiple files by criteria.
    
    Args:
        filepath: Path to .ipynb file
        output_dir: Directory for output files
        split_by: Split criteria ('markdown_headers' or 'cell_count')
        
    Returns:
        Dict with 'success' and 'files_created' or 'error' key
    """
    try:
        files = operations.split_notebook(filepath, output_dir, split_by)
        return {"success": True, "files_created": files}
    except FileNotFoundError:
        return {"error": f"Notebook not found: {filepath}"}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to split notebook: {str(e)}"}


@mcp.tool
def apply_to_notebooks(filepaths: list[str], operation: str, operation_params: dict | None = None) -> dict:
    """Apply same operation to multiple notebooks.
    
    Args:
        filepaths: List of notebook paths
        operation: Operation name ('set_kernel', 'clear_outputs', 'update_metadata')
        operation_params: Parameters for the operation as a dictionary
        
    Returns:
        Dict with 'success' and 'results' or 'error' key
    """
    try:
        params = operation_params or {}
        results = operations.apply_operation_to_notebooks(filepaths, operation, **params)
        success_count = sum(1 for v in results.values() if v)
        return {"success": True, "results": results, "successful": success_count, "failed": len(results) - success_count}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to apply operation: {str(e)}"}


@mcp.tool
def search_notebooks(filepaths: list[str], pattern: str, return_context: bool = True) -> dict:
    """Search across multiple notebooks.
    
    Args:
        filepaths: List of notebook paths
        pattern: Search pattern (regex)
        return_context: Whether to include context
        
    Returns:
        Dict with 'results' and 'match_count' or 'error' key
    """
    try:
        results = operations.search_across_notebooks(filepaths, pattern, return_context)
        return {"results": results, "match_count": len(results)}
    except Exception as e:
        return {"error": f"Failed to search notebooks: {str(e)}"}


@mcp.tool
def sync_metadata(filepaths: list[str], metadata: dict, merge: bool = False) -> dict:
    """Synchronize metadata across multiple notebooks.
    
    Args:
        filepaths: List of notebook paths
        metadata: Metadata to apply
        merge: Whether to merge with existing metadata
        
    Returns:
        Dict with 'success' and 'notebooks_updated' or 'error' key
    """
    try:
        operations.sync_metadata_across_notebooks(filepaths, metadata, merge)
        return {"success": True, "notebooks_updated": len(filepaths)}
    except Exception as e:
        return {"error": f"Failed to sync metadata: {str(e)}"}


@mcp.tool
def extract_cells(output_filepath: str, input_filepaths: list[str], 
                  pattern: str | None = None, cell_type: str | None = None) -> dict:
    """Extract matching cells from multiple notebooks into new notebook.
    
    Args:
        output_filepath: Path for output notebook
        input_filepaths: List of source notebook paths
        pattern: Optional regex pattern to match
        cell_type: Optional cell type filter
        
    Returns:
        Dict with 'success', 'cells_extracted', 'source_notebooks' or 'error' key
    """
    try:
        operations.extract_cells_from_notebooks(output_filepath, input_filepaths, pattern, cell_type)
        nb = operations.read_notebook_file(output_filepath)
        return {"success": True, "cells_extracted": len(nb['cells']), "source_notebooks": len(input_filepaths)}
    except Exception as e:
        return {"error": f"Failed to extract cells: {str(e)}"}


@mcp.tool
def clear_outputs(filepaths: str | list[str]) -> dict:
    """Clear all outputs from code cells in one or more notebooks.
    
    Args:
        filepaths: Single filepath or list of filepaths
        
    Returns:
        Dict with 'success' and 'notebooks_processed' or 'error' key
    """
    try:
        operations.clear_outputs(filepaths)
        count = 1 if isinstance(filepaths, str) else len(filepaths)
        return {"success": True, "notebooks_processed": count}
    except FileNotFoundError as e:
        return {"error": f"Notebook not found: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to clear outputs: {str(e)}"}


# Validation Operations

@mcp.tool
def validate_notebook(filepath: str) -> dict:
    """Validate notebook structure.
    
    Args:
        filepath: Path to .ipynb file
        
    Returns:
        Dict with 'valid' boolean and optional 'errors' list
    """
    try:
        is_valid, error = operations.validate_notebook_file(filepath)
        if is_valid:
            return {"valid": True}
        else:
            return {"valid": False, "errors": [error]}
    except Exception as e:
        return {"error": f"Failed to validate notebook: {str(e)}"}


@mcp.tool
def get_notebook_info(filepath: str) -> dict:
    """Get summary information about notebook.
    
    Args:
        filepath: Path to .ipynb file
        
    Returns:
        Dict with cell_count, cell_types, kernel, format_version, file_size or 'error' key
    """
    try:
        return operations.get_notebook_info(filepath)
    except FileNotFoundError:
        return {"error": f"Notebook not found: {filepath}"}
    except Exception as e:
        return {"error": f"Failed to get notebook info: {str(e)}"}


@mcp.tool
def validate_notebooks_batch(filepaths: list[str]) -> dict:
    """Validate multiple notebooks.
    
    Args:
        filepaths: List of notebook paths
        
    Returns:
        Dict with 'results' mapping filepath to validation status
    """
    try:
        raw_results = operations.validate_multiple_notebooks(filepaths)
        
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
            "total": len(filepaths),
            "valid": valid_count,
            "invalid": len(filepaths) - valid_count
        }
    except Exception as e:
        return {"error": f"Failed to validate notebooks: {str(e)}"}


def main():
    """Entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
