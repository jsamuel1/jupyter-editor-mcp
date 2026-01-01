"""Core notebook operations using nbformat."""

import nbformat
from pathlib import Path
import re

# Project scope for file operations
_project_scope: Path | None = None


def set_project_scope(project_dir: str) -> None:
    """Set project directory to scope file operations.
    
    Args:
        project_dir: Path to project directory
        
    Raises:
        ValueError: If directory doesn't exist
    """
    global _project_scope
    path = Path(project_dir).resolve()
    if not path.is_dir():
        raise ValueError(f"Project directory does not exist: {project_dir}")
    _project_scope = path


def _validate_filepath(filepath: str) -> Path:
    """Validate filepath is within project scope if set.
    
    Args:
        filepath: Path to validate
        
    Returns:
        Resolved Path object
        
    Raises:
        ValueError: If filepath is outside project scope
    """
    path = Path(filepath).resolve()
    if _project_scope and not path.is_relative_to(_project_scope):
        raise ValueError(f"File access denied: {filepath} is outside project scope {_project_scope}")
    return path


def read_notebook_file(filepath: str) -> nbformat.NotebookNode:
    """Read notebook file and return NotebookNode.
    
    Args:
        filepath: Path to .ipynb file
        
    Returns:
        NotebookNode object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not valid notebook or outside project scope
    """
    path = _validate_filepath(filepath)
    with open(path, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)


def write_notebook_file(filepath: str, nb: nbformat.NotebookNode) -> None:
    """Write notebook to file with validation.
    
    Args:
        filepath: Path to .ipynb file
        nb: NotebookNode to write
        
    Raises:
        ValidationError: If notebook is invalid
        ValueError: If filepath is outside project scope
    """
    path = _validate_filepath(filepath)
    nbformat.validate(nb)
    temp_path = f"{path}.tmp"
    with open(temp_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    Path(temp_path).rename(path)


def get_notebook_summary(filepath: str) -> dict:
    """Get summary of notebook structure.
    
    Args:
        filepath: Path to .ipynb file
        
    Returns:
        Dict with cell_count, cell_types, kernel_info, format_version
    """
    nb = read_notebook_file(filepath)
    
    cell_types = {}
    for cell in nb['cells']:
        cell_type = cell['cell_type']
        cell_types[cell_type] = cell_types.get(cell_type, 0) + 1
    
    kernel_info = nb.metadata.get('kernelspec', {})
    
    return {
        'cell_count': len(nb['cells']),
        'cell_types': cell_types,
        'kernel_info': {
            'name': kernel_info.get('name'),
            'display_name': kernel_info.get('display_name'),
            'language': kernel_info.get('language')
        },
        'format_version': f"{nb.nbformat}.{nb.nbformat_minor}"
    }


def list_all_cells(filepath: str) -> list[dict]:
    """List all cells with index, type, and preview.
    
    Args:
        filepath: Path to .ipynb file
        
    Returns:
        List of dicts with index, type, preview, execution_count
    """
    nb = read_notebook_file(filepath)
    
    cells = []
    for i, cell in enumerate(nb['cells']):
        preview = cell['source'][:100]
        if len(cell['source']) > 100:
            preview += '...'
            
        cells.append({
            'index': i,
            'type': cell['cell_type'],
            'preview': preview,
            'execution_count': cell.get('execution_count')
        })
    
    return cells


def get_cell_content(filepath: str, cell_index: int) -> str:
    """Get content of specific cell.
    
    Args:
        filepath: Path to .ipynb file
        cell_index: Index of cell (supports negative indexing)
        
    Returns:
        Cell source content
        
    Raises:
        IndexError: If cell_index out of range
    """
    nb = read_notebook_file(filepath)
    return nb['cells'][cell_index]['source']


def search_cells(filepath: str, pattern: str, case_sensitive: bool = False) -> list[dict]:
    """Search for pattern in cell content.
    
    Args:
        filepath: Path to .ipynb file
        pattern: Search pattern (regex supported)
        case_sensitive: Whether search is case-sensitive
        
    Returns:
        List of dicts with cell_index, cell_type, match, context
    """
    nb = read_notebook_file(filepath)
    
    flags = 0 if case_sensitive else re.IGNORECASE
    regex = re.compile(pattern, flags)
    
    results = []
    for i, cell in enumerate(nb['cells']):
        source = cell['source']
        matches = regex.finditer(source)
        
        for match in matches:
            # Get context (50 chars before and after)
            start = max(0, match.start() - 50)
            end = min(len(source), match.end() + 50)
            context = source[start:end]
            
            results.append({
                'cell_index': i,
                'cell_type': cell['cell_type'],
                'match': match.group(),
                'context': context
            })
    
    return results


# Cell Modification Operations

def replace_cell_content(filepath: str, cell_index: int, new_content: str) -> None:
    """Replace entire cell content.
    
    Args:
        filepath: Path to .ipynb file
        cell_index: Index of cell to replace
        new_content: New content for cell
        
    Raises:
        IndexError: If cell_index out of range
    """
    nb = read_notebook_file(filepath)
    nb['cells'][cell_index]['source'] = new_content
    write_notebook_file(filepath, nb)


def insert_cell(filepath: str, cell_index: int, content: str, cell_type: str = "code") -> None:
    """Insert new cell at specified position.
    
    Args:
        filepath: Path to .ipynb file
        cell_index: Position to insert cell
        content: Cell content
        cell_type: Type of cell ('code', 'markdown', 'raw')
        
    Raises:
        ValueError: If cell_type is invalid
    """
    if cell_type not in ['code', 'markdown', 'raw']:
        raise ValueError(f"Invalid cell_type: {cell_type}")
    
    nb = read_notebook_file(filepath)
    
    if cell_type == 'code':
        new_cell = nbformat.v4.new_code_cell(content)
    elif cell_type == 'markdown':
        new_cell = nbformat.v4.new_markdown_cell(content)
    else:  # raw
        new_cell = nbformat.v4.new_raw_cell(content)
    
    # Remove id field if notebook format doesn't support it
    if 'id' in new_cell and nb.nbformat == 4 and nb.nbformat_minor < 5:
        del new_cell['id']
    
    nb['cells'].insert(cell_index, new_cell)
    write_notebook_file(filepath, nb)


def append_cell(filepath: str, content: str, cell_type: str = "code") -> None:
    """Append cell to end of notebook.
    
    Args:
        filepath: Path to .ipynb file
        content: Cell content
        cell_type: Type of cell ('code', 'markdown', 'raw')
        
    Raises:
        ValueError: If cell_type is invalid
    """
    if cell_type not in ['code', 'markdown', 'raw']:
        raise ValueError(f"Invalid cell_type: {cell_type}")
    
    nb = read_notebook_file(filepath)
    
    if cell_type == 'code':
        new_cell = nbformat.v4.new_code_cell(content)
    elif cell_type == 'markdown':
        new_cell = nbformat.v4.new_markdown_cell(content)
    else:  # raw
        new_cell = nbformat.v4.new_raw_cell(content)
    
    # Remove id field if notebook format doesn't support it
    if 'id' in new_cell and nb.nbformat == 4 and nb.nbformat_minor < 5:
        del new_cell['id']
    
    nb['cells'].append(new_cell)
    write_notebook_file(filepath, nb)


def delete_cell(filepath: str, cell_index: int) -> None:
    """Delete cell at specified index.
    
    Args:
        filepath: Path to .ipynb file
        cell_index: Index of cell to delete
        
    Raises:
        IndexError: If cell_index out of range
    """
    nb = read_notebook_file(filepath)
    del nb['cells'][cell_index]
    write_notebook_file(filepath, nb)


def str_replace_in_cell(filepath: str, cell_index: int, old_str: str, new_str: str) -> None:
    """Replace substring within cell content.
    
    Args:
        filepath: Path to .ipynb file
        cell_index: Index of cell
        old_str: String to replace
        new_str: Replacement string
        
    Raises:
        IndexError: If cell_index out of range
        ValueError: If old_str not found or not unique
    """
    nb = read_notebook_file(filepath)
    cell = nb['cells'][cell_index]
    source = cell['source']
    
    if old_str not in source:
        raise ValueError(f"String not found in cell: {old_str}")
    
    if source.count(old_str) > 1:
        raise ValueError(f"String appears multiple times in cell: {old_str}")
    
    cell['source'] = source.replace(old_str, new_str)
    write_notebook_file(filepath, nb)


# Metadata Operations

def get_metadata(filepath: str, cell_index: int | None = None) -> dict:
    """Get notebook or cell metadata.
    
    Args:
        filepath: Path to .ipynb file
        cell_index: Index of cell (None for notebook metadata)
        
    Returns:
        Metadata dictionary
        
    Raises:
        IndexError: If cell_index out of range
    """
    nb = read_notebook_file(filepath)
    
    if cell_index is None:
        return dict(nb.metadata)
    else:
        return dict(nb['cells'][cell_index].metadata)


def update_metadata(filepath: str, metadata: dict, cell_index: int | None = None) -> None:
    """Update notebook or cell metadata.
    
    Args:
        filepath: Path to .ipynb file
        metadata: Metadata dictionary to merge
        cell_index: Index of cell (None for notebook metadata)
        
    Raises:
        IndexError: If cell_index out of range
    """
    nb = read_notebook_file(filepath)
    
    if cell_index is None:
        nb.metadata.update(metadata)
    else:
        nb['cells'][cell_index].metadata.update(metadata)
    
    write_notebook_file(filepath, nb)


def set_kernel_spec(filepath: str, kernel_name: str, display_name: str, language: str = "python") -> None:
    """Set kernel specification.
    
    Args:
        filepath: Path to .ipynb file
        kernel_name: Kernel name (e.g., 'python3')
        display_name: Display name (e.g., 'Python 3')
        language: Programming language (default: 'python')
    """
    nb = read_notebook_file(filepath)
    
    nb.metadata['kernelspec'] = {
        'name': kernel_name,
        'display_name': display_name,
        'language': language
    }
    
    write_notebook_file(filepath, nb)


# Batch Operations - Multi-Cell

def replace_cells_batch(filepath: str, replacements: list[dict]) -> None:
    """Replace multiple cells in one operation.
    
    Args:
        filepath: Path to .ipynb file
        replacements: List of dicts with 'cell_index' and 'content' keys
        
    Raises:
        IndexError: If any cell_index out of range
    """
    nb = read_notebook_file(filepath)
    
    for replacement in replacements:
        cell_index = replacement['cell_index']
        content = replacement['content']
        nb['cells'][cell_index]['source'] = content
    
    write_notebook_file(filepath, nb)


def delete_cells_batch(filepath: str, cell_indices: list[int]) -> None:
    """Delete multiple cells by indices.
    
    Args:
        filepath: Path to .ipynb file
        cell_indices: List of cell indices to delete
        
    Raises:
        IndexError: If any cell_index out of range
    """
    nb = read_notebook_file(filepath)
    
    # Sort in reverse to delete from end first (preserves indices)
    for index in sorted(cell_indices, reverse=True):
        del nb['cells'][index]
    
    write_notebook_file(filepath, nb)


def insert_cells_batch(filepath: str, insertions: list[dict]) -> None:
    """Insert multiple cells at specified positions.
    
    Args:
        filepath: Path to .ipynb file
        insertions: List of dicts with 'cell_index', 'content', 'cell_type' keys
        
    Raises:
        ValueError: If cell_type is invalid
    """
    nb = read_notebook_file(filepath)
    
    # Sort by cell_index in reverse to maintain correct positions
    sorted_insertions = sorted(insertions, key=lambda x: x['cell_index'], reverse=True)
    
    for insertion in sorted_insertions:
        cell_index = insertion['cell_index']
        content = insertion['content']
        cell_type = insertion.get('cell_type', 'code')
        
        if cell_type not in ['code', 'markdown', 'raw']:
            raise ValueError(f"Invalid cell_type: {cell_type}")
        
        if cell_type == 'code':
            new_cell = nbformat.v4.new_code_cell(content)
        elif cell_type == 'markdown':
            new_cell = nbformat.v4.new_markdown_cell(content)
        else:  # raw
            new_cell = nbformat.v4.new_raw_cell(content)
        
        # Remove id field if notebook format doesn't support it
        if 'id' in new_cell and nb.nbformat == 4 and nb.nbformat_minor < 5:
            del new_cell['id']
        
        nb['cells'].insert(cell_index, new_cell)
    
    write_notebook_file(filepath, nb)


def search_replace_all(filepath: str, pattern: str, replacement: str, cell_type: str | None = None) -> int:
    """Search and replace across all cells.
    
    Args:
        filepath: Path to .ipynb file
        pattern: Pattern to search for (regex)
        replacement: Replacement string
        cell_type: Optional filter by cell type
        
    Returns:
        Number of replacements made
    """
    nb = read_notebook_file(filepath)
    
    regex = re.compile(pattern)
    replacements_made = 0
    
    for cell in nb['cells']:
        if cell_type and cell['cell_type'] != cell_type:
            continue
        
        original = cell['source']
        new_source, count = regex.subn(replacement, original)
        
        if count > 0:
            cell['source'] = new_source
            replacements_made += count
    
    write_notebook_file(filepath, nb)
    return replacements_made


def reorder_cells(filepath: str, new_order: list[int]) -> None:
    """Reorder cells by providing new index mapping.
    
    Args:
        filepath: Path to .ipynb file
        new_order: List of indices in desired order
        
    Raises:
        ValueError: If new_order is invalid
    """
    nb = read_notebook_file(filepath)
    
    if len(new_order) != len(nb['cells']):
        raise ValueError(f"new_order length ({len(new_order)}) must match cell count ({len(nb['cells'])})")
    
    if set(new_order) != set(range(len(nb['cells']))):
        raise ValueError("new_order must contain each index exactly once")
    
    reordered_cells = [nb['cells'][i] for i in new_order]
    nb['cells'] = reordered_cells
    
    write_notebook_file(filepath, nb)


def filter_cells(filepath: str, cell_type: str | None = None, pattern: str | None = None) -> None:
    """Keep only cells matching criteria, delete others.
    
    Args:
        filepath: Path to .ipynb file
        cell_type: Optional filter by cell type
        pattern: Optional regex pattern to match in content
        
    Raises:
        ValueError: If no filters provided
    """
    if cell_type is None and pattern is None:
        raise ValueError("At least one filter (cell_type or pattern) must be provided")
    
    nb = read_notebook_file(filepath)
    
    regex = re.compile(pattern) if pattern else None
    
    filtered_cells = []
    for cell in nb['cells']:
        keep = True
        
        if cell_type and cell['cell_type'] != cell_type:
            keep = False
        
        if pattern and regex and not regex.search(cell['source']):
            keep = False
        
        if keep:
            filtered_cells.append(cell)
    
    nb['cells'] = filtered_cells
    write_notebook_file(filepath, nb)


# Batch Operations - Multi-Notebook

def merge_notebooks(output_filepath: str, input_filepaths: list[str], add_separators: bool = True) -> None:
    """Merge multiple notebooks into one.
    
    Args:
        output_filepath: Path for merged notebook
        input_filepaths: List of notebook paths to merge
        add_separators: Whether to add separator cells between notebooks
    """
    merged_nb = nbformat.v4.new_notebook()
    
    # Use first notebook's metadata as base
    first_nb = read_notebook_file(input_filepaths[0])
    merged_nb.metadata = first_nb.metadata
    
    for i, filepath in enumerate(input_filepaths):
        nb = read_notebook_file(filepath)
        
        if add_separators and i > 0:
            separator = nbformat.v4.new_markdown_cell(f"---\n## From: {Path(filepath).name}\n---")
            merged_nb['cells'].append(separator)
        
        merged_nb['cells'].extend(nb['cells'])
    
    write_notebook_file(output_filepath, merged_nb)


def split_notebook(filepath: str, output_dir: str, split_by: str = "markdown_headers") -> list[str]:
    """Split notebook into multiple files by criteria.
    
    Args:
        filepath: Path to .ipynb file
        output_dir: Directory for output files
        split_by: Split criteria ('markdown_headers', 'cell_count', 'cell_indices')
        
    Returns:
        List of created file paths
        
    Raises:
        ValueError: If split_by is invalid
    """
    if split_by not in ['markdown_headers', 'cell_count']:
        raise ValueError(f"Invalid split_by: {split_by}. Use 'markdown_headers' or 'cell_count'")
    
    nb = read_notebook_file(filepath)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(filepath).stem
    created_files = []
    
    if split_by == 'markdown_headers':
        # Split on markdown cells starting with #
        current_cells = []
        part_num = 1
        
        for cell in nb['cells']:
            if cell['cell_type'] == 'markdown' and cell['source'].strip().startswith('#'):
                if current_cells:
                    # Save previous section
                    new_nb = nbformat.v4.new_notebook()
                    new_nb.metadata = nb.metadata
                    new_nb['cells'] = current_cells
                    output_path = str(output_dir_path / f"{base_name}_part{part_num}.ipynb")
                    write_notebook_file(output_path, new_nb)
                    created_files.append(output_path)
                    part_num += 1
                    current_cells = []
            
            current_cells.append(cell)
        
        # Save last section
        if current_cells:
            new_nb = nbformat.v4.new_notebook()
            new_nb.metadata = nb.metadata
            new_nb['cells'] = current_cells
            output_path = str(output_dir_path / f"{base_name}_part{part_num}.ipynb")
            write_notebook_file(output_path, new_nb)
            created_files.append(output_path)
    
    elif split_by == 'cell_count':
        # Split into chunks of ~10 cells
        chunk_size = 10
        for i in range(0, len(nb['cells']), chunk_size):
            new_nb = nbformat.v4.new_notebook()
            new_nb.metadata = nb.metadata
            new_nb['cells'] = nb['cells'][i:i+chunk_size]
            output_path = str(output_dir_path / f"{base_name}_part{i//chunk_size + 1}.ipynb")
            write_notebook_file(output_path, new_nb)
            created_files.append(output_path)
    
    return created_files


def apply_operation_to_notebooks(filepaths: list[str], operation: str, **params) -> dict[str, bool]:
    """Apply same operation to multiple notebooks.
    
    Args:
        filepaths: List of notebook paths
        operation: Operation name ('set_kernel', 'clear_outputs', 'update_metadata')
        **params: Parameters for the operation
        
    Returns:
        Dict mapping filepath to success status
    """
    results = {}
    
    for filepath in filepaths:
        try:
            if operation == 'set_kernel':
                set_kernel_spec(filepath, params['kernel_name'], params['display_name'], params.get('language', 'python'))
            elif operation == 'clear_outputs':
                clear_outputs(filepath)
            elif operation == 'update_metadata':
                update_metadata(filepath, params['metadata'])
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            results[filepath] = True
        except Exception:
            results[filepath] = False
    
    return results


def search_across_notebooks(filepaths: list[str], pattern: str, return_context: bool = True) -> list[dict]:
    """Search across multiple notebooks.
    
    Args:
        filepaths: List of notebook paths
        pattern: Search pattern (regex)
        return_context: Whether to include context
        
    Returns:
        List of dicts with filepath, cell_index, match, context
    """
    all_results = []
    
    for filepath in filepaths:
        try:
            results = search_cells(filepath, pattern, case_sensitive=False)
            for result in results:
                result['filepath'] = filepath
                if not return_context:
                    result.pop('context', None)
                all_results.append(result)
        except Exception:
            continue
    
    return all_results


def sync_metadata_across_notebooks(filepaths: list[str], metadata: dict, merge: bool = False) -> None:
    """Synchronize metadata across multiple notebooks.
    
    Args:
        filepaths: List of notebook paths
        metadata: Metadata to apply
        merge: Whether to merge with existing metadata (True) or replace (False)
    """
    for filepath in filepaths:
        if merge:
            update_metadata(filepath, metadata)
        else:
            nb = read_notebook_file(filepath)
            nb.metadata = metadata
            write_notebook_file(filepath, nb)


def extract_cells_from_notebooks(output_filepath: str, input_filepaths: list[str], 
                                  pattern: str | None = None, cell_type: str | None = None) -> None:
    """Extract matching cells from multiple notebooks into new notebook.
    
    Args:
        output_filepath: Path for output notebook
        input_filepaths: List of source notebook paths
        pattern: Optional regex pattern to match
        cell_type: Optional cell type filter
    """
    extracted_nb = nbformat.v4.new_notebook()
    
    # Use first notebook's metadata
    if input_filepaths:
        first_nb = read_notebook_file(input_filepaths[0])
        extracted_nb.metadata = first_nb.metadata
    
    regex = re.compile(pattern) if pattern else None
    
    for filepath in input_filepaths:
        try:
            nb = read_notebook_file(filepath)
            
            for cell in nb['cells']:
                keep = True
                
                if cell_type and cell['cell_type'] != cell_type:
                    keep = False
                
                if pattern and regex and not regex.search(cell['source']):
                    keep = False
                
                if keep:
                    extracted_nb['cells'].append(cell)
        except Exception:
            continue
    
    write_notebook_file(output_filepath, extracted_nb)


def clear_outputs(filepaths: str | list[str]) -> None:
    """Clear all outputs from code cells.
    
    Args:
        filepaths: Single filepath or list of filepaths
    """
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    
    for filepath in filepaths:
        nb = read_notebook_file(filepath)
        
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                cell['outputs'] = []
                cell['execution_count'] = None
        
        write_notebook_file(filepath, nb)


# Validation Operations

def validate_notebook_file(filepath: str) -> tuple[bool, str | None]:
    """Validate notebook structure.
    
    Args:
        filepath: Path to .ipynb file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        nb = read_notebook_file(filepath)
        nbformat.validate(nb)
        return (True, None)
    except nbformat.ValidationError as e:
        return (False, str(e))
    except Exception as e:
        return (False, f"Failed to validate: {str(e)}")


def get_notebook_info(filepath: str) -> dict:
    """Get summary information about notebook.
    
    Args:
        filepath: Path to .ipynb file
        
    Returns:
        Dict with cell_count, cell_types, kernel, format_version, file_size
    """
    nb = read_notebook_file(filepath)
    
    cell_types = {}
    for cell in nb['cells']:
        cell_type = cell['cell_type']
        cell_types[cell_type] = cell_types.get(cell_type, 0) + 1
    
    file_size = Path(filepath).stat().st_size
    
    return {
        'cell_count': len(nb['cells']),
        'cell_types': cell_types,
        'kernel': nb.metadata.get('kernelspec', {}),
        'format_version': f"{nb.nbformat}.{nb.nbformat_minor}",
        'file_size': file_size
    }


def validate_multiple_notebooks(filepaths: list[str]) -> dict[str, tuple[bool, str | None]]:
    """Validate multiple notebooks.
    
    Args:
        filepaths: List of notebook paths
        
    Returns:
        Dict mapping filepath to (is_valid, error_message) tuple
    """
    results = {}
    
    for filepath in filepaths:
        results[filepath] = validate_notebook_file(filepath)
    
    return results
