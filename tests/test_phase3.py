"""Unit tests for Phase 3: Batch operations."""

import pytest
import nbformat
from pathlib import Path
from src.jupyter_editor import operations


# Multi-Cell Batch Operations

def test_replace_cells_batch(temp_notebook_file):
    """Test US-014: Replace Multiple Cells."""
    replacements = [
        {"cell_index": 0, "content": "print('replaced 0')"},
        {"cell_index": 2, "content": "print('replaced 2')"}
    ]
    
    operations.replace_cells_batch(temp_notebook_file, replacements)
    
    assert operations.get_cell_content(temp_notebook_file, 0) == "print('replaced 0')"
    assert operations.get_cell_content(temp_notebook_file, 2) == "print('replaced 2')"
    
    nbformat.validate(operations.read_notebook_file(temp_notebook_file))


def test_delete_cells_batch(temp_notebook_file):
    """Test US-015: Delete Multiple Cells."""
    operations.delete_cells_batch(temp_notebook_file, [0, 2])
    
    nb = operations.read_notebook_file(temp_notebook_file)
    assert len(nb['cells']) == 1
    
    nbformat.validate(nb)


def test_insert_cells_batch(temp_notebook_file):
    """Test US-016: Insert Multiple Cells."""
    insertions = [
        {"cell_index": 0, "content": "# Header", "cell_type": "markdown"},
        {"cell_index": 2, "content": "print('inserted')", "cell_type": "code"}
    ]
    
    operations.insert_cells_batch(temp_notebook_file, insertions)
    
    nb = operations.read_notebook_file(temp_notebook_file)
    assert len(nb['cells']) == 5
    assert nb['cells'][0]['cell_type'] == 'markdown'
    
    nbformat.validate(nb)


def test_search_replace_all(temp_notebook_file):
    """Test US-017: Search and Replace All."""
    count = operations.search_replace_all(temp_notebook_file, "hello", "goodbye")
    
    assert count >= 1
    content = operations.get_cell_content(temp_notebook_file, 0)
    assert "goodbye" in content
    
    nbformat.validate(operations.read_notebook_file(temp_notebook_file))


def test_search_replace_all_with_cell_type_filter(temp_notebook_file):
    """Test search/replace with cell type filter."""
    count = operations.search_replace_all(temp_notebook_file, "Test", "Demo", cell_type="markdown")
    
    assert count >= 1
    
    nbformat.validate(operations.read_notebook_file(temp_notebook_file))


def test_reorder_cells(temp_notebook_file):
    """Test US-018: Reorder Cells."""
    operations.reorder_cells(temp_notebook_file, [2, 1, 0])
    
    nb = operations.read_notebook_file(temp_notebook_file)
    assert "x = 42" in nb['cells'][0]['source']
    
    nbformat.validate(nb)


def test_reorder_cells_invalid_length(temp_notebook_file):
    """Test reorder with invalid length."""
    with pytest.raises(ValueError, match="must match cell count"):
        operations.reorder_cells(temp_notebook_file, [0, 1])


def test_reorder_cells_invalid_indices(temp_notebook_file):
    """Test reorder with invalid indices."""
    with pytest.raises(ValueError, match="must contain each index exactly once"):
        operations.reorder_cells(temp_notebook_file, [0, 0, 1])


def test_filter_cells_by_type(temp_notebook_file):
    """Test US-019: Filter Cells by type."""
    operations.filter_cells(temp_notebook_file, cell_type="code")
    
    nb = operations.read_notebook_file(temp_notebook_file)
    assert all(cell['cell_type'] == 'code' for cell in nb['cells'])
    
    nbformat.validate(nb)


def test_filter_cells_by_pattern(temp_notebook_file):
    """Test filter cells by pattern."""
    operations.filter_cells(temp_notebook_file, pattern="hello")
    
    nb = operations.read_notebook_file(temp_notebook_file)
    assert len(nb['cells']) >= 1
    assert "hello" in nb['cells'][0]['source']
    
    nbformat.validate(nb)


def test_filter_cells_no_filters(temp_notebook_file):
    """Test filter cells with no filters raises error."""
    with pytest.raises(ValueError, match="At least one filter"):
        operations.filter_cells(temp_notebook_file)


# Multi-Notebook Batch Operations

def test_merge_notebooks(tmp_path):
    """Test US-020: Merge Notebooks."""
    # Create two notebooks
    nb1 = nbformat.v4.new_notebook()
    nb1['cells'] = [nbformat.v4.new_code_cell("print('nb1')")]
    nb1.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    
    nb2 = nbformat.v4.new_notebook()
    nb2['cells'] = [nbformat.v4.new_code_cell("print('nb2')")]
    
    file1 = tmp_path / "nb1.ipynb"
    file2 = tmp_path / "nb2.ipynb"
    output = tmp_path / "merged.ipynb"
    
    with open(file1, 'w') as f:
        nbformat.write(nb1, f)
    with open(file2, 'w') as f:
        nbformat.write(nb2, f)
    
    operations.merge_notebooks(str(output), [str(file1), str(file2)], add_separators=True)
    
    merged = operations.read_notebook_file(str(output))
    assert len(merged['cells']) == 3  # 2 code cells + 1 separator
    
    nbformat.validate(merged)


def test_split_notebook_by_headers(temp_notebook_file, tmp_path):
    """Test US-021: Split Notebook by markdown headers."""
    # Add more cells with headers
    operations.insert_cell(temp_notebook_file, 0, "# Section 1", "markdown")
    operations.append_cell(temp_notebook_file, "# Section 2", "markdown")
    operations.append_cell(temp_notebook_file, "print('section 2')", "code")
    
    output_dir = tmp_path / "split"
    files = operations.split_notebook(temp_notebook_file, str(output_dir), "markdown_headers")
    
    assert len(files) >= 1
    assert all(Path(f).exists() for f in files)


def test_split_notebook_by_cell_count(temp_notebook_file, tmp_path):
    """Test split notebook by cell count."""
    output_dir = tmp_path / "split"
    files = operations.split_notebook(temp_notebook_file, str(output_dir), "cell_count")
    
    assert len(files) >= 1
    assert all(Path(f).exists() for f in files)


def test_apply_operation_to_notebooks(tmp_path):
    """Test US-022: Apply to Multiple Notebooks."""
    # Create two notebooks
    nb1 = nbformat.v4.new_notebook()
    nb1['cells'] = [nbformat.v4.new_code_cell("print('test')")]
    
    file1 = tmp_path / "nb1.ipynb"
    file2 = tmp_path / "nb2.ipynb"
    
    with open(file1, 'w') as f:
        nbformat.write(nb1, f)
    with open(file2, 'w') as f:
        nbformat.write(nb1, f)
    
    results = operations.apply_operation_to_notebooks(
        [str(file1), str(file2)],
        "set_kernel",
        kernel_name="python3.11",
        display_name="Python 3.11"
    )
    
    assert results[str(file1)] is True
    assert results[str(file2)] is True
    
    # Verify kernel was set
    nb = operations.read_notebook_file(str(file1))
    assert nb.metadata['kernelspec']['name'] == "python3.11"


def test_search_across_notebooks(tmp_path):
    """Test US-023: Search Multiple Notebooks."""
    # Create two notebooks with searchable content
    nb1 = nbformat.v4.new_notebook()
    nb1['cells'] = [nbformat.v4.new_code_cell("import pandas")]
    
    nb2 = nbformat.v4.new_notebook()
    nb2['cells'] = [nbformat.v4.new_code_cell("import numpy")]
    
    file1 = tmp_path / "nb1.ipynb"
    file2 = tmp_path / "nb2.ipynb"
    
    with open(file1, 'w') as f:
        nbformat.write(nb1, f)
    with open(file2, 'w') as f:
        nbformat.write(nb2, f)
    
    results = operations.search_across_notebooks([str(file1), str(file2)], "import")
    
    assert len(results) == 2
    assert any(r['filepath'] == str(file1) for r in results)
    assert any(r['filepath'] == str(file2) for r in results)


def test_sync_metadata_across_notebooks(tmp_path):
    """Test US-024: Sync Metadata."""
    # Create two notebooks
    nb = nbformat.v4.new_notebook()
    nb['cells'] = [nbformat.v4.new_code_cell("print('test')")]
    
    file1 = tmp_path / "nb1.ipynb"
    file2 = tmp_path / "nb2.ipynb"
    
    with open(file1, 'w') as f:
        nbformat.write(nb, f)
    with open(file2, 'w') as f:
        nbformat.write(nb, f)
    
    operations.sync_metadata_across_notebooks(
        [str(file1), str(file2)],
        {"author": "Test Author"},
        merge=True
    )
    
    # Verify metadata was synced
    nb1 = operations.read_notebook_file(str(file1))
    nb2 = operations.read_notebook_file(str(file2))
    
    assert nb1.metadata['author'] == "Test Author"
    assert nb2.metadata['author'] == "Test Author"


def test_extract_cells_from_notebooks(tmp_path):
    """Test US-025: Extract Cells."""
    # Create notebooks with different cell types
    nb1 = nbformat.v4.new_notebook()
    nb1['cells'] = [
        nbformat.v4.new_code_cell("import pandas"),
        nbformat.v4.new_markdown_cell("# Header")
    ]
    
    nb2 = nbformat.v4.new_notebook()
    nb2['cells'] = [
        nbformat.v4.new_code_cell("import numpy"),
        nbformat.v4.new_markdown_cell("# Another Header")
    ]
    
    file1 = tmp_path / "nb1.ipynb"
    file2 = tmp_path / "nb2.ipynb"
    output = tmp_path / "extracted.ipynb"
    
    with open(file1, 'w') as f:
        nbformat.write(nb1, f)
    with open(file2, 'w') as f:
        nbformat.write(nb2, f)
    
    operations.extract_cells_from_notebooks(
        str(output),
        [str(file1), str(file2)],
        cell_type="code"
    )
    
    extracted = operations.read_notebook_file(str(output))
    assert len(extracted['cells']) == 2
    assert all(cell['cell_type'] == 'code' for cell in extracted['cells'])
    
    nbformat.validate(extracted)


def test_clear_outputs_single_file(tmp_path):
    """Test US-026: Clear Outputs (single file)."""
    # Create notebook with outputs
    nb = nbformat.v4.new_notebook()
    cell = nbformat.v4.new_code_cell("print('test')")
    # Use proper nbformat output object
    cell['outputs'] = [nbformat.v4.new_output('stream', name='stdout', text='test')]
    cell['execution_count'] = 1
    nb['cells'] = [cell]
    
    filepath = tmp_path / "test.ipynb"
    with open(filepath, 'w') as f:
        nbformat.write(nb, f)
    
    operations.clear_outputs(str(filepath))
    
    nb = operations.read_notebook_file(str(filepath))
    assert nb['cells'][0]['outputs'] == []
    assert nb['cells'][0]['execution_count'] is None
    
    nbformat.validate(nb)


def test_clear_outputs_multiple_files(tmp_path):
    """Test clear outputs on multiple files."""
    # Create two notebooks with outputs
    nb = nbformat.v4.new_notebook()
    cell = nbformat.v4.new_code_cell("print('test')")
    cell['outputs'] = [nbformat.v4.new_output('stream', name='stdout', text='test')]
    cell['execution_count'] = 1
    nb['cells'] = [cell]
    
    file1 = tmp_path / "nb1.ipynb"
    file2 = tmp_path / "nb2.ipynb"
    
    with open(file1, 'w') as f:
        nbformat.write(nb, f)
    with open(file2, 'w') as f:
        nbformat.write(nb, f)
    
    operations.clear_outputs([str(file1), str(file2)])
    
    nb1 = operations.read_notebook_file(str(file1))
    nb2 = operations.read_notebook_file(str(file2))
    
    assert nb1['cells'][0]['outputs'] == []
    assert nb2['cells'][0]['outputs'] == []
