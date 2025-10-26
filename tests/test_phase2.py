"""Unit tests for Phase 2: Cell modification and metadata operations."""

import pytest
import nbformat
from jupyter_editor import operations


# Cell Modification Tests

def test_replace_cell_content(temp_notebook_file):
    """Test US-005: Replace Cell Content."""
    operations.replace_cell_content(temp_notebook_file, 0, "print('replaced')")
    
    content = operations.get_cell_content(temp_notebook_file, 0)
    assert content == "print('replaced')"
    
    # Verify format preserved
    nb = operations.read_notebook_file(temp_notebook_file)
    nbformat.validate(nb)


def test_insert_cell(temp_notebook_file):
    """Test US-006: Insert Cell."""
    operations.insert_cell(temp_notebook_file, 1, "print('inserted')", "code")
    
    nb = operations.read_notebook_file(temp_notebook_file)
    assert len(nb['cells']) == 4
    assert nb['cells'][1]['source'] == "print('inserted')"
    assert nb['cells'][1]['cell_type'] == 'code'
    
    nbformat.validate(nb)


def test_insert_markdown_cell(temp_notebook_file):
    """Test inserting markdown cell."""
    operations.insert_cell(temp_notebook_file, 0, "# Header", "markdown")
    
    nb = operations.read_notebook_file(temp_notebook_file)
    assert nb['cells'][0]['cell_type'] == 'markdown'
    assert nb['cells'][0]['source'] == "# Header"


def test_insert_cell_invalid_type(temp_notebook_file):
    """Test inserting cell with invalid type."""
    with pytest.raises(ValueError, match="Invalid cell_type"):
        operations.insert_cell(temp_notebook_file, 0, "content", "invalid")


def test_append_cell(temp_notebook_file):
    """Test US-007: Append Cell."""
    operations.append_cell(temp_notebook_file, "print('appended')", "code")
    
    nb = operations.read_notebook_file(temp_notebook_file)
    assert len(nb['cells']) == 4
    assert nb['cells'][-1]['source'] == "print('appended')"
    
    nbformat.validate(nb)


def test_delete_cell(temp_notebook_file):
    """Test US-008: Delete Cell."""
    operations.delete_cell(temp_notebook_file, 1)
    
    nb = operations.read_notebook_file(temp_notebook_file)
    assert len(nb['cells']) == 2
    
    nbformat.validate(nb)


def test_delete_cell_invalid_index(temp_notebook_file):
    """Test deleting cell with invalid index."""
    with pytest.raises(IndexError):
        operations.delete_cell(temp_notebook_file, 10)


def test_str_replace_in_cell(temp_notebook_file):
    """Test US-009: String Replace in Cell."""
    operations.str_replace_in_cell(temp_notebook_file, 0, "hello world", "goodbye world")
    
    content = operations.get_cell_content(temp_notebook_file, 0)
    assert content == "print('goodbye world')"
    
    nbformat.validate(operations.read_notebook_file(temp_notebook_file))


def test_str_replace_in_cell_not_found(temp_notebook_file):
    """Test string replace when string not found."""
    with pytest.raises(ValueError, match="String not found"):
        operations.str_replace_in_cell(temp_notebook_file, 0, "nonexistent", "new")


def test_str_replace_in_cell_multiple_occurrences(temp_notebook_file):
    """Test string replace when string appears multiple times."""
    # Add cell with duplicate string
    operations.replace_cell_content(temp_notebook_file, 0, "test test")
    
    with pytest.raises(ValueError, match="appears multiple times"):
        operations.str_replace_in_cell(temp_notebook_file, 0, "test", "new")


# Metadata Tests

def test_get_notebook_metadata(temp_notebook_file):
    """Test US-010: Get Metadata (notebook level)."""
    metadata = operations.get_metadata(temp_notebook_file)
    
    assert 'kernelspec' in metadata
    assert metadata['kernelspec']['name'] == 'python3'


def test_get_cell_metadata(temp_notebook_file):
    """Test US-010: Get Metadata (cell level)."""
    metadata = operations.get_metadata(temp_notebook_file, cell_index=0)
    
    assert isinstance(metadata, dict)


def test_update_notebook_metadata(temp_notebook_file):
    """Test US-011: Update Metadata (notebook level)."""
    operations.update_metadata(temp_notebook_file, {"author": "Test Author"})
    
    metadata = operations.get_metadata(temp_notebook_file)
    assert metadata['author'] == "Test Author"
    
    nbformat.validate(operations.read_notebook_file(temp_notebook_file))


def test_update_cell_metadata(temp_notebook_file):
    """Test US-011: Update Metadata (cell level)."""
    operations.update_metadata(temp_notebook_file, {"tags": ["important"]}, cell_index=0)
    
    metadata = operations.get_metadata(temp_notebook_file, cell_index=0)
    assert metadata['tags'] == ["important"]
    
    nbformat.validate(operations.read_notebook_file(temp_notebook_file))


def test_set_kernel_spec(temp_notebook_file):
    """Test US-012: Set Kernel."""
    operations.set_kernel_spec(temp_notebook_file, "python3.11", "Python 3.11", "python")
    
    metadata = operations.get_metadata(temp_notebook_file)
    assert metadata['kernelspec']['name'] == "python3.11"
    assert metadata['kernelspec']['display_name'] == "Python 3.11"
    assert metadata['kernelspec']['language'] == "python"
    
    nbformat.validate(operations.read_notebook_file(temp_notebook_file))
