"""Unit tests for operations.py"""

import pytest
import nbformat
from src.jupyter_editor import operations


def test_read_notebook_file(temp_notebook_file):
    """Test reading notebook file."""
    nb = operations.read_notebook_file(temp_notebook_file)
    assert isinstance(nb, nbformat.NotebookNode)
    assert len(nb['cells']) == 3


def test_read_notebook_file_not_found():
    """Test reading non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        operations.read_notebook_file("nonexistent.ipynb")


def test_write_notebook_file(temp_notebook_file, simple_notebook):
    """Test writing notebook file."""
    operations.write_notebook_file(temp_notebook_file, simple_notebook)
    nb = operations.read_notebook_file(temp_notebook_file)
    assert len(nb['cells']) == 3


def test_get_notebook_summary(temp_notebook_file):
    """Test US-001: Read Notebook Structure."""
    summary = operations.get_notebook_summary(temp_notebook_file)
    
    assert summary['cell_count'] == 3
    assert summary['cell_types'] == {'code': 2, 'markdown': 1}
    assert summary['kernel_info']['name'] == 'python3'
    assert summary['kernel_info']['display_name'] == 'Python 3'
    assert summary['kernel_info']['language'] == 'python'
    assert summary['format_version'] == '4.5'


def test_list_all_cells(temp_notebook_file):
    """Test US-002: List All Cells."""
    cells = operations.list_all_cells(temp_notebook_file)
    
    assert len(cells) == 3
    assert cells[0]['index'] == 0
    assert cells[0]['type'] == 'code'
    assert 'hello world' in cells[0]['preview']
    assert cells[1]['type'] == 'markdown'
    assert cells[2]['type'] == 'code'


def test_list_all_cells_preview_truncation(tmp_path):
    """Test cell preview truncation for long content."""
    nb = nbformat.v4.new_notebook()
    long_content = "x = " + "1" * 200
    nb['cells'] = [nbformat.v4.new_code_cell(long_content)]
    
    filepath = tmp_path / "long.ipynb"
    with open(filepath, 'w') as f:
        nbformat.write(nb, f)
    
    cells = operations.list_all_cells(str(filepath))
    assert len(cells[0]['preview']) == 103  # 100 + '...'
    assert cells[0]['preview'].endswith('...')


def test_get_cell_content(temp_notebook_file):
    """Test US-003: Get Specific Cell."""
    content = operations.get_cell_content(temp_notebook_file, 0)
    assert content == "print('hello world')"
    
    content = operations.get_cell_content(temp_notebook_file, 1)
    assert "# Test Notebook" in content


def test_get_cell_content_negative_index(temp_notebook_file):
    """Test getting cell with negative index."""
    content = operations.get_cell_content(temp_notebook_file, -1)
    assert "x = 42" in content


def test_get_cell_content_invalid_index(temp_notebook_file):
    """Test getting cell with invalid index raises IndexError."""
    with pytest.raises(IndexError):
        operations.get_cell_content(temp_notebook_file, 10)


def test_search_cells(temp_notebook_file):
    """Test US-004: Search Cell Content."""
    results = operations.search_cells(temp_notebook_file, "hello")
    
    assert len(results) == 1
    assert results[0]['cell_index'] == 0
    assert results[0]['cell_type'] == 'code'
    assert results[0]['match'] == 'hello'
    assert 'hello world' in results[0]['context']


def test_search_cells_case_insensitive(temp_notebook_file):
    """Test case-insensitive search."""
    results = operations.search_cells(temp_notebook_file, "HELLO", case_sensitive=False)
    assert len(results) == 1
    
    results = operations.search_cells(temp_notebook_file, "HELLO", case_sensitive=True)
    assert len(results) == 0


def test_search_cells_regex(temp_notebook_file):
    """Test regex pattern search."""
    results = operations.search_cells(temp_notebook_file, r"x\s*=\s*\d+")
    assert len(results) == 1
    assert results[0]['cell_index'] == 2


def test_search_cells_no_matches(temp_notebook_file):
    """Test search with no matches."""
    results = operations.search_cells(temp_notebook_file, "nonexistent")
    assert len(results) == 0
