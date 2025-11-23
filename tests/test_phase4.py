"""Unit tests for Phase 4: Validation operations."""

import pytest
import nbformat
from src.jupyter_editor import operations


def test_validate_notebook_file_valid(temp_notebook_file):
    """Test US-027: Validate Notebook (valid notebook)."""
    is_valid, error = operations.validate_notebook_file(temp_notebook_file)
    
    assert is_valid is True
    assert error is None


def test_validate_notebook_file_invalid(tmp_path):
    """Test validate notebook with invalid notebook."""
    # Create notebook with invalid cell structure
    nb = nbformat.v4.new_notebook()
    # Manually create invalid cell (missing required 'source' field)
    invalid_cell = {'cell_type': 'code', 'metadata': {}}  # Missing 'source'
    nb['cells'] = [invalid_cell]
    
    filepath = tmp_path / "invalid.ipynb"
    import json
    # Convert to dict manually
    nb_dict = dict(nb)
    nb_dict['cells'] = [invalid_cell]
    with open(filepath, 'w') as f:
        json.dump(nb_dict, f)
    
    is_valid, error = operations.validate_notebook_file(str(filepath))
    
    assert is_valid is False
    assert error is not None


def test_get_notebook_info(temp_notebook_file):
    """Test US-028: Get Notebook Info."""
    info = operations.get_notebook_info(temp_notebook_file)
    
    assert info['cell_count'] == 3
    assert 'code' in info['cell_types']
    assert 'markdown' in info['cell_types']
    assert info['cell_types']['code'] == 2
    assert info['cell_types']['markdown'] == 1
    assert 'kernel' in info
    assert info['kernel']['name'] == 'python3'
    assert info['format_version'] == '4.5'
    assert info['file_size'] > 0


def test_validate_multiple_notebooks(tmp_path):
    """Test US-029: Validate Multiple Notebooks."""
    # Create valid notebook
    nb1 = nbformat.v4.new_notebook()
    nb1['cells'] = [nbformat.v4.new_code_cell("print('test')")]
    
    file1 = tmp_path / "valid.ipynb"
    with open(file1, 'w') as f:
        nbformat.write(nb1, f)
    
    # Create invalid notebook with invalid cell
    nb2 = nbformat.v4.new_notebook()
    invalid_cell = {'cell_type': 'code', 'metadata': {}}  # Missing 'source'
    
    file2 = tmp_path / "invalid.ipynb"
    import json
    nb_dict = dict(nb2)
    nb_dict['cells'] = [invalid_cell]
    with open(file2, 'w') as f:
        json.dump(nb_dict, f)
    
    results = operations.validate_multiple_notebooks([str(file1), str(file2)])
    
    assert len(results) == 2
    assert results[str(file1)][0] is True  # valid
    assert results[str(file2)][0] is False  # invalid
    assert results[str(file2)][1] is not None  # has error message
