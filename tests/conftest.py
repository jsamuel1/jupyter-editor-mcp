"""Pytest fixtures for testing."""

import pytest
import nbformat
import tempfile
from pathlib import Path


@pytest.fixture
def simple_notebook():
    """Create a simple test notebook."""
    nb = nbformat.v4.new_notebook()
    nb['cells'] = [
        nbformat.v4.new_code_cell("print('hello world')"),
        nbformat.v4.new_markdown_cell("# Test Notebook\n\nThis is a simple test notebook."),
        nbformat.v4.new_code_cell("x = 42\ny = x * 2")
    ]
    nb.metadata['kernelspec'] = {
        'name': 'python3',
        'display_name': 'Python 3',
        'language': 'python'
    }
    return nb


@pytest.fixture
def temp_notebook_file(simple_notebook, tmp_path):
    """Create a temporary notebook file."""
    filepath = tmp_path / "test.ipynb"
    with open(filepath, 'w') as f:
        nbformat.write(simple_notebook, f)
    return str(filepath)


@pytest.fixture
def fixtures_dir():
    """Return path to fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def simple_fixture_file(fixtures_dir):
    """Return path to simple.ipynb fixture."""
    return str(fixtures_dir / "simple.ipynb")
