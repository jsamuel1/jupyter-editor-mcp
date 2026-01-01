"""Tests for CLI argument parsing and project scoping."""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.jupyter_editor import operations
from src import jupyter_editor


class TestPackageVersion:
    """Test package version functionality."""
    
    def test_version_matches_pyproject(self):
        """Test that package version matches pyproject.toml."""
        from importlib.metadata import version
        expected = version("jupyter-editor-mcp")
        assert jupyter_editor.__version__ == expected
    
    def test_version_format(self):
        """Test that version follows semver format."""
        import re
        assert re.match(r'^\d+\.\d+\.\d+', jupyter_editor.__version__)


class TestProjectScoping:
    """Test project directory scoping functionality."""
    
    def test_set_project_scope_valid_directory(self, tmp_path):
        """Test setting project scope with valid directory."""
        operations.set_project_scope(str(tmp_path))
        assert operations._project_scope == tmp_path.resolve()
    
    def test_set_project_scope_invalid_directory(self):
        """Test setting project scope with invalid directory."""
        with pytest.raises(ValueError, match="Project directory does not exist"):
            operations.set_project_scope("/nonexistent/directory")
    
    def test_validate_filepath_within_scope(self, tmp_path):
        """Test filepath validation within project scope."""
        operations.set_project_scope(str(tmp_path))
        test_file = tmp_path / "test.ipynb"
        test_file.touch()
        
        # Should not raise
        validated = operations._validate_filepath(str(test_file))
        assert validated == test_file.resolve()
    
    def test_validate_filepath_outside_scope(self, tmp_path):
        """Test filepath validation outside project scope."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        operations.set_project_scope(str(project_dir))
        
        outside_file = tmp_path / "outside.ipynb"
        outside_file.touch()
        
        with pytest.raises(ValueError, match="File access denied.*outside project scope"):
            operations._validate_filepath(str(outside_file))
    
    def test_validate_filepath_no_scope(self, tmp_path):
        """Test filepath validation with no project scope set."""
        # Reset scope
        operations._project_scope = None
        
        test_file = tmp_path / "test.ipynb"
        test_file.touch()
        
        # Should not raise when no scope is set
        validated = operations._validate_filepath(str(test_file))
        assert validated == test_file.resolve()
    
    def test_read_notebook_respects_scope(self, tmp_path, simple_notebook):
        """Test that read_notebook respects project scope."""
        import nbformat
        
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        operations.set_project_scope(str(project_dir))
        
        # File inside scope
        inside_file = project_dir / "inside.ipynb"
        with open(inside_file, 'w') as f:
            nbformat.write(simple_notebook, f)
        
        # Should succeed
        nb = operations.read_notebook_file(str(inside_file))
        assert nb is not None
        
        # File outside scope
        outside_file = tmp_path / "outside.ipynb"
        with open(outside_file, 'w') as f:
            nbformat.write(simple_notebook, f)
        
        # Should fail
        with pytest.raises(ValueError, match="File access denied"):
            operations.read_notebook_file(str(outside_file))
    
    def test_write_notebook_respects_scope(self, tmp_path, simple_notebook):
        """Test that write_notebook respects project scope."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        operations.set_project_scope(str(project_dir))
        
        # File inside scope
        inside_file = project_dir / "inside.ipynb"
        operations.write_notebook_file(str(inside_file), simple_notebook)
        assert inside_file.exists()
        
        # File outside scope
        outside_file = tmp_path / "outside.ipynb"
        with pytest.raises(ValueError, match="File access denied"):
            operations.write_notebook_file(str(outside_file), simple_notebook)
    
    def test_read_invalid_notebook(self, tmp_path):
        """Test reading a corrupted notebook file."""
        invalid_file = tmp_path / "invalid.ipynb"
        invalid_file.write_text('{"invalid": "json", "not": "a notebook"}')
        
        with pytest.raises(Exception):  # nbformat will raise validation error
            operations.read_notebook_file(str(invalid_file))
    
    def test_read_non_notebook_file(self, tmp_path):
        """Test reading a non-notebook file."""
        md_file = tmp_path / "readme.md"
        md_file.write_text("# This is markdown\n\nNot a notebook.")
        
        with pytest.raises(Exception):  # Should fail to parse as notebook
            operations.read_notebook_file(str(md_file))


class TestCLIArguments:
    """Test CLI argument parsing."""
    
    def test_help_argument(self):
        """Test --help displays usage information."""
        import subprocess
        result = subprocess.run(
            ['uv', 'run', 'jupyter-editor-mcp', '--help'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert 'MCP server for programmatic Jupyter notebook editing' in result.stdout
        assert '--version' in result.stdout
        assert '--transport' in result.stdout
    
    def test_version_argument(self):
        """Test --version displays correct version information."""
        import subprocess
        from importlib.metadata import version
        
        result = subprocess.run(
            ['uv', 'run', 'jupyter-editor-mcp', '--version'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert 'jupyter-editor-mcp' in result.stdout
        assert version("jupyter-editor-mcp") in result.stdout
        assert 'GitHub:' in result.stdout
        assert 'PyPI:' in result.stdout
