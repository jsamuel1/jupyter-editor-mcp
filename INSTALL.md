# Installation Guide

## Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager

## Installation Methods

### Method 1: Install as a Tool (Recommended for Users)

This method installs the MCP server as a standalone tool that can be run from anywhere.

```bash
# Install from local directory
cd /path/to/jupyter-editor
uv tool install .

# Or install directly from git
uv tool install git+https://github.com/yourusername/jupyter-editor.git
```

**Verify installation:**

```bash
jupyter-editor-mcp --help
```

The tool is now available globally and can be used with any MCP client.

### Method 2: Development Installation (For Contributors)

This method installs the package in editable mode for development.

```bash
# Clone the repository
git clone <repository-url>
cd jupyter-editor

# Create virtual environment
uv venv

# Install in development mode with dev dependencies
uv pip install -e ".[dev]"
```

**Run tests:**

```bash
uv run pytest
```

## Configuration

### Claude Desktop

#### For Tool Installation

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "jupyter-editor": {
      "command": "jupyter-editor-mcp"
    }
  }
}
```

#### For Development Installation

```json
{
  "mcpServers": {
    "jupyter-editor": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/jupyter-editor",
        "python",
        "-m",
        "jupyter_editor.server"
      ]
    }
  }
}
```

### VS Code (with MCP extension)

Add to `.vscode/settings.json`:

```json
{
  "mcp.servers": {
    "jupyter-editor": {
      "command": "jupyter-editor-mcp"
    }
  }
}
```

## Updating

### Tool Installation

```bash
# Update to latest version
uv tool install --force .

# Or from git
uv tool install --force git+https://github.com/yourusername/jupyter-editor.git
```

### Development Installation

```bash
cd /path/to/jupyter-editor
git pull
uv pip install -e ".[dev]"
```

## Uninstalling

### Tool Installation

```bash
uv tool uninstall jupyter-editor-mcp
```

### Development Installation

```bash
uv pip uninstall jupyter-editor-mcp
```

## Troubleshooting

### Tool not found after installation

Make sure uv's tool directory is in your PATH:

```bash
# Add to ~/.zshrc or ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
```

### Import errors

Ensure all dependencies are installed:

```bash
uv pip install nbformat fastmcp
```

### MCP server not connecting

1. Check that the command in your MCP client config is correct
2. Verify the server starts manually: `jupyter-editor-mcp`
3. Check MCP client logs for connection errors

## Verification

Test that all tools are available:

```bash
# Start the server (will wait for MCP client connection)
jupyter-editor-mcp
```

The server should start without errors and wait for connections.

## Next Steps

- See [README.md](README.md) for usage examples
- See [DESIGN.md](DESIGN.md) for API documentation
- See [REQUIREMENTS.md](REQUIREMENTS.md) for feature details
