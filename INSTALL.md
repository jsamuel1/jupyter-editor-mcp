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
uv tool install git+https://github.com/jsamuel1/jupyter-editor-mcp.git
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
git clone https://github.com/jsamuel1/jupyter-editor-mcp.git
cd jupyter-editor-mcp

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
        "/path/to/jupyter-editor-mcp",
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

### Kiro Power (Agent Plugin)

`power/` is an [Agent Plugins](https://agent-plugins.org) 1.0.0 package: `plugin.json` (manifest), `mcp.json` (stdio server `uvx jupyter-editor-mcp`), and `skills/notebook-editing/SKILL.md` (usage guide). It needs `uvx` on PATH; nothing else to install.

- **From GitHub:** Kiro Powers panel → **Add Custom Power** → **Import power from GitHub** → `https://github.com/jsamuel1/jupyter-editor-mcp`.
- **From a local clone:** **Add Custom Power** → **Import power from a folder** → select the `power/` directory.

The portable `mcp.json` format has no `autoApprove` field, so the plugin does not pre-approve any tool. Kiro asks before each tool call until you allow it. To skip prompts for the read-only tools only, add rules to `~/.kiro/settings/permissions.yaml`. Kiro namespaces a power's MCP server (for example `power-jupyter-editor-jupyter-editor`), so check the server name Kiro shows and use it in place of `<server>`:

```yaml
rules:
  - capability: mcp
    match:
      - "<server>/ipynb_read_notebook"
      - "<server>/ipynb_list_cells"
      - "<server>/ipynb_get_cell"
      - "<server>/ipynb_search_cells"
      - "<server>/ipynb_get_notebook_info"
    effect: allow
```

Do not allow the write, delete, batch, or multi-notebook tools this way. They change notebook files.

If you installed the earlier `POWER.md` version of this power, uninstall it and install it again. Both versions use the plugin name `jupyter-editor`, but Kiro now manages the MCP server inside the power and no longer adds it to `~/.kiro/settings/mcp.json`.

## Updating

### Tool Installation

```bash
# Update to latest version
uv tool install --force .

# Or from git
uv tool install --force git+https://github.com/jsamuel1/jupyter-editor-mcp.git
```

### Development Installation

```bash
cd /path/to/jupyter-editor-mcp
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
- See [docs/DESIGN.md](docs/DESIGN.md) for API documentation
- See [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) for feature details
