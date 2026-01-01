# Contributing

Thanks for your interest in contributing to jupyter-editor-mcp!

## Development Setup

```bash
git clone https://github.com/jsamuel1/jupyter-editor-mcp.git
cd jupyter-editor-mcp
uv venv
uv pip install -e ".[dev]"
```

## Running Tests

```bash
uv run pytest
uv run pytest --cov  # with coverage
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests to ensure they pass
5. Commit your changes (`git commit -m 'Add my feature'`)
6. Push to your fork (`git push origin feature/my-feature`)
7. Open a Pull Request

## Code Style

- Use type hints for all functions
- Add docstrings to public functions
- Follow existing code patterns

## Reporting Issues

Use GitHub Issues for bugs and feature requests. Please include:
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Python version and OS
