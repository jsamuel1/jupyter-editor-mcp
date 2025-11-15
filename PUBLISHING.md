# Publishing to PyPI

This project uses GitHub Actions to automatically publish to PyPI when a new release is created.

## Setup (One-time)

1. **Configure PyPI Trusted Publishing** (recommended):
   - Go to https://pypi.org/manage/account/publishing/
   - Add a new publisher:
     - PyPI Project Name: `jupyter-editor-mcp`
     - Owner: `jsamuel1`
     - Repository: `jupyter-editor-mcp`
     - Workflow: `publish.yml`
     - Environment: (leave blank)

2. **Alternative: API Token** (if trusted publishing doesn't work):
   - Create token at https://pypi.org/manage/account/token/
   - Add as GitHub secret: `PYPI_API_TOKEN`
   - Update workflow to use token instead of OIDC

## Publishing a New Version

1. **Update version** in `pyproject.toml`:
   ```toml
   version = "0.1.3"
   ```

2. **Commit and push**:
   ```bash
   git add pyproject.toml
   git commit -m "chore: bump version to 0.1.3"
   git push
   ```

3. **Create GitHub release**:
   ```bash
   gh release create v0.1.3 --title "v0.1.3" --notes "Release notes here"
   ```
   
   Or use GitHub web UI: https://github.com/jsamuel1/jupyter-editor-mcp/releases/new

4. **GitHub Actions will automatically**:
   - Build the package
   - Publish to PyPI
   - Package will be available at: https://pypi.org/project/jupyter-editor-mcp/

## Manual Publishing (for testing)

```bash
# Install build tools
uv pip install build twine

# Build
uv build

# Check the build
twine check dist/*

# Upload to TestPyPI (for testing)
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

## Verification

After publishing, verify installation:

```bash
uv pip install jupyter-editor-mcp
jupyter-editor-mcp --version
```
