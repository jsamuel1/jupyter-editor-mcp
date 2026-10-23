"""Packaging checks for the Agent Plugins (Kiro power) package in power/.

Validates the plugin against the vendored Agent Plugins 1.0.0 JSON Schemas
(tests/schemas/agent-plugins-1.0.0, copied verbatim from
https://agent-plugins.org/schemas/1.0.0/) and the Agent Skills frontmatter
rules, and checks the packaged skill stays in step with the server's tools.
Schemas are vendored because the spec forbids fetching them at load time and
tests should not depend on the network.
"""

import asyncio
import json
import re
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from jupyter_editor import server

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_ROOT = REPO_ROOT / "power"
SCHEMA_DIR = Path(__file__).resolve().parent / "schemas" / "agent-plugins-1.0.0"

PLUGIN_SCHEMA_ID = "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
MCP_SCHEMA_ID = "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json"

# Deliberately conservative persistent-approval allowlist. Each of these only
# reads a notebook, but the set is not a full side-effect classification: some
# unlisted tools also only read and still stay prompt-gated.
CONSERVATIVE_ALLOWLIST = {
    "ipynb_read_notebook",
    "ipynb_list_cells",
    "ipynb_get_cell",
    "ipynb_search_cells",
    "ipynb_get_notebook_info",
}

# Agent Skills specification (https://agentskills.io/specification).
SKILL_FRONTMATTER_KEYS = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
SKILL_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _schema(name: str) -> dict:
    schema = _load_json(SCHEMA_DIR / name)
    Draft202012Validator.check_schema(schema)
    return schema


def _errors(schema: dict, instance: dict) -> list[str]:
    validator = Draft202012Validator(schema)
    return [f"{list(e.absolute_path)}: {e.message}" for e in validator.iter_errors(instance)]


def _skill_dirs() -> list[Path]:
    return sorted(p for p in (PLUGIN_ROOT / "skills").iterdir() if p.is_dir())


def _frontmatter(skill_md: Path) -> tuple[dict, str]:
    text = skill_md.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    assert match, f"{skill_md} has no YAML frontmatter"
    return yaml.safe_load(match.group(1)), match.group(2)


def _server_tool_names() -> set[str]:
    return {tool.name for tool in asyncio.run(server.mcp.list_tools())}


@pytest.fixture(scope="module")
def manifest() -> dict:
    return _load_json(PLUGIN_ROOT / "plugin.json")


@pytest.fixture(scope="module")
def mcp_config() -> dict:
    return _load_json(PLUGIN_ROOT / "mcp.json")


def test_vendored_schemas_are_canonical():
    assert _schema("plugin.schema.json")["$id"] == PLUGIN_SCHEMA_ID
    assert _schema("mcp.schema.json")["$id"] == MCP_SCHEMA_ID


def test_manifest_validates_against_schema(manifest):
    assert _errors(_schema("plugin.schema.json"), manifest) == []
    assert manifest["$schema"] == PLUGIN_SCHEMA_ID


def test_manifest_has_fields_kiro_requires(manifest):
    # kiro.dev/docs/powers/create lists these as required for a power.
    for field in ("name", "version", "description", "author", "keywords"):
        assert manifest.get(field), f"plugin.json missing {field}"
    assert manifest["name"] == "jupyter-editor"


def test_manifest_author_identity(manifest):
    assert manifest["author"] == {
        "name": "Joshua Samuel",
        "email": "sauhsoj@amazon.com",
        "url": "https://github.com/jsamuel1",
    }


def test_mcp_config_validates_against_schema(manifest, mcp_config):
    assert _errors(_schema("mcp.schema.json"), mcp_config) == []
    # The mcp.json spec version must match plugin.json (spec section 10.1).
    assert mcp_config["$schema"].split("/")[-2] == manifest["$schema"].split("/")[-2]


def test_mcp_config_launches_published_server(mcp_config):
    assert mcp_config["mcpServers"] == {
        "jupyter-editor": {
            "type": "stdio",
            "command": "uvx",
            "args": ["jupyter-editor-mcp"],
        }
    }


def test_no_client_specific_approval_fields(mcp_config):
    # autoApprove/disabled/disabledTools are not part of the portable format;
    # tool approval stays with the user's client permissions.
    for entry in mcp_config["mcpServers"].values():
        assert not {"autoApprove", "disabled", "disabledTools"} & entry.keys()


def test_no_legacy_power_files():
    assert not (PLUGIN_ROOT / "POWER.md").exists()
    assert not (PLUGIN_ROOT / "steering").exists()


def test_package_paths_stay_inside_plugin_root():
    root = PLUGIN_ROOT.resolve()
    for path in PLUGIN_ROOT.rglob("*"):
        assert path.resolve().is_relative_to(root), f"{path} escapes the plugin root"


def test_skills_follow_agent_skills_spec():
    skills = _skill_dirs()
    assert skills, "plugin ships no skills"
    for skill_dir in skills:
        skill_md = skill_dir / "SKILL.md"
        assert skill_md.is_file(), f"{skill_dir} has no SKILL.md"
        meta, body = _frontmatter(skill_md)
        assert set(meta) <= SKILL_FRONTMATTER_KEYS, f"unknown frontmatter keys: {set(meta) - SKILL_FRONTMATTER_KEYS}"
        name = meta["name"]
        assert name == skill_dir.name
        assert len(name) <= 64 and SKILL_NAME_RE.match(name) and "--" not in name
        assert isinstance(meta["description"], str) and 0 < len(meta["description"]) <= 1024
        if "compatibility" in meta:
            assert isinstance(meta["compatibility"], str) and len(meta["compatibility"]) <= 500
        if "metadata" in meta:
            assert all(isinstance(k, str) and isinstance(v, str) for k, v in meta["metadata"].items())
        assert body.strip()


def test_skill_documents_exactly_the_server_tools():
    server_tools = _server_tool_names()
    assert CONSERVATIVE_ALLOWLIST <= server_tools
    documented: set[str] = set()
    for skill_dir in _skill_dirs():
        documented |= set(re.findall(r"`(ipynb_[a-z_]+)`", (skill_dir / "SKILL.md").read_text(encoding="utf-8")))
    assert documented == server_tools


def test_skill_keeps_safety_boundaries():
    _, body = _frontmatter(PLUGIN_ROOT / "skills" / "notebook-editing" / "SKILL.md")
    assert "Executing notebook cells" in body
    assert "non-.ipynb files" in body
    assert "Use these tools only with local `.ipynb` files" in body
    assert "never executes code" in body
    section = body.split("## Safety Boundaries", 1)[1].split("\n## ", 1)[0]
    allowlist_line = next(
        line for line in section.splitlines() if line.startswith("- Recommended persistent approvals:")
    )
    assert set(re.findall(r"`(ipynb_[a-z_]+)`", allowlist_line)) == CONSERVATIVE_ALLOWLIST


def test_install_permissions_rule_matches_allowlist():
    install = (REPO_ROOT / "INSTALL.md").read_text(encoding="utf-8")
    allowed = set(re.findall(r'"<server>/(ipynb_[a-z_]+)"', install))
    assert allowed == CONSERVATIVE_ALLOWLIST
