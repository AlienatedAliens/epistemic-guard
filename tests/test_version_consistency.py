"""Every hard-coded release version must agree (a pre-registered study pins one version)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import epistemic_guard
from epistemic_guard import mcp_server

ROOT = Path(__file__).resolve().parents[1]


def _first(pattern: str, path: str) -> str:
    m = re.search(pattern, (ROOT / path).read_text(encoding="utf-8"), re.MULTILINE)
    assert m, path
    return m.group(1)


def test_all_release_versions_agree() -> None:
    expected = _first(r'^version = "([^"]+)"', "pyproject.toml")
    assert epistemic_guard.__version__ == expected
    assert mcp_server._SERVER_VERSION == expected
    assert _first(r'^version: "([^"]+)"', "CITATION.cff") == expected
    server = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    assert server["version"] == expected
    assert {p["version"] for p in server["packages"]} == {expected}
    assert f"## {expected}" in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
