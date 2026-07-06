"""Docs-vs-code consistency checks.

Code is the source of truth. When one of these tests fails, fix the doc to
match the code, not the test. If the stale claim was already published,
record the correction in docs/HISTORY.md as well.

Pure text parsing on purpose: these tests need no package install, no API
key, and no fixtures, so they run in every CI job.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Version and platform claims
# ---------------------------------------------------------------------------
def test_package_version_consistent():
    pyproject = re.search(r'^version = "([^"]+)"', read("pyproject.toml"), re.M)
    init = re.search(r'^__version__ = "([^"]+)"', read("widemem/__init__.py"), re.M)
    assert pyproject and init, "version string missing from pyproject.toml or widemem/__init__.py"
    assert pyproject.group(1) == init.group(1), (
        f"pyproject.toml says {pyproject.group(1)}, "
        f"widemem/__init__.py says {init.group(1)}"
    )


def test_python_floor_matches_readme_badge():
    floor = re.search(r'requires-python = ">=(\d+\.\d+)"', read("pyproject.toml"))
    assert floor, "requires-python missing from pyproject.toml"
    badge = re.search(r"python-(\d+\.\d+)%2B", read("README.md"))
    assert badge, "python version badge missing from README.md"
    assert badge.group(1) == floor.group(1), (
        f"README badge claims python {badge.group(1)}+, "
        f"pyproject requires >={floor.group(1)}"
    )


# ---------------------------------------------------------------------------
# MCP tool surface
# ---------------------------------------------------------------------------
def _tools_in_code() -> set[str]:
    return set(re.findall(r'name="(widemem_[a-z_]+)"', read("widemem/mcp_server.py")))


def _tools_in_docs() -> set[str]:
    return set(re.findall(r"\|\s*`(widemem_[a-z_]+)`", read("docs/mcp.md")))


def test_every_mcp_tool_is_documented():
    missing = _tools_in_code() - _tools_in_docs()
    assert not missing, f"tools in mcp_server.py but absent from docs/mcp.md: {sorted(missing)}"


def test_every_documented_mcp_tool_exists():
    ghosts = _tools_in_docs() - _tools_in_code()
    assert not ghosts, f"tools documented in docs/mcp.md but absent from code: {sorted(ghosts)}"


# ---------------------------------------------------------------------------
# Scoring and decay claims
# ---------------------------------------------------------------------------
def test_readme_step_decay_matches_code():
    decay = read("widemem/scoring/decay.py")
    for age, score in [(7, "1.0"), (30, "0.7"), (90, "0.4")]:
        assert re.search(rf"age_days < {age}:\s*\n\s*return {score}", decay), (
            f"decay.py step tier changed: expected <{age} days -> {score}"
        )
    readme = read("README.md")
    assert "1.0 / 0.7 / 0.4 / 0.1 at 7/30/90 days" in readme, (
        "README step-decay row no longer matches decay.py tiers"
    )


def test_configuration_doc_defaults_match_types():
    types_src = read("widemem/core/types.py")
    config_doc = read("docs/configuration.md")
    for field in [
        "similarity_weight",
        "importance_weight",
        "recency_weight",
        "decay_rate",
        "hybrid_bm25_weight",
    ]:
        code = re.search(rf"^\s+{field}: float = ([0-9.]+)", types_src, re.M)
        assert code, f"{field} default not found in types.py"
        doc = re.search(rf"\|\s*`{field}`\s*\|\s*`float`\s*\|\s*`([0-9.]+)`", config_doc)
        assert doc, f"{field} row not found in docs/configuration.md"
        assert float(doc.group(1)) == float(code.group(1)), (
            f"docs/configuration.md says {field}={doc.group(1)}, "
            f"types.py says {code.group(1)}"
        )


def test_readme_formula_names_real_fields():
    types_src = read("widemem/core/types.py")
    readme = read("README.md")
    for field in ["similarity_weight", "importance_weight", "recency_weight"]:
        assert field in types_src, f"{field} gone from types.py; update the README formula"
        assert field in readme, f"README formula section no longer mentions {field}"


# ---------------------------------------------------------------------------
# Benchmark claim hygiene
# ---------------------------------------------------------------------------
def test_no_retired_superlatives():
    """Claims retracted in docs/HISTORY.md must not reappear.

    Superlatives about benchmark standing require a held-out run and a
    HISTORY.md-clean record; until then they are banned strings.
    """
    banned = ["best-in-class", "best in class", "state-of-the-art", "state of the art"]
    for rel in ["README.md", *[f"docs/{p.name}" for p in (ROOT / "docs").glob("*.md")],
                *[f"benchmark/{p.name}" for p in (ROOT / "benchmark").glob("*.md")]]:
        text = read(rel).lower()
        for phrase in banned:
            assert phrase not in text, f"retired claim {phrase!r} found in {rel}"


def test_corrections_log_exists_and_is_linked():
    assert (ROOT / "docs/HISTORY.md").exists(), "docs/HISTORY.md corrections log missing"
    assert "docs/HISTORY.md" in read("README.md"), "README no longer links docs/HISTORY.md"
