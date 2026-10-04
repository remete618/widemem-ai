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


def test_readme_mcp_tool_list_matches_code():
    line = re.search(r"^Tools exposed: (.+)$", read("README.md"), re.M)
    assert line, "README.md lost its 'Tools exposed:' line"
    listed = set(re.findall(r"`(widemem_[a-z_]+)`", line.group(1)))
    assert listed == _tools_in_code(), (
        f"README lists {sorted(listed)}, mcp_server.py has {sorted(_tools_in_code())}"
    )


_ENV_READ = re.compile(r'os\.(?:environ\.get|getenv)\(\s*"([A-Z][A-Z0-9_]+)"\s*(?:,\s*([^)]*?))?\s*\)')
_SDK_READS = {"ANTHROPIC_API_KEY", "OPENAI_BASE_URL"}  # read by the provider SDKs, not by widemem


def _env_reads(module: str) -> dict[str, str | None]:
    return {name: default.strip() if default else None for name, default in _ENV_READ.findall(read(module))}


def _literal(expr: str | None) -> str | None:
    if expr is None:
        return None
    m = re.fullmatch(r'"([^"]*)"', expr)
    assert m, f"env default {expr!r} is not a string literal; this test cannot compare it to the docs"
    return m.group(1)


def _user_docs() -> dict[str, str]:
    docs = {rel: read(rel) for rel in ("README.md", "YMYL.md")}
    docs.update(
        {f"docs/{p.name}": p.read_text(encoding="utf-8") for p in (ROOT / "docs").glob("*.md") if p.name != "HISTORY.md"}
    )
    return docs


def test_every_documented_env_var_is_read_somewhere():
    source = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "widemem").rglob("*.py"))
    for rel, text in _user_docs().items():
        for name in set(re.findall(r"`([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)`", text)):
            assert f'"{name}"' in source or name in _SDK_READS, f"{rel} documents {name}, which nothing reads"


def test_mcp_doc_env_defaults_match_the_server():
    mcp, rest = _env_reads("widemem/mcp_server.py"), _env_reads("widemem/server.py")
    rows = re.findall(r"^\|\s*`(WIDEMEM_[A-Z_]+)`\s*\|\s*([^|]+?)\s*\|", read("docs/mcp.md"), re.M)
    assert rows, "env var table missing from docs/mcp.md"
    for name, doc_default in rows:
        assert name in mcp or name in rest, f"docs/mcp.md documents {name}, which no server reads"
        if name not in mcp:
            continue
        code_default = _literal(mcp[name])
        if doc_default.startswith("`"):
            assert doc_default.strip("`") == code_default, (
                f"docs/mcp.md says {name} defaults to {doc_default}, mcp_server.py uses {code_default!r}"
            )
        else:
            assert not code_default, (
                f"docs/mcp.md says {name} is {doc_default}, mcp_server.py defaults it to {code_default!r}"
            )


def _clarification_fields() -> set[str]:
    body = read("widemem/retrieval/active.py").split("class Clarification", 1)[1]
    body = re.split(r"\n(?=\S)", body, maxsplit=1)[0]
    return set(re.findall(r"^    (\w+):", body, re.M))


def test_documented_clarification_fields_exist():
    fields = _clarification_fields()
    assert {"existing_content", "new_fact", "question"} <= fields, f"parsed Clarification fields: {sorted(fields)}"
    for rel in ("README.md", "examples/ymyl_active_retrieval.py"):
        text = read(rel)
        loop_vars = set(re.findall(r"for (\w+) in clarifications", text))
        assert loop_vars, f"{rel} no longer loops over clarifications; update this test"
        used = {attr for var in loop_vars for attr in re.findall(rf"\b{var}\.(\w+)", text)}
        assert used <= fields, f"{rel} reads Clarification.{sorted(used - fields)}, fields are {sorted(fields)}"


def test_no_doc_or_example_sets_the_unread_uncertainty_mode():
    if "config.uncertainty_mode" in "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "widemem").rglob("*.py")):
        return
    texts = dict(_user_docs())
    texts.update({f"examples/{p.name}": p.read_text(encoding="utf-8") for p in (ROOT / "examples").glob("*.py")})
    for rel, text in texts.items():
        assert not re.search(r"uncertainty_mode\s*=(?!=)", text), (
            f"{rel} sets uncertainty_mode, which nothing reads; show build_uncertainty_guidance() instead"
        )


def test_ymyl_doc_floors_match_the_extractor():
    doc, ext = read("YMYL.md"), read("widemem/extraction/llm_extractor.py")
    default = re.search(r"min_importance: float = ([\d.]+)", read("widemem/core/types.py"))
    assert default, "YMYLConfig.min_importance default not found"
    assert set(re.findall(r"importance = max\(importance, ([\d.]+)\)", doc)) == {default.group(1)}
    config_floors = ext.count("importance = max(importance, self.ymyl_config.min_importance)")
    assert config_floors == doc.count("→ importance = max(importance,"), (
        f"extractor applies the configured floor on {config_floors} paths; YMYL.md's diagram disagrees"
    )
    assert not re.search(r"importance = max\(importance, \d", ext), "extractor has a literal floor YMYL.md does not show"
    assert not re.search(r"(?:nudged|bump|boost)\w* to (?:importance )?\d", doc), "YMYL.md promises a floor by number"


def test_qdrant_url_claim_matches_the_store():
    store = read("widemem/storage/vector/qdrant_store.py")
    row = re.search(r"^\| `url` \|.*$", read("docs/configuration.md"), re.M)
    assert row, "url row missing from docs/configuration.md"
    reads_url = "config.url" in store
    assert ("Qdrant ignores" in row.group(0)) != reads_url, (
        "docs/configuration.md and qdrant_store.py disagree on whether Qdrant reads VectorStoreConfig.url"
    )


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
# Audit-trail claims
# ---------------------------------------------------------------------------
_AUDIT_SECTION = r"## History & Audit Trail\n(.*?)\n---"


def _audit_section() -> str:
    """The audit section as one whitespace-normalised line.

    Collapsing newlines keeps these guards from passing or failing on where
    a sentence happens to wrap.
    """
    match = re.search(_AUDIT_SECTION, read("README.md"), re.S)
    assert match, "README 'History & Audit Trail' section not found"
    return re.sub(r"\s+", " ", match.group(1))


def _logged_actions() -> set[str]:
    """MemoryAction members passed to a history log call in core/memory.py.

    Scoped to that module on purpose. `pipeline.py` names all four actions
    permanently, so a union across the package would pass no matter what the
    public methods do, which is the tautology this guard exists to avoid.
    The audit section describes the public API, and every write path it
    describes lives here.
    """
    calls = re.findall(
        r"_history_store\.log(?:_many)?\((?:[^()]|\([^()]*\))*\)",
        read("widemem/core/memory.py"),
        re.S,
    )
    return {m for call in calls for m in re.findall(r"MemoryAction\.([A-Z]+)", call)}


def test_readme_names_only_actions_the_public_api_logs():
    """Every write the audit section advertises must reach the history store.

    `delete()`, `pin()`, `import_json()` and `backfill_entities()` all wrote
    to the vector store with no entry while the README claimed otherwise.
    """
    section = _audit_section().lower()
    claimed = {
        word.upper()
        for word in ("add", "update", "delete")
        if re.search(rf"\b{word}s?\b", section)
    }
    assert claimed, "audit section no longer names any logged action"
    missing = claimed - _logged_actions()
    assert not missing, (
        f"README's audit section claims {sorted(missing)} is logged, but no "
        "history log call passes it. Log the write or drop the claim."
    )


def test_readme_discloses_that_entries_carry_no_actor():
    """While HistoryEntry has no actor field, the README must say so.

    The section read 'who changed this and when' against a schema with no
    column naming a caller. Banning the phrasing would only move it around,
    so the guard requires the limitation to be stated instead. When an actor
    field lands, the requirement lifts on its own and the stronger claim is
    allowed.
    """
    actor_fields = {"actor_id", "actor_type", "user_id", "agent_id", "run_id", "source_ref"}
    entry_block = re.search(
        r"class HistoryEntry\(BaseModel\):(.*?)\n\n", read("widemem/core/types.py"), re.S
    )
    assert entry_block, "HistoryEntry model not found in types.py"
    if any(f"{field}:" in entry_block.group(1) for field in actor_fields):
        return

    section = _audit_section().lower()
    assert "not attributed" in section, (
        "HistoryEntry carries no actor field, so the README audit section "
        "must state that entries are not attributed to a caller"
    )


def test_history_entry_fields_match_the_api_doc():
    entry_block = re.search(r"class HistoryEntry\(BaseModel\):(.*?)\n\n", read("widemem/core/types.py"), re.S)
    assert entry_block, "HistoryEntry model not found in types.py"
    code_fields = set(re.findall(r"^\s+([a-z_]+):", entry_block.group(1), re.M))

    doc = re.search(r"## HistoryEntry\n(.*?)\n## ", read("docs/api.md"), re.S)
    assert doc, "HistoryEntry table missing from docs/api.md"
    doc_fields = set(re.findall(r"\|\s*`([a-z_]+)`\s*\|", doc.group(1)))

    assert code_fields == doc_fields, (
        f"HistoryEntry fields drifted: only in code {sorted(code_fields - doc_fields)}, "
        f"only in docs/api.md {sorted(doc_fields - code_fields)}"
    )


def test_security_policy_covers_the_shipped_minor():
    version = re.search(r'^version = "(\d+)\.(\d+)\.', read("pyproject.toml"), re.M)
    assert version, "version string missing from pyproject.toml"
    shipped = f"{version.group(1)}.{version.group(2)}.x"
    assert re.search(rf"\|\s*{re.escape(shipped)}\s*\|\s*Yes\s*\|", read("SECURITY.md")), (
        f"SECURITY.md does not list {shipped} as supported"
    )


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
