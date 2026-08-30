"""Packaging regression tests.

The image and the [all] extra both declared providers they never
installed. Both defects are static and cheap to pin here.
"""

from __future__ import annotations

import re
from pathlib import Path


class TestPackagingDefaults:
    """Dockerfile:8 and pyproject [all] - declared defaults must be installable."""

    @staticmethod
    def _root():
        return Path(__file__).resolve().parent.parent

    @classmethod
    def _extra(cls, name):
        """Read one optional-dependencies list without tomllib (3.11+ only)."""
        text = (cls._root() / "pyproject.toml").read_text()
        match = re.search(rf"^{name} = \[(.*?)^\]", text, re.S | re.M)
        assert match, f"no {name!r} extra found in pyproject.toml"
        return {
            re.split(r"[><=\[]", line.strip().strip('",'))[0].strip()
            for line in match.group(1).splitlines()
            if line.strip().startswith('"')
        }

    def test_all_extra_can_run_the_server(self):
        all_names = self._extra("all")
        server_names = self._extra("server")

        missing = server_names - all_names
        assert not missing, (
            f"pip install widemem-ai[all] cannot start the server; missing {sorted(missing)}"
        )

    def test_dockerfile_installs_the_providers_its_env_selects(self):
        text = (self._root() / "Dockerfile").read_text()

        install_line = next(line for line in text.splitlines() if "pip install" in line)
        selected = {
            line.split("=", 1)[1].strip()
            for line in text.splitlines()
            if line.startswith(("ENV WIDEMEM_LLM_PROVIDER", "ENV WIDEMEM_EMBEDDING_PROVIDER"))
        }

        for provider in selected:
            assert provider in install_line, (
                f"Dockerfile defaults to the {provider!r} provider but never installs it; "
                "the container cannot serve a single request"
            )

    def test_dockerfile_does_not_store_state_in_tmp(self):
        text = (self._root() / "Dockerfile").read_text()
        data_path = next(
            line.split("=", 1)[1].strip()
            for line in text.splitlines()
            if line.startswith("ENV WIDEMEM_DATA_PATH")
        )
        assert not data_path.startswith("/tmp"), (
            "memory state under /tmp is wiped on restart on most container runtimes"
        )

