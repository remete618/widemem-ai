"""Packaging regression tests.

The image and the [all] extra both declared providers they never
installed. Both defects are static and cheap to pin here.
"""

from __future__ import annotations

from pathlib import Path

import tomllib


class TestPackagingDefaults:
    """Dockerfile:8 and pyproject [all] - declared defaults must be installable."""

    @staticmethod
    def _root():
        return Path(__file__).resolve().parent.parent

    def test_all_extra_can_run_the_server(self):
        with open(self._root() / "pyproject.toml", "rb") as fh:
            data = tomllib.load(fh)
        extras = data["project"]["optional-dependencies"]
        all_names = {dep.split(">")[0].split("[")[0].split("=")[0].strip() for dep in extras["all"]}
        server_names = {dep.split(">")[0].split("[")[0].split("=")[0].strip() for dep in extras["server"]}

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

