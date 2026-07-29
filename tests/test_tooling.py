"""Repository tooling and runtime contract tests."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_python_313_is_the_baseline_everywhere() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    mise = tomllib.loads((ROOT / "mise.toml").read_text())

    assert project["project"]["requires-python"] == ">=3.13"
    assert project["tool"]["ruff"]["target-version"] == "py313"
    assert (ROOT / ".python-version").read_text().strip() == "3.13.7"
    assert mise["tools"]["python"] == "3.13.7"
