"""The vendored model code and the weights are MIT, and the attribution says so.

Vendoring upstream code into an Apache-2.0 repository is only fine while its licence
travels with it and NOTICE names it. Both are easy to lose in a refactor, so they
are tested rather than trusted.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDORED = ROOT / "src" / "medmcp_cardiac" / "_cinema"
NOTICE = ROOT / "NOTICE"
README = ROOT / "README.md"

# The six modules scripts/vendor-cinema.sh writes, plus the hand-maintained pair.
_VENDORED_MODULES = {"conv", "convvit", "vit", "rotary", "log", "transform", "convunetr"}


def test_vendored_license_travels_with_the_code() -> None:
    """Upstream's MIT licence text sits beside the copied files."""
    text = (VENDORED / "LICENSE").read_text()
    assert "MIT License" in text
    assert "Permission is hereby granted, free of charge" in text


def test_upstream_provenance_is_recorded() -> None:
    """UPSTREAM pins the exact commit the files came from, for the next re-vendoring."""
    text = (VENDORED / "UPSTREAM").read_text()
    assert "https://github.com/mathpluscode/CineMA" in text
    rev = re.search(r"^rev: ([0-9a-f]{40})$", text, re.MULTILINE)
    assert rev, "UPSTREAM must record a full 40-character commit hash"
    listed = set(re.findall(r"-> (\w+)\.py$", text, re.MULTILINE))
    assert listed == _VENDORED_MODULES


def test_every_vendored_module_is_present_and_rewritten() -> None:
    """Each listed module exists and reaches for nothing under the upstream names."""
    for name in _VENDORED_MODULES:
        path = VENDORED / f"{name}.py"
        assert path.is_file(), path
        source = path.read_text()
        assert not re.search(r"^(from|import) (cinema|timm)\b", source, re.MULTILINE), (
            f"{name}.py still imports an upstream package name"
        )


def test_notice_attributes_cinema_and_timm() -> None:
    """NOTICE names both MIT upstreams: the model code+weights and the timm layers."""
    text = NOTICE.read_text()
    assert "CineMA" in text and "MIT" in text
    assert "mathpluscode/CineMA" in text
    assert "timm" in text and "Apache" in text
    assert "doi:10.1038/s43856-026-01636-0" in text or "s43856-026-01636-0" in text


def test_readme_citation_matches_notice() -> None:
    """The README's citation and NOTICE must point at the same paper."""
    for text in (README.read_text(), NOTICE.read_text()):
        assert "Communications Medicine" in text
        assert "s43856-026-01636-0" in text
