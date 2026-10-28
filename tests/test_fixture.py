import subprocess
import sys
from pathlib import Path

import pytest

MAKE = Path(__file__).resolve().parent.parent / "fixtures" / "make_vault.py"


@pytest.mark.parametrize("n", [30, 100])
def test_fixture_generator_self_check(n):
    r = subprocess.run([sys.executable, "-B", str(MAKE), "--self-check", "--notes", str(n)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]


def test_shipped_config_matches_the_defaults_file():
    """config.toml is what people edit; defaults.toml is the fallback for any key they delete.
    They must say the same thing."""
    root = Path(__file__).resolve().parent.parent
    assert (root / "config.toml").read_text() == (root / "rag" / "defaults.toml").read_text()
