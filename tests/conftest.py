"""Tests train into a temporary folder (AI_A2_ROOT) so real artifacts are never touched."""
import os
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
TMP = tempfile.mkdtemp(prefix="ai_a2_test_")
os.environ["AI_A2_ROOT"] = TMP

import pytest  # noqa: E402

VALID = {"rainfall_mm": 1000, "avg_temp_c": 18, "soil_moisture_pct": 55,
         "fertilizer_kg": 120, "field_size_ha": 1.5, "season": "A", "cooperative_support": 1}


@pytest.fixture(scope="session")
def trained():
    import run_all
    assert run_all.main(["--student-id", "ug0001234@ines.ac.rw", "--seed", "1234",
                         "--reg-digits", "TEST"]) == 0
    return Path(TMP)


@pytest.fixture()
def valid():
    return dict(VALID)
