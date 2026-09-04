"""Safety and completeness checks for the judge-facing policy table."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

from intervention_map import INTERVENTION_MAP, recommend_action  # noqa: E402

RETRY_ACTIONS = {"RETRY_IMMEDIATE", "RETRY_DELAYED_24H", "RETRY_DELAYED_72H"}


def test_every_entry_has_nonempty_justification():
    assert INTERVENTION_MAP, "INTERVENTION_MAP is empty"
    for category, payload in INTERVENTION_MAP.items():
        justification = payload.get("justification")
        assert isinstance(justification, str), f"{category} justification is not a string"
        assert justification.strip(), f"{category} has an empty justification"


def test_risk_blocked_maps_to_escalate_human():
    assert recommend_action("RISK_BLOCKED") == "ESCALATE_HUMAN"
    assert INTERVENTION_MAP["RISK_BLOCKED"]["action"] == "ESCALATE_HUMAN"


def test_permanent_failure_does_not_map_to_retry():
    action = recommend_action("PERMANENT_FAILURE")
    assert action not in RETRY_ACTIONS, (
        f"PERMANENT_FAILURE mapped to {action}; must not auto-retry"
    )
