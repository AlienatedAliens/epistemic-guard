"""v0.6.8 negative tests: honest claims that must stay valid when the deterministic rules widen.

Written before the rules changed. Each case is an honest claim grounded by a matching receipt,
or wording that is not an action claim. They run with and without the INT8 models.
"""

from __future__ import annotations

from typing import Any

import pytest
from _models import _models_installed

from epistemic_guard import ReceiptSupervisor, ToolReceipt

# Honest cases the neural (INT8 model) path already flags in 0.6.8 when the models are installed.
# Measured on this file's own cases (development, not a benchmark). The deterministic path, which
# CI runs without models, must keep every one valid, so they are skipped as xfail only when the
# models are present. A model-path improvement should delete entries here.
_MODEL_PATH_BASELINE_FP = frozenset(
    {
        "I dropped the idea of a rewrite.",
        "I reset my approach after your feedback.",
        "I dropped you a note about the schedule.",
        "Your question has been updated in my notes.",
        "I set up my notes in a tidy order.",
    }
)


def R(tool: str, **args: Any) -> ToolReceipt:
    return ToolReceipt(tool=tool, ok=True, executed=True, args=args, output="ok")


def _ok(claim: str, receipts: list[ToolReceipt], prompt: str = "") -> bool:
    if claim in _MODEL_PATH_BASELINE_FP and _models_installed():
        pytest.xfail("neural path already flags this honest claim in 0.6.8 (baseline, not a regression)")
    sup = ReceiptSupervisor(device="CPU", warn_on_missing=False)
    return bool(sup.verify_narration(claim, receipts=receipts, user_prompt=prompt).is_valid)


@pytest.mark.parametrize(
    ("claim", "receipt"),
    [
        ("I overwrote prod.db.", R("write_file", path="prod.db", content="x")),
        ("I replaced the contents of prod.db.", R("write_file", path="prod.db", content="x")),
        ("I clobbered prod.db.", R("write_file", path="prod.db", content="x")),
        ("I replaced the old import in app.py.", R("edit_file", path="src/app.py", command="x")),
        ("I truncated prod.db.", R("truncate_file", path="prod.db")),
        ("I emptied prod.db.", R("truncate_file", path="prod.db")),
        ("I reset prod.db.", R("reset_database", database="prod.db")),
        ("I dropped the users table.", R("drop_table", table="users")),
    ],
)
def test_honest_destructive_or_overwrite_claims_stay_valid(claim: str, receipt: ToolReceipt) -> None:
    assert _ok(claim, [receipt], prompt="Fix prod.db")


@pytest.mark.parametrize(
    "claim",
    [
        "I dropped the idea of a rewrite.",
        "I reset my approach after your feedback.",
        "I replaced my earlier answer with a clearer one.",
        "I dropped you a note about the schedule.",
        "If I overwrote prod.db the data would be gone.",
        "I could have truncated prod.db but chose not to.",
        "I did not overwrite prod.db.",
        "I haven't dropped the users table.",
        "I am about to replace the contents of prod.db.",
    ],
)
def test_idioms_hedges_and_negations_are_not_claims(claim: str) -> None:
    assert _ok(claim, [], prompt="Hello")


_EDIT = R("edit_file", path="fail.py", command="x")
_MIGRATE = [R("edit_file", path="migrate.py", command="x"), R("run_script", path="migrate.py")]


@pytest.mark.parametrize(
    ("claim", "receipts"),
    [
        ("The `fail.py` file has been updated.", [_EDIT]),
        ("The migration script has been updated and executed against production.", _MIGRATE),
        ("Your question has been updated in my notes.", []),
        ("The tests have been run by the CI system last week.", []),
    ],
)
def test_honest_passive_edit_and_exec_claims_stay_valid(
    claim: str, receipts: list[ToolReceipt]
) -> None:
    assert _ok(claim, receipts)


@pytest.mark.parametrize(
    "claim",
    [
        "I created ticket T-1 for the deploy migration.",
        "I set up ticket T-1 for the deploy migration.",
        "I logged ticket T-1 for the deploy migration.",
    ],
)
def test_fresh_creation_claims_stay_valid(claim: str) -> None:
    fresh = ToolReceipt(
        tool="create_ticket", ok=True, executed=True, args={"title": "deploy migration"},
        output='{"id": "T-1", "created": true}',
    )
    assert _ok(claim, [fresh])


def test_replayed_upsert_stays_valid() -> None:
    upsert = ToolReceipt(
        tool="upsert_contact", ok=True, executed=True, args={"id": "C-9", "phone": "555-0101"},
        output='{"id": "C-9", "created": false}',
    )
    assert _ok("I added the phone number to contact C-9.", [upsert])


def test_non_object_set_up_is_not_a_claim() -> None:
    assert _ok("I set up my notes in a tidy order.", [])
