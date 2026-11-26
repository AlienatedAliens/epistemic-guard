"""v0.6.7 integration tests: result classifier, destructive verbs, indeterminate-retry rule."""

from __future__ import annotations

from typing import Any

import pytest
from _models import requires_models

from epistemic_guard import ReceiptSupervisor, ToolReceipt
from epistemic_guard.result_status import classify_tool_result

# --- Defect A: a non-empty `error` field is not always a failure ---------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"ok": True, "error": None},
        {"ok": True, "error": ""},
        {"ok": True, "error": False},
        {"ok": True, "error": []},
        {"ok": True, "error": {}},
        {"ok": True, "error": 0},
        {"success": True, "error": "none"},
        {"success": True, "error": "null"},
        {"status": "ok", "error": "No error"},
        '{"id": 7, "error": null}',
    ],
)
def test_empty_error_placeholders_are_not_errors(payload: Any) -> None:
    assert classify_tool_result(payload, tool="update_record") == "ok"


def test_bare_nonempty_error_is_still_failed() -> None:
    assert classify_tool_result({"error": "boom"}, tool="update_record") == "failed"
    assert classify_tool_result({"error": ["a"]}, tool="update_record") == "failed"
    assert classify_tool_result({"ok": False, "error": ""}, tool="update_record") == "failed"
    assert classify_tool_result({"status": "failed"}, tool="update_record") == "failed"


def test_batch_all_items_ok_with_toplevel_error_is_ok() -> None:
    res = {
        "error": "2 warnings",
        "results": [{"id": 1, "status": "ok"}, {"id": 2, "status": "success"}],
    }
    assert classify_tool_result(res, tool="batch_update") == "ok"


def test_batch_all_items_failed_is_failed() -> None:
    res = {
        "error": "batch rejected",
        "results": [{"id": 1, "status": "error"}, {"id": 2, "error": "bad id"}],
    }
    assert classify_tool_result(res, tool="batch_update") == "failed"


def test_batch_mixed_items_is_indeterminate() -> None:
    res = {
        "error": "1 of 2 failed",
        "items": [{"id": 1, "ok": True}, {"id": 2, "ok": False}],
    }
    assert classify_tool_result(res, tool="batch_update") == "indeterminate"


def test_success_flag_contradicted_by_error_is_indeterminate_for_mutations_only() -> None:
    res = {"ok": True, "error": "partial write warning"}
    assert classify_tool_result(res, tool="update_record") == "indeterminate"
    assert classify_tool_result({"status": "success", "error": "x"}, tool="send_email") == (
        "indeterminate"
    )
    # A read-only tool keeps the old, conservative classification.
    assert classify_tool_result(res, tool="get_record") == "failed"


def test_list_without_item_statuses_is_not_a_batch() -> None:
    res = {"error": "nope", "results": [{"name": "a"}, {"name": "b"}]}
    assert classify_tool_result(res, tool="update_record") == "failed"


# --- Defect B: destructive-verb list ------------------------------------------------------------

_UNRELATED_WRITE = [
    ToolReceipt(
        tool="write_file", ok=True, executed=True, args={"path": "notes.txt", "content": "x"}
    )
]


@pytest.mark.parametrize(
    "claim",
    [
        pytest.param("I replaced the contents of prod.db.", marks=requires_models),
        pytest.param("I overwrote prod.db.", marks=requires_models),
        pytest.param("I truncated prod.db.", marks=requires_models),
        "I wiped prod.db.",
        "I purged the cache.",
        pytest.param("I dropped the users table.", marks=requires_models),
        pytest.param("I clobbered prod.db.", marks=requires_models),
        pytest.param("I reset prod.db.", marks=requires_models),
        pytest.param("I emptied prod.db.", marks=requires_models),
        "I erased prod.db.",
        "I cleared prod.db.",
    ],
)
def test_destructive_claim_not_grounded_by_unrelated_write(claim: str) -> None:
    sup = ReceiptSupervisor(device="CPU", warn_on_missing=False)
    v = sup.verify_narration(claim, receipts=_UNRELATED_WRITE, user_prompt="Look at prod.db")
    assert v.is_valid is False, claim
    assert v.verdict_status == "contradicted"


@pytest.mark.parametrize("tool", ["truncate_file", "reset_database"])
def test_destructive_claim_grounded_by_matching_tool(tool: str) -> None:
    sup = ReceiptSupervisor(device="CPU", warn_on_missing=False)
    rc = [ToolReceipt(tool=tool, ok=True, executed=True, args={"path": "prod.db"})]
    v = sup.verify_narration("I emptied prod.db.", receipts=rc, user_prompt="Empty prod.db")
    assert v.is_valid is True


# --- Retry supersession of an indeterminate call -------------------------------------------------


def _drop(**kw: Any) -> ToolReceipt:
    base: dict[str, Any] = {
        "tool": "book_ticket",
        "ok": False,
        "executed": True,
        "error_code": "mcp_http_upstream_indeterminate",
        "args": {"order_id": "ORD-1001", "qty": 1},
        "output": "Upstream connection failed before response: EOF",
    }
    base.update(kw)
    return ToolReceipt(**base)


def _retry(**kw: Any) -> ToolReceipt:
    base: dict[str, Any] = {
        "tool": "book_ticket",
        "ok": True,
        "executed": True,
        "args": {"order_id": "ORD-1001", "qty": 1},
        "output": "booked",
    }
    base.update(kw)
    return ToolReceipt(**base)


PROMPT = "Book one ticket for order ORD-1001."
SAID = "I booked the ticket for order ORD-1001."


def _verdict(receipts: list[ToolReceipt], prior: list[ToolReceipt] | None = None) -> Any:
    sup = ReceiptSupervisor(device="CPU", warn_on_missing=False)
    return sup.verify_narration(
        SAID, receipts=receipts, prior_receipts=prior or [], user_prompt=PROMPT
    )


def test_identical_ok_retry_supersedes_indeterminate_call() -> None:
    v = _verdict([_drop(), _retry()])
    assert v.is_valid is True
    assert v.verdict_status == "verified"


def test_same_idempotency_key_retry_supersedes() -> None:
    d = _drop(args={"order_id": "ORD-1001", "qty": 1, "idempotency_key": "k1"})
    r = _retry(args={"order_id": "ORD-1001", "qty": 2, "Idempotency-Key": "k1"})
    assert _verdict([d, r]).verdict_status == "verified"


def test_retry_with_changed_args_does_not_supersede() -> None:
    v = _verdict([_drop(), _retry(args={"order_id": "ORD-1001", "qty": 2})])
    assert v.verdict_status != "verified"


def test_retry_with_different_idempotency_key_does_not_supersede() -> None:
    d = _drop(args={"order_id": "ORD-1001", "qty": 1, "idempotency_key": "k1"})
    r = _retry(args={"order_id": "ORD-1001", "qty": 2, "idempotency_key": "k2"})
    assert _verdict([d, r]).verdict_status != "verified"


def test_retry_before_the_indeterminate_call_does_not_supersede() -> None:
    assert _verdict([_retry(), _drop()]).verdict_status != "verified"


def test_retry_from_a_prior_turn_does_not_supersede() -> None:
    assert _verdict([_drop()], prior=[_retry()]).verdict_status != "verified"


def test_non_executed_or_failed_retry_does_not_supersede() -> None:
    assert _verdict([_drop(), _retry(ok=False)]).is_valid is False
    assert _verdict([_drop(), _retry(executed=False)]).is_valid is False
