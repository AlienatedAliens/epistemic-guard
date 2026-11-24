"""v0.6.8 tests: the result classifier matches unknown-outcome shapes on value, not field name."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from epistemic_guard import (
    McpStdioProxy,
    McpStreamableHttpProxy,
    ReceiptSupervisor,
    extract_receipts_from_messages,
)
from epistemic_guard.result_status import classify_tool_result

WRITE = "update_record"


def _text(obj: object) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(obj)}]}


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "submission_uncertain"},
        {"outcome": "outcome_unknown"},
        {"state": "maybe_applied"},
        {"result_status": "Unknown"},
        {"status": "unknown", "id": 7},
        {"ok": False, "outcome": "unknown"},
        {"error": "rent_outcome_unknown"},
        {"error_code": "ads_boost_unknown", "ok": False},
        {"code": "payment_uncertain", "error": "payment_failed"},
        {"error": {"code": "rent_outcome_unknown", "message": "timed out"}},
        {"side_effect_status": "unknown"},
        {"side_effect_status": "unknown", "retry_safe": False},
        {"success": False, "error": "ads_boost_unknown", "status": "ads_boost_failed"},
    ],
)
def test_unknown_outcome_shapes_are_indeterminate(payload: dict[str, Any]) -> None:
    assert classify_tool_result(payload, tool=WRITE) == "indeterminate"
    assert classify_tool_result(json.dumps(payload), tool=WRITE) == "indeterminate"


@pytest.mark.parametrize(
    "payload",
    [
        {"structuredContent": {"status": "submission_uncertain"}, "content": []},
        _text({"outcome": "outcome_unknown"}),
        _text({"error": "rent_outcome_unknown"}),
        _text({"side_effect_status": "unknown"}),
        _text({"result": {"status": "submission_uncertain"}}),
        {"data": {"error_code": "ads_boost_unknown"}},
    ],
)
def test_unknown_shapes_inside_mcp_envelopes(payload: dict[str, Any]) -> None:
    assert classify_tool_result(payload, tool=WRITE) == "indeterminate"
    assert classify_tool_result(json.dumps(payload), tool=WRITE) == "indeterminate"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"error": "rent_failed"}, "failed"),
        ({"error_code": "ads_boost_failed", "ok": False}, "failed"),
        ({"code": "not_found", "error": "missing"}, "failed"),
        ({"status": "failed"}, "failed"),
        ({"side_effect_status": "none", "error": "denied"}, "failed"),
        ({"status": "ok", "retry_safe": False}, "ok"),
        ({"retry_safe": False}, "ok"),
        ({"side_effect_status": "applied"}, "ok"),
        ({"side_effect_status": "none", "status": "success"}, "ok"),
        ({"status": "submitted", "id": 3}, "ok"),
        ({"code": "known_good", "ok": True}, "ok"),
        (_text({"status": "ok", "error": None}), "ok"),
    ],
)
def test_honest_failed_and_ok_stay_put(payload: dict[str, Any], expected: str) -> None:
    assert classify_tool_result(payload, tool=WRITE) == expected


@pytest.mark.parametrize(
    "payload",
    [
        {"order_id": 7, "status": "unknown"},
        {"outcome": "outcome_unknown"},
        {"state": "uncertain"},
        {"side_effect_status": "unknown"},
        _text({"order_id": 7, "status": "submission_uncertain"}),
    ],
)
def test_read_only_tool_is_exempt_from_unknown_values(payload: dict[str, Any]) -> None:
    assert classify_tool_result(payload, tool="get_order") == "ok"


def test_read_only_tool_with_unknown_error_code_is_not_indeterminate() -> None:
    assert classify_tool_result({"error": "lookup_outcome_unknown"}, tool="get_order") == "failed"


def test_new_shapes_end_to_end_across_proxies_and_extractor(tmp_path: Path) -> None:
    from epistemic_guard.mcp_proxy import ReceiptLogWriter, load_or_create_mcp_signer

    shapes: dict[str, dict[str, Any]] = {
        "submit_order": _text({"status": "submission_uncertain"}),
        "rent_vehicle": {"content": [], "structuredContent": {"error": "rent_outcome_unknown"}},
        "boost_ads": _text({"side_effect_status": "unknown", "retry_safe": False}),
    }
    signer = load_or_create_mcp_signer(key_file=tmp_path / "signer.key")
    stdio = McpStdioProxy(ReceiptLogWriter(tmp_path / "stdio.jsonl", signer))
    http = McpStreamableHttpProxy("http://127.0.0.1:9/mcp", ReceiptLogWriter(tmp_path / "h.jsonl", signer))
    for i, (tool, result) in enumerate(shapes.items(), start=1):
        call = {"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": tool, "arguments": {}}}
        stdio.process_client_line((json.dumps(call) + "\n").encode("utf-8"))
        stdio.process_server_line((json.dumps({"jsonrpc": "2.0", "id": i, "result": result}) + "\n").encode("utf-8"))
        h = http._build_receipt(f"h-{i}", tool, {}, {"jsonrpc": "2.0", "id": i, "result": result})
        assert (h.ok, h.error_code) == (False, "mcp_tool_result_indeterminate"), tool

    msgs: list[dict[str, Any]] = [
        {"role": "assistant", "tool_calls": [
            {"id": f"c{i}", "function": {"name": t, "arguments": "{}"}} for i, t in enumerate(shapes)
        ]},
        *[
            {"role": "tool", "tool_call_id": f"c{i}", "content": json.dumps(r)}
            for i, r in enumerate(shapes.values())
        ],
    ]
    receipts = extract_receipts_from_messages(msgs)
    assert [(r.ok, r.error_code) for r in receipts] == [(False, "tool_result_indeterminate")] * 3

    sup = ReceiptSupervisor(device="CPU", warn_on_missing=False)
    rent = [r for r in receipts if r.tool == "rent_vehicle"]
    v_failed = sup.verify_narration("The vehicle rental failed.", receipts=rent)
    assert v_failed.is_valid is False
    assert v_failed.unsupported_spans[0].claimed_family == "INDETERMINATE_OUTCOME_MISSTATED"
    v_honest = sup.verify_narration(
        "The rental outcome is unknown, so I cannot say whether it went through; I'll check.",
        receipts=rent,
    )
    assert v_honest.is_valid is True
