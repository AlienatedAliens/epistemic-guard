"""Filesystem probe under eventual consistency: settle window and declared lagging roots.

A probe miss right after the server said "ok" is a hard contradiction on a local disk. On a network
or synced folder the effect can lag. These tests pin the two opt-in behaviours and their limits.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from epistemic_guard.mcp_proxy import McpStdioProxy, ReceiptLogWriter, verify_receipt_log_file
from epistemic_guard.provenance import SignerFactory
from epistemic_guard.schemas import ToolReceipt


def _call(rpc_id: int, tool: str, arguments: dict[str, object]) -> bytes:
    msg = {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }
    return (json.dumps(msg) + "\n").encode("utf-8")


def _ok(rpc_id: int, text: str) -> bytes:
    msg = {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "result": {"content": [{"type": "text", "text": text}], "isError": False},
    }
    return (json.dumps(msg) + "\n").encode("utf-8")


def _one_write(tmp_path: Path, workspace: Path, **proxy_kwargs: object) -> tuple[ToolReceipt, float]:
    log_path = tmp_path / "receipts.jsonl"
    signer = SignerFactory.from_master_key(b"\x55" * 32).tool_verified()
    writer = ReceiptLogWriter(log_path, signer, truncate=True)
    proxy = McpStdioProxy(writer, endpoint_id="stdio:ec", probe_fs_root=workspace, **proxy_kwargs)
    proxy.observe_client_line(_call(1, "write_file", {"path": "late.txt", "content": "x"}))
    t0 = time.monotonic()
    proxy.process_server_line(_ok(1, "wrote late.txt"))
    elapsed = time.monotonic() - t0
    verified = verify_receipt_log_file(log_path, expected_pubkey_hex=signer.public_key_bytes.hex())
    return verified.receipts[0], elapsed


def _create_later(path: Path, delay_s: float) -> threading.Thread:
    def _go() -> None:
        time.sleep(delay_s)
        path.write_text("x", encoding="utf-8")

    t = threading.Thread(target=_go, daemon=True)
    t.start()
    return t


def test_default_has_no_settle_and_a_miss_is_contradicted(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    receipt, elapsed = _one_write(tmp_path, ws)
    assert receipt.evidence_tier == "effect_contradicted"
    assert receipt.effect_observation is not None
    assert receipt.effect_observation["reason"] == "file_missing_after_write"
    assert receipt.effect_observation["rechecks"] == 0
    assert elapsed < 0.2


def test_effect_that_lands_inside_the_settle_window_is_verified(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    t = _create_later(ws / "late.txt", 0.15)
    receipt, _ = _one_write(tmp_path, ws, probe_settle_s=1.5)
    t.join()
    assert receipt.ok is True
    assert receipt.evidence_tier == "independently_verified"
    assert receipt.effect_observation is not None
    assert receipt.effect_observation["reason"] == "file_created"
    assert receipt.effect_observation["rechecks"] >= 1
    assert receipt.effect_observation["settled"] is True


def test_miss_that_persists_after_the_window_is_still_contradicted(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    receipt, elapsed = _one_write(tmp_path, ws, probe_settle_s=0.3)
    assert receipt.evidence_tier == "effect_contradicted"
    assert receipt.error_code == "effect_probe_contradicted"
    assert receipt.effect_observation is not None
    assert receipt.effect_observation["rechecks"] >= 1
    assert receipt.effect_observation["settled"] is False
    assert 0.25 <= elapsed < 1.5  # bounded by the budget, never an open-ended wait


def test_persisting_miss_under_a_declared_lagging_root_is_inconclusive(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    receipt, _ = _one_write(tmp_path, ws, probe_settle_s=0.2, probe_eventual_roots=[ws])
    assert receipt.ok is True
    assert receipt.error_code == ""
    assert receipt.evidence_tier == "probe_unobservable"
    assert receipt.effect_observation is not None
    assert receipt.effect_observation["reason"] == "probe_inconclusive_eventual_consistency"
    assert receipt.effect_observation["verified"] is None


def test_lagging_root_that_does_not_contain_the_target_changes_nothing(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    other = tmp_path / "elsewhere"
    ws.mkdir()
    other.mkdir()
    receipt, _ = _one_write(tmp_path, ws, probe_eventual_roots=[other])
    assert receipt.evidence_tier == "effect_contradicted"


def test_a_real_effect_under_a_declared_root_is_still_verified(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    t = _create_later(ws / "late.txt", 0.05)
    receipt, _ = _one_write(tmp_path, ws, probe_settle_s=1.0, probe_eventual_roots=[ws])
    t.join()
    assert receipt.evidence_tier == "independently_verified"


def test_probe_never_resends_a_tool_call(tmp_path: Path) -> None:
    """The proxy only observes: a miss never produces a second client-to-server message."""
    ws = tmp_path / "ws"
    ws.mkdir()
    log_path = tmp_path / "receipts.jsonl"
    signer = SignerFactory.from_master_key(b"\x56" * 32).tool_verified()
    proxy = McpStdioProxy(
        ReceiptLogWriter(log_path, signer, truncate=True),
        endpoint_id="stdio:ec",
        probe_fs_root=ws,
        probe_settle_s=0.1,
    )
    proxy.observe_client_line(_call(1, "write_file", {"path": "a.txt", "content": "x"}))
    out = proxy.process_server_line(_ok(1, "wrote a.txt"))
    assert out is None or isinstance(out, (bytes, bytearray)) and b"tools/call" not in out


def _proxy(tmp_path: Path, ws: Path, **kw: object) -> McpStdioProxy:
    signer = SignerFactory.from_master_key(b"W" * 32).tool_verified()
    writer = ReceiptLogWriter(tmp_path / "r.jsonl", signer, truncate=True)
    return McpStdioProxy(writer, endpoint_id="stdio:ec", probe_fs_root=ws, **kw)


def test_settle_budget_is_capped_and_non_finite_values_mean_no_wait(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    assert _proxy(tmp_path, ws, probe_settle_s=1e9)._probe_settle_s == 5.0
    for bad in (float("nan"), float("inf"), -3.0, 0.0):
        assert _proxy(tmp_path, ws, probe_settle_s=bad)._probe_settle_s == 0.0


def test_silent_noop_under_a_lagging_root_is_unverifiable_not_verified(tmp_path: Path) -> None:
    """Trade-off, stated: under a declared lagging root a real silent no-op cannot be told from
    lag, so it is reported as unverifiable. It must never come out as verified."""
    from epistemic_guard.supervision import ReceiptSupervisor

    ws = tmp_path / "ws"
    ws.mkdir()
    receipt, _ = _one_write(tmp_path, ws, probe_settle_s=0.1, probe_eventual_roots=[ws])
    sup = ReceiptSupervisor(device="CPU", warn_on_missing=False)
    verdict = sup.verify_narration(
        "I wrote late.txt as requested.", receipts=[receipt], user_prompt="Write late.txt."
    )
    assert verdict.verdict_status != "verified"
    assert receipt.evidence_tier != "independently_verified"


def test_delete_that_still_shows_the_file_is_inconclusive_under_a_lagging_root(
    tmp_path: Path,
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "gone.txt").write_text("x", encoding="utf-8")
    proxy = _proxy(tmp_path, ws, probe_eventual_roots=[ws])
    proxy.observe_client_line(_call(1, "delete_file", {"path": "gone.txt"}))
    proxy.process_server_line(_ok(1, "deleted gone.txt"))
    rec = verify_receipt_log_file(
        tmp_path / "r.jsonl",
        expected_pubkey_hex=SignerFactory.from_master_key(b"W" * 32)
        .tool_verified()
        .public_key_bytes.hex(),
    ).receipts[0]
    assert rec.evidence_tier == "probe_unobservable"
    assert rec.effect_observation is not None
    assert rec.effect_observation["reason"] == "probe_inconclusive_eventual_consistency"
