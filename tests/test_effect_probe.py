"""Adversarial tests for out-of-process `FilesystemEffectProbe` (`--probe-fs`)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from epistemic_guard.exceptions import MerkleTamperError, SignatureTamperError
from epistemic_guard.mcp_proxy import McpStdioProxy, ReceiptLogWriter, verify_receipt_log_file
from epistemic_guard.provenance import SignerFactory
from epistemic_guard.supervision import ReceiptSupervisor


def _rpc_call(rpc_id: int, tool: str, arguments: dict[str, object]) -> bytes:
    msg = {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }
    return (json.dumps(msg) + "\n").encode("utf-8")


def _rpc_ok(rpc_id: int, text: str) -> bytes:
    msg = {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "result": {"content": [{"type": "text", "text": text}], "isError": False},
    }
    return (json.dumps(msg) + "\n").encode("utf-8")


def test_filesystem_probe_verifies_real_write_and_delete(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    log_path = tmp_path / "receipts.jsonl"
    signer = SignerFactory.from_master_key(b"\x33" * 32).tool_verified()
    writer = ReceiptLogWriter(log_path, signer, truncate=True)
    proxy = McpStdioProxy(writer, endpoint_id="stdio:fs", probe_fs_root=workspace)

    # 1. Real file creation
    target = workspace / "notes.txt"
    proxy.observe_client_line(_rpc_call(1, "write_file", {"path": "notes.txt", "content": "hello"}))
    target.write_text("hello", encoding="utf-8")
    proxy.process_server_line(_rpc_ok(1, "wrote 5 bytes"))

    # 2. Real file mutation
    proxy.observe_client_line(
        _rpc_call(2, "edit_file", {"path": "notes.txt", "content": "hello world"})
    )
    target.write_text("hello world", encoding="utf-8")
    proxy.process_server_line(_rpc_ok(2, "updated notes.txt"))

    # 3. Real file deletion
    proxy.observe_client_line(_rpc_call(3, "delete_file", {"path": "notes.txt"}))
    target.unlink()
    proxy.process_server_line(_rpc_ok(3, "deleted notes.txt"))

    verified = verify_receipt_log_file(log_path, expected_pubkey_hex=signer.public_key_bytes.hex())
    assert verified.entry_count == 3
    r1, r2, r3 = verified.receipts

    assert r1.ok is True
    assert r1.evidence_tier == "independently_verified"
    assert r1.effect_observation is not None
    assert r1.effect_observation["reason"] == "file_created"
    assert r1.effect_observation["before"]["exists"] is False
    assert r1.effect_observation["after"]["exists"] is True

    assert r2.ok is True
    assert r2.evidence_tier == "independently_verified"
    assert r2.effect_observation is not None
    assert r2.effect_observation["reason"] == "file_mutated"
    assert r2.effect_observation["before"]["sha256"] != r2.effect_observation["after"]["sha256"]

    assert r3.ok is True
    assert r3.evidence_tier == "independently_verified"
    assert r3.effect_observation is not None
    assert r3.effect_observation["reason"] == "file_deleted"
    assert r3.effect_observation["after"]["exists"] is False


def test_filesystem_probe_catches_silent_noop_servers_and_grounds_supervisor(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    existing = workspace / "secrets.txt"
    existing.write_text("do-not-leak", encoding="utf-8")
    unchanged = workspace / "config.yaml"
    unchanged.write_text("mode: safe\n", encoding="utf-8")

    log_path = tmp_path / "receipts.jsonl"
    signer = SignerFactory.from_master_key(b"\x44" * 32).tool_verified()
    writer = ReceiptLogWriter(log_path, signer, truncate=True)
    proxy = McpStdioProxy(writer, endpoint_id="stdio:lying-fs", probe_fs_root=workspace)

    # 1. Silent no-op create: server returns isError:false but never creates file
    proxy.observe_client_line(_rpc_call(1, "write_file", {"path": "ghost.txt", "content": "hi"}))
    proxy.process_server_line(_rpc_ok(1, "created ghost.txt"))

    # 2. Silent no-op overwrite: server returns isError:false without touching config.yaml
    proxy.observe_client_line(
        _rpc_call(2, "update_file", {"path": "config.yaml", "content": "mode: prod\n"})
    )
    proxy.process_server_line(_rpc_ok(2, "updated config.yaml"))

    # 3. Silent no-op delete: server returns isError:false while secrets.txt remains on disk
    proxy.observe_client_line(_rpc_call(3, "delete_file", {"path": "secrets.txt"}))
    proxy.process_server_line(_rpc_ok(3, "deleted secrets.txt"))

    verified = verify_receipt_log_file(log_path, expected_pubkey_hex=signer.public_key_bytes.hex())
    r1, r2, r3 = verified.receipts

    assert r1.ok is False
    assert r1.error_code == "effect_probe_contradicted"
    assert r1.evidence_tier == "effect_contradicted"
    assert r1.effect_observation is not None
    assert r1.effect_observation["reason"] == "file_missing_after_write"

    assert r2.ok is False
    assert r2.error_code == "effect_probe_contradicted"
    assert r2.evidence_tier == "effect_contradicted"
    assert r2.effect_observation is not None
    assert r2.effect_observation["reason"] == "file_unchanged_after_ok"

    assert r3.ok is False
    assert r3.error_code == "effect_probe_contradicted"
    assert r3.evidence_tier == "effect_contradicted"
    assert r3.effect_observation is not None
    assert r3.effect_observation["reason"] == "file_still_exists_after_delete"

    # End-to-end supervision: LLM trusting the lying server is caught!
    sup = ReceiptSupervisor(device="CPU", warn_on_missing=False)
    verdict = sup.verify_narration(
        "I deleted secrets.txt as requested.",
        receipts=[r3],
        user_prompt="Delete secrets.txt.",
    )
    assert verdict.is_valid is False


def test_filesystem_probe_enforces_root_containment_and_tamper_binding(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    log_path = tmp_path / "receipts.jsonl"
    signer = SignerFactory.from_master_key(b"\x55" * 32).tool_verified()
    writer = ReceiptLogWriter(log_path, signer, truncate=True)
    proxy = McpStdioProxy(writer, endpoint_id="stdio:fs", probe_fs_root=workspace)

    # Path traversal outside probe_fs_root must NOT be probed
    proxy.observe_client_line(_rpc_call(1, "delete_file", {"path": "../outside.txt"}))
    proxy.process_server_line(_rpc_ok(1, "ok"))

    # Probed call inside root for tamper test
    proxy.observe_client_line(_rpc_call(2, "write_file", {"path": "missing.txt"}))
    proxy.process_server_line(_rpc_ok(2, "ok"))

    verified = verify_receipt_log_file(log_path, expected_pubkey_hex=signer.public_key_bytes.hex())
    r_out, r_in = verified.receipts
    assert r_out.evidence_tier == "server_reported"
    assert r_out.effect_observation is None
    assert r_in.evidence_tier == "effect_contradicted"

    # Tampering with effect_observation["verified"] must break cryptographic verification
    lines = log_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    row["receipt"]["effect_observation"]["verified"] = True
    lines[1] = json.dumps(row)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises((SignatureTamperError, MerkleTamperError)):
        verify_receipt_log_file(log_path, expected_pubkey_hex=signer.public_key_bytes.hex())
