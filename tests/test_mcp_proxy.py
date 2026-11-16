"""End-to-end tests for out-of-process `mcp-proxy` and `verify-log`."""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import pytest

from epistemic_guard import (
    MerkleTamperError,
    ProvenanceError,
    ReceiptLogWriter,
    SignatureTamperError,
    ToolReceipt,
    load_or_create_mcp_signer,
    run_mcp_stdio_proxy,
    verify_receipt_log_file,
)
from epistemic_guard.cli import main as cli_main

_MOCK_MCP_SERVER = """
import json
import os
import sys

for raw in sys.stdin:
    line = raw.strip()
    if not line:
        continue
    msg = json.loads(line)
    if isinstance(msg, list):
        replies = []
        for item in msg:
            rid = item.get("id")
            tname = item.get("params", {}).get("name")
            replies.append({
                "jsonrpc": "2.0",
                "id": rid,
                "result": {
                    "content": [{"type": "text", "text": f"batch-ok:{tname}"}],
                    "isError": False,
                },
            })
        sys.stdout.write(json.dumps(replies) + "\\n")
        sys.stdout.flush()
        continue

    method = msg.get("method")
    rid = msg.get("id")
    if rid is None:
        continue
    if method == "initialize":
        sys.stdout.write(json.dumps({
            "jsonrpc": "2.0",
            "id": rid,
            "result": {
                "protocolVersion": "2025-06-18",
                "leaked_key": os.environ.get("EPISTEMIC_GUARD_MCP_KEY_HEX", ""),
            },
        }) + "\\n")
        sys.stdout.flush()
    elif method == "tools/list":
        sys.stdout.write(json.dumps({
            "jsonrpc": "2.0",
            "id": rid,
            "result": {"tools": [{"name": "write_file"}, {"name": "delete_file"}]},
        }) + "\\n")
        sys.stdout.flush()
    elif method == "tools/call":
        params = msg.get("params", {})
        name = params.get("name")
        args = params.get("arguments", {})
        if name == "write_file":
            if "path" not in args:
                sys.stdout.write(json.dumps({
                    "jsonrpc": "2.0",
                    "id": rid,
                    "error": {
                        "code": -32602,
                        "message": "Invalid params: missing required argument 'path'",
                    },
                }) + "\\n")
            else:
                sys.stdout.write(json.dumps({
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": {
                        "content": [{"type": "text", "text": f"wrote {args['path']}"}],
                        "isError": False,
                    },
                }) + "\\n")
            sys.stdout.flush()
        elif name == "delete_file":
            if args.get("mode") == "schema_err":
                sys.stdout.write(json.dumps({
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": {
                        "content": [{
                            "type": "text",
                            "text": "Input validation error: missing required property 'confirm'",
                        }],
                        "isError": True,
                    },
                }) + "\\n")
            else:
                sys.stdout.write(json.dumps({
                    "jsonrpc": "2.0",
                    "id": rid,
                    "result": {
                        "content": [{"type": "text", "text": "Permission denied: prod.db"}],
                        "isError": True,
                    },
                }) + "\\n")
            sys.stdout.flush()
        elif name == "crash_now":
            sys.exit(17)
"""


def test_mcp_stdio_proxy_end_to_end_and_env_key_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_script = tmp_path / "mock_mcp_server.py"
    server_script.write_text(_MOCK_MCP_SERVER, encoding="utf-8")

    log_file = tmp_path / "receipts.jsonl"
    pub_file = tmp_path / "receipts.pub.json"
    env_seed_hex = os.urandom(32).hex()
    monkeypatch.setenv("EPISTEMIC_GUARD_MCP_KEY_HEX", env_seed_hex)

    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": "call-ok",
            "method": "tools/call",
            "params": {"name": "write_file", "arguments": {"path": "notes.txt"}},
        },
        {
            "jsonrpc": "2.0",
            "id": "call-rpc-rej",
            "method": "tools/call",
            "params": {"name": "write_file", "arguments": {"wrong_key": "notes.txt"}},
        },
        {
            "jsonrpc": "2.0",
            "id": "call-mcp-val-rej",
            "method": "tools/call",
            "params": {"name": "delete_file", "arguments": {"path": "prod.db", "mode": "schema_err"}},
        },
        {
            "jsonrpc": "2.0",
            "id": "call-runtime-err",
            "method": "tools/call",
            "params": {"name": "delete_file", "arguments": {"path": "prod.db"}},
        },
        [
            {
                "jsonrpc": "2.0",
                "id": "b1",
                "method": "tools/call",
                "params": {"name": "ping_a", "arguments": {"n": 1}},
            },
            {
                "jsonrpc": "2.0",
                "id": "b2",
                "method": "tools/call",
                "params": {"name": "ping_b", "arguments": {"n": 2}},
            },
        ],
    ]
    wire_in = io.BytesIO(
        "".join(json.dumps(r) + "\n" for r in requests).encode("utf-8")
    )
    wire_out = io.BytesIO()

    rc = run_mcp_stdio_proxy(
        [sys.executable, "-u", str(server_script)],
        log_path=log_file,
        pubkey_out=pub_file,
        include_aer1=True,
        client_in=wire_in,
        client_out=wire_out,
    )
    assert rc == 0

    responses = [
        json.loads(ln)
        for ln in wire_out.getvalue().decode("utf-8").splitlines()
        if ln.strip()
    ]
    # Child MCP server must NOT inherit EPISTEMIC_GUARD_MCP_KEY_HEX
    assert responses[0]["result"]["leaked_key"] == ""
    assert len(responses) == 7  # 6 single responses + 1 batch response array

    verified = verify_receipt_log_file(log_file, pubkey_file=pub_file)
    assert verified.key_pinned is True
    assert verified.entry_count == 6

    r_ok, r_rpc_rej, r_val_rej, r_err, r_b1, r_b2 = verified.receipts
    assert r_ok.tool == "write_file" and r_ok.ok is True and r_ok.executed is True
    assert "wrote notes.txt" in r_ok.output

    assert r_rpc_rej.tool == "write_file"
    assert r_rpc_rej.ok is False and r_rpc_rej.executed is False
    assert r_rpc_rej.error_code == "jsonrpc_-32602"

    assert r_val_rej.tool == "delete_file"
    assert r_val_rej.ok is False and r_val_rej.executed is False
    assert r_val_rej.error_code == "invalid_tool_arguments"

    assert r_err.tool == "delete_file"
    assert r_err.ok is False and r_err.executed is True
    assert r_err.error_code == "mcp_tool_error"

    assert r_b1.tool == "ping_a" and r_b1.ok is True
    assert r_b2.tool == "ping_b" and r_b2.ok is True

    # Verify exported AER-1 payload roundtrips cleanly
    first_line = json.loads(log_file.read_text(encoding="utf-8").splitlines()[0])
    parsed_aer1 = ToolReceipt.from_aer1(first_line["aer1"])
    assert parsed_aer1.tool == "write_file"
    assert parsed_aer1.ok is True


def test_mcp_proxy_records_eof_receipt_when_server_crashes_mid_call(tmp_path: Path) -> None:
    server_script = tmp_path / "mock_mcp_server.py"
    server_script.write_text(_MOCK_MCP_SERVER, encoding="utf-8")
    log_file = tmp_path / "crash.jsonl"
    key_file = tmp_path / "proxy.key"

    wire_in = io.BytesIO(
        (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 99,
                    "method": "tools/call",
                    "params": {"name": "crash_now", "arguments": {"target": "db"}},
                }
            )
            + "\n"
        ).encode("utf-8")
    )
    wire_out = io.BytesIO()

    rc = run_mcp_stdio_proxy(
        [sys.executable, "-u", str(server_script)],
        log_path=log_file,
        key_file=key_file,
        client_in=wire_in,
        client_out=wire_out,
    )
    assert rc == 17

    verified = verify_receipt_log_file(log_file)
    assert verified.entry_count == 1
    rcpt = verified.receipts[0]
    assert rcpt.tool == "crash_now"
    assert rcpt.ok is False and rcpt.executed is True
    assert rcpt.error_code == "mcp_server_eof"


def test_log_resumption_tamper_detection_and_verify_log_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    key_file = tmp_path / "signer.key"
    pub_file = tmp_path / "signer.pub.json"
    log_file = tmp_path / "audit.jsonl"

    signer = load_or_create_mcp_signer(key_file=key_file, pubkey_out=pub_file)
    writer1 = ReceiptLogWriter(log_file, signer)
    writer1.append(
        ToolReceipt(tool="write_file", ok=True, args={"path": "notes.txt"}, output="wrote 12B")
    )

    # Restart writer with same key_file -> resumes Merkle chain at seq=1
    signer_reloaded = load_or_create_mcp_signer(key_file=key_file)
    writer2 = ReceiptLogWriter(log_file, signer_reloaded)
    writer2.append(
        ToolReceipt(
            tool="delete_file",
            ok=False,
            executed=False,
            error_code="invalid_tool_arguments",
            args={"wrong": "staging.db"},
            output="missing path",
        )
    )
    assert len(writer2.merkle_log) == 2

    # Different key without --truncate must refuse to mix keys in existing log
    other_signer = load_or_create_mcp_signer()
    with pytest.raises(ProvenanceError):
        ReceiptLogWriter(log_file, other_signer)

    # CLI verify-log passes with pinned key + grounded narration
    assert (
        cli_main(
            [
                "verify-log",
                str(log_file),
                "--pubkey-file",
                str(pub_file),
                "--narration",
                "I saved the checklist to notes.txt.",
                "--prompt",
                "Save the checklist to notes.txt.",
            ]
        )
        == 0
    )
    capsys.readouterr()

    # CLI verify-log returns exit code 1 on phantom/rejected deletion claim
    rc_ungrounded = cli_main(
        [
            "verify-log",
            str(log_file),
            "--pubkey-file",
            str(pub_file),
            "--narration",
            "I saved the checklist to notes.txt; I also deleted staging.db.",
            "--prompt",
            "Save notes.txt and delete staging.db.",
            "--json",
        ]
    )
    assert rc_ungrounded == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["log_valid"] is True
    assert payload["key_pinned"] is True
    assert payload["narration_valid"] is False
    assert "REJECTED BEFORE EXECUTION" in payload["retry_prompt"]

    # Key-replacement attack: attacker overwrites log using their own Ed25519 keypair
    forged_log = tmp_path / "forged.jsonl"
    attacker_signer = load_or_create_mcp_signer()
    attacker_writer = ReceiptLogWriter(forged_log, attacker_signer)
    attacker_writer.append(
        ToolReceipt(tool="delete_file", ok=True, args={"path": "staging.db"}, output="deleted")
    )
    # Without pinned key it's self-consistent, WITH pinned pub_file it raises SignatureTamperError
    with pytest.raises(SignatureTamperError):
        verify_receipt_log_file(forged_log, pubkey_file=pub_file)
    assert cli_main(["verify-log", str(forged_log), "--pubkey-file", str(pub_file)]) == 2

    # Line deletion / sequence gap in legitimate log raises MerkleTamperError
    lines = log_file.read_text(encoding="utf-8").splitlines()
    truncated_head_log = tmp_path / "gap.jsonl"
    truncated_head_log.write_text(lines[1] + "\n", encoding="utf-8")
    with pytest.raises(MerkleTamperError):
        verify_receipt_log_file(truncated_head_log, pubkey_file=pub_file)

    # Output mutation in legitimate log raises SignatureTamperError
    mutated_obj = json.loads(lines[0])
    mutated_obj["receipt"]["output"] = "tampered bytes"
    tampered_log = tmp_path / "tampered.jsonl"
    tampered_log.write_text(
        json.dumps(mutated_obj) + "\n" + lines[1] + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SignatureTamperError):
        verify_receipt_log_file(tampered_log, pubkey_file=pub_file)
