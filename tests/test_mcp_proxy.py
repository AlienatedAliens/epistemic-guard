"""End-to-end tests for out-of-process `mcp-proxy` and `verify-log`."""

from __future__ import annotations

import io
import json
import os
import subprocess
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


def test_multiprocess_concurrent_receipt_log_writer_and_key_rotation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    key_file = tmp_path / "shared.key"
    pub_file = tmp_path / "shared.pub.json"
    shared_log = tmp_path / "concurrent.jsonl"
    load_or_create_mcp_signer(key_file=key_file, pubkey_out=pub_file)

    worker_script = tmp_path / "worker_append.py"
    worker_script.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "from epistemic_guard import ReceiptLogWriter, ToolReceipt, load_or_create_mcp_signer\n"
        "log_p, key_p, worker_id = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]\n"
        "signer = load_or_create_mcp_signer(key_file=key_p)\n"
        "writer = ReceiptLogWriter(log_p, signer, session_id='sess-shared-1')\n"
        "for idx in range(8):\n"
        "    writer.append(ToolReceipt(\n"
        "        call_id=f'{worker_id}-{idx}',\n"
        "        tool='write_file',\n"
        "        ok=True,\n"
        "        args={'worker': worker_id, 'idx': idx},\n"
        "        output=f'ok-{worker_id}-{idx}',\n"
        "    ))\n",
        encoding="utf-8",
    )

    procs = [
        subprocess.Popen(
            [sys.executable, str(worker_script), str(shared_log), str(key_file), f"w{w}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for w in range(3)
    ]
    for proc in procs:
        out, err = proc.communicate(timeout=30)
        assert proc.returncode == 0, f"worker failed: {out}\n{err}"

    verified_shared = verify_receipt_log_file(
        shared_log,
        pubkey_file=pub_file,
        expected_session_id="sess-shared-1",
    )
    assert verified_shared.entry_count == 24
    assert verified_shared.session_id == "sess-shared-1"
    assert verified_shared.prev_chain_sha256 == "0" * 64

    # Key rotation: rotate to a new Ed25519 key while cryptographically chaining to shared_log tip
    rot_key = tmp_path / "rotated.key"
    rot_pub = tmp_path / "rotated.pub.json"
    rot_log = tmp_path / "rotated.jsonl"
    rot_signer = load_or_create_mcp_signer(key_file=rot_key, pubkey_out=rot_pub)
    rot_writer = ReceiptLogWriter(
        rot_log,
        rot_signer,
        session_id="sess-rotated-2",
        prev_log_path=shared_log,
    )
    rot_writer.append(
        ToolReceipt(
            call_id="turn0-del",
            tool="delete_file",
            ok=True,
            args={"path": "old_cache.db"},
            output="deleted old_cache.db",
        )
    )
    rot_writer.append(
        ToolReceipt(
            call_id="turn1-write",
            tool="write_file",
            ok=True,
            args={"path": "summary.md"},
            output="wrote summary.md",
        )
    )

    verified_rot = verify_receipt_log_file(
        rot_log,
        pubkey_file=rot_pub,
        expected_session_id="sess-rotated-2",
        prev_log_path=shared_log,
    )
    assert rot_writer.merkle_log.verify_integrity() is True
    assert verified_rot.entry_count == 2
    assert verified_rot.prev_chain_sha256 == verified_shared.head_chain_hash

    # Tampering or swapping predecessor log breaks rotation chain continuity
    fake_prev = tmp_path / "fake_prev.jsonl"
    fake_signer = load_or_create_mcp_signer()
    ReceiptLogWriter(fake_prev, fake_signer).append(
        ToolReceipt(tool="ping", ok=True, output="pong")
    )
    with pytest.raises(MerkleTamperError):
        verify_receipt_log_file(rot_log, pubkey_file=rot_pub, prev_log_path=fake_prev)

    # CLI verify-log with --prev-log, --session-id, and --since-seq blocks stale Turn-0 claim
    rc_stale = cli_main(
        [
            "verify-log",
            str(rot_log),
            "--pubkey-file",
            str(rot_pub),
            "--prev-log",
            str(shared_log),
            "--session-id",
            "sess-rotated-2",
            "--since-seq",
            "1",
            "--narration",
            "I wrote summary.md; I also deleted old_cache.db.",
            "--json",
        ]
    )
    assert rc_stale == 1
    stale_out = json.loads(capsys.readouterr().out)
    assert stale_out["log_valid"] is True
    assert stale_out["narration_valid"] is False

    # Current-turn narration (seq >= 1) passes cleanly
    rc_fresh = cli_main(
        [
            "verify-log",
            str(rot_log),
            "--pubkey-file",
            str(rot_pub),
            "--prev-log",
            str(shared_log),
            "--session-id",
            "sess-rotated-2",
            "--since-seq",
            "1",
            "--narration",
            "I wrote summary.md.",
        ]
    )
    assert rc_fresh == 0
    capsys.readouterr()


def test_mcp_proxy_scrubs_master_and_secret_env_vars_and_handles_duplicate_rpc_ids(
    tmp_path: Path,
) -> None:
    from epistemic_guard.mcp_proxy import McpStdioProxy, _scrub_child_env

    raw_env = {
        "PATH": "/usr/bin",
        "EPISTEMIC_GUARD_MCP_KEY_HEX": "aa" * 32,
        "EPISTEMIC_GUARD_MASTER_KEY": "bb" * 32,
        "epistemic_guard_signing_secret": "cc" * 32,
        "EPISTEMIC_GUARD_SESSION_ID": "sess-keep",
        "EPISTEMIC_GUARD_CACHE_DIR": str(tmp_path / "eg_cache"),
    }
    scrubbed = _scrub_child_env(raw_env)
    assert "EPISTEMIC_GUARD_MCP_KEY_HEX" not in scrubbed
    assert "EPISTEMIC_GUARD_MASTER_KEY" not in scrubbed
    assert "epistemic_guard_signing_secret" not in scrubbed
    assert scrubbed["EPISTEMIC_GUARD_SESSION_ID"] == "sess-keep"
    assert scrubbed["EPISTEMIC_GUARD_CACHE_DIR"] == str(tmp_path / "eg_cache")

    # Duplicate in-flight JSON-RPC IDs are matched FIFO instead of overwriting
    log_path = tmp_path / "dup_ids.jsonl"
    signer = load_or_create_mcp_signer()
    writer = ReceiptLogWriter(log_path, signer)
    proxy = McpStdioProxy(writer)

    proxy.observe_client_line(
        b'{"jsonrpc":"2.0","id":7,"method":"tools/call",'
        b'"params":{"name":"write_a","arguments":{"p":"a.txt"}}}\n'
    )
    proxy.observe_client_line(
        b'{"jsonrpc":"2.0","id":7,"method":"tools/call",'
        b'"params":{"name":"write_b","arguments":{"p":"b.txt"}}}\n'
    )
    proxy.observe_server_line(
        b'{"jsonrpc":"2.0","id":7,"result":{"content":[{"type":"text","text":"done-a"}]}}\n'
    )
    proxy.observe_server_line(
        b'{"jsonrpc":"2.0","id":7,"result":{"content":[{"type":"text","text":"done-b"}]}}\n'
    )

    verified = verify_receipt_log_file(log_path)
    assert verified.entry_count == 2
    assert verified.receipts[0].tool == "write_a"
    assert verified.receipts[0].output == "done-a"
    assert verified.receipts[0].call_id == "mcp-7"
    assert verified.receipts[1].tool == "write_b"
    assert verified.receipts[1].output == "done-b"
    assert verified.receipts[1].call_id == "mcp-7#2"


def test_active_rehash_reorder_aer1_tamper_and_cli_bounds_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from epistemic_guard.store import MerkleReceiptLog

    key_file = tmp_path / "entry_sig.key"
    pub_file = tmp_path / "entry_sig.pub.json"
    log_path = tmp_path / "entry_sig.jsonl"
    signer = load_or_create_mcp_signer(key_file=key_file, pubkey_out=pub_file)
    writer = ReceiptLogWriter(log_path, signer, include_aer1=True, session_id="sess-bind-1")
    writer.append(
        ToolReceipt(call_id="c-0", tool="write_file", ok=True, args={"p": "a.txt"}, output="ok-a")
    )
    writer.append(
        ToolReceipt(call_id="c-1", tool="delete_file", ok=False, args={"p": "b.db"}, output="err-b")
    )

    raw_entries = [
        json.loads(ln) for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    assert len(raw_entries) == 2

    # 1. Active attacker swaps seq=0 and seq=1 AND recomputes public SHA-256 chain & Merkle roots
    reordered = [dict(raw_entries[1]), dict(raw_entries[0])]
    rehash_merkle = MerkleReceiptLog()
    rebuilt_lines: list[str] = []
    for new_seq, item in enumerate(reordered):
        rcpt = ToolReceipt.model_validate(item["receipt"])
        new_chain = rehash_merkle.append(rcpt)
        proof = rehash_merkle.inclusion_proof(new_seq)
        item["seq"] = new_seq
        item["leaf_sha256"] = proof.leaf_sha256
        item["chain_sha256"] = new_chain
        item["merkle_root"] = proof.root_sha256
        item.pop("prev_chain_sha256", None)
        rebuilt_lines.append(json.dumps(item, sort_keys=True, separators=(",", ":")))

    swapped_log = tmp_path / "swapped_rehashed.jsonl"
    swapped_log.write_text("\n".join(rebuilt_lines) + "\n", encoding="utf-8")
    with pytest.raises(SignatureTamperError):
        verify_receipt_log_file(swapped_log, pubkey_file=pub_file)

    # 2. Tampering embedded entry.aer1 without touching signed ToolReceipt is rejected
    aer1_tampered = [dict(raw_entries[0]), dict(raw_entries[1])]
    forged_rcpt = ToolReceipt(
        call_id="c-0", tool="delete_file", ok=True, args={"p": "prod.db"}, output="wiped"
    )
    aer1_tampered[0]["aer1"] = forged_rcpt.to_aer1()
    aer1_log = tmp_path / "aer1_tampered.jsonl"
    aer1_log.write_text(
        "\n".join(json.dumps(x, sort_keys=True, separators=(",", ":")) for x in aer1_tampered)
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SignatureTamperError):
        verify_receipt_log_file(aer1_log, pubkey_file=pub_file)

    # 3. Duplicate call_id in ReceiptLogWriter is rejected
    with pytest.raises(MerkleTamperError):
        writer.append(ToolReceipt(call_id="c-0", tool="ping", ok=True, output="dup"))

    # 4. Out-of-range --since-seq and missing --narration-file fail closed with exit code 2
    rc_bad_seq = cli_main(
        [
            "verify-log",
            str(log_path),
            "--pubkey-file",
            str(pub_file),
            "--since-seq",
            "99",
            "--narration",
            "Done.",
            "--json",
        ]
    )
    assert rc_bad_seq == 2
    err_seq = json.loads(capsys.readouterr().out)
    assert err_seq["log_valid"] is False

    rc_missing_file = cli_main(
        [
            "verify-log",
            str(log_path),
            "--pubkey-file",
            str(pub_file),
            "--narration-file",
            str(tmp_path / "missing_reply.txt"),
            "--json",
        ]
    )
    assert rc_missing_file == 2
    err_file = json.loads(capsys.readouterr().out)
    assert err_file["log_valid"] is False




def test_strip_untrusted_annotations_monotonic_sanitization(tmp_path: Path) -> None:
    """`--strip-untrusted-annotations` enforces the monotonic ('remove, never add') rule:
    privilege-raising wire hints are stripped before forwarding `tools/list`, while
    restrictive hints (`destructiveHint: true`, `openWorldHint: true`) and `title` remain,
    and `--pin-tools` still pins the raw server-sent annotations against rug-pulls.
    """
    from epistemic_guard.mcp_proxy import McpStdioProxy, ReceiptLogWriter
    from epistemic_guard.provenance import SignerFactory

    log_path = tmp_path / "ann_proxy.jsonl"
    signer = SignerFactory.from_master_key(b"\x33" * 32).tool_verified()
    writer = ReceiptLogWriter(log_path, signer)
    proxy = McpStdioProxy(
        writer,
        pin_tools=True,
        strip_untrusted_annotations=True,
    )

    req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}).encode()
    proxy.process_client_line(req + b"\n")

    resp = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "tools": [
                {
                    "name": "update_issue_labels",
                    "description": "Update labels on an issue",
                    "inputSchema": {"type": "object"},
                    "annotations": {
                        "title": "Update Issue Labels",
                        "readOnlyHint": True,
                        "destructiveHint": False,
                        "idempotentHint": True,
                        "openWorldHint": False,
                    },
                },
                {
                    "name": "drop_table",
                    "description": "Contradictory hints",
                    "inputSchema": {"type": "object"},
                    "annotations": {
                        "readOnlyHint": True,
                        "destructiveHint": True,
                        "openWorldHint": True,
                    },
                },
            ]
        },
    }
    forwarded = proxy.process_server_line(json.dumps(resp).encode() + b"\n")
    out_doc = json.loads(forwarded.decode("utf-8"))
    t0, t1 = out_doc["result"]["tools"]

    # Privilege-raising hints stripped; display title kept
    assert t0["annotations"] == {"title": "Update Issue Labels"}
    # Contradictory readOnlyHint=True dropped; restrictive destructiveHint/openWorldHint kept
    assert t1["annotations"] == {"destructiveHint": True, "openWorldHint": True}
    # Pinning still recorded raw server-sent annotations
    assert "update_issue_labels" in proxy.tool_pins
