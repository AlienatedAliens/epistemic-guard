"""Tests for the standalone `epistemic-guard mcp-server` stdio JSON-RPC handler."""

from __future__ import annotations

import io
import json
from pathlib import Path

from epistemic_guard.mcp_proxy import ReceiptLogWriter, load_or_create_mcp_signer
from epistemic_guard.mcp_server import MCP_TOOLS, EpistemicGuardMcpServer, run_mcp_stdio_server
from epistemic_guard.schemas import ToolReceipt
from epistemic_guard.store import MerkleInclusionProof, MerkleReceiptLog


def test_mcp_server_introspection_handshake() -> None:
    lines = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "glama-inspector", "version": "1.0.0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "resources/list"},
        {"jsonrpc": "2.0", "id": 4, "method": "prompts/list"},
        {"jsonrpc": "2.0", "id": 5, "method": "ping"},
    ]
    raw_in = "".join(json.dumps(item) + "\n" for item in lines).encode("utf-8")
    in_buf = io.BytesIO(raw_in)
    out_buf = io.BytesIO()

    rc = run_mcp_stdio_server(client_in=in_buf, client_out=out_buf)
    assert rc == 0

    responses = [
        json.loads(line)
        for line in out_buf.getvalue().decode("utf-8").splitlines()
        if line.strip()
    ]
    assert [r["id"] for r in responses] == [1, 2, 3, 4, 5]
    assert responses[0]["result"]["serverInfo"]["name"] == "epistemic-guard"
    tools = responses[1]["result"]["tools"]
    assert len(tools) == len(MCP_TOOLS) == 4
    assert {t["name"] for t in tools} == {
        "verify_narration",
        "verify_receipt_log",
        "inspect_receipt",
        "corroborate_sources",
    }
    for tool_def in tools:
        assert tool_def["annotations"]["readOnlyHint"] is True
        assert tool_def["inputSchema"]["additionalProperties"] is False
        assert tool_def["outputSchema"]["type"] == "object"
        assert tool_def["outputSchema"]["additionalProperties"] is False


def test_mcp_server_verify_narration_and_corroborate() -> None:
    srv = EpistemicGuardMcpServer()

    resp = srv.handle_message(
        {
            "jsonrpc": "2.0",
            "id": "v1",
            "method": "tools/call",
            "params": {
                "name": "verify_narration",
                "arguments": {
                    "narration": (
                        "[ACTION: write_file notes.txt] I saved notes.txt; "
                        "[ACTION: delete_file staging.db] I also deleted staging.db."
                    ),
                    "receipts": [
                        {
                            "call_id": "c1",
                            "tool": "write_file",
                            "ok": True,
                            "executed": True,
                            "args": {"path": "notes.txt"},
                            "output": "ok",
                        }
                    ],
                    "user_prompt": "Save notes.txt.",
                },
            },
        }
    )
    assert isinstance(resp, dict)
    result = resp["result"]
    assert isinstance(result, dict)
    assert result["isError"] is False
    payload = json.loads(result["content"][0]["text"])
    assert payload["is_valid"] is False
    assert payload["retry_prompt"] != ""

    wire = "Initial adult warfarin dosing starts at 2 mg to 5 mg once daily."
    corr_resp = srv.handle_message(
        {
            "jsonrpc": "2.0",
            "id": "c1",
            "method": "tools/call",
            "params": {
                "name": "corroborate_sources",
                "arguments": {
                    "query": "What is starting adult warfarin dosage in mg?",
                    "sources": [
                        {"url": "https://a.example/1", "snippet": wire},
                        {"url": "https://b.example/2", "snippet": wire},
                    ],
                },
            },
        }
    )
    assert isinstance(corr_resp, dict)
    corr_payload = json.loads(corr_resp["result"]["content"][0]["text"])
    assert corr_payload["deduplicated_duplicates_collapsed"] == 1
    assert corr_payload["independent_sources_count"] == 1
    assert corr_payload["disputed"] is False


def test_mcp_server_verify_receipt_log_and_errors(tmp_path: Path) -> None:
    log_path = tmp_path / "receipts.jsonl"
    pub_path = tmp_path / "mcp.pub.json"
    signer = load_or_create_mcp_signer(pubkey_out=pub_path)
    writer = ReceiptLogWriter(log_path, signer)
    writer.append(
        ToolReceipt(
            call_id="1",
            tool="write_file",
            ok=True,
            executed=True,
            args={"path": "notes.txt"},
            output="ok",
        )
    )

    srv = EpistemicGuardMcpServer()
    ok_resp = srv.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "verify_receipt_log",
                "arguments": {
                    "log_path": str(log_path),
                    "pubkey_file": str(pub_path),
                },
            },
        }
    )
    assert isinstance(ok_resp, dict)
    parsed = json.loads(ok_resp["result"]["content"][0]["text"])
    assert parsed["log_valid"] is True
    assert parsed["entry_count"] == 1
    assert parsed["key_pinned"] is True

    bad_resp = srv.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
        }
    )
    assert isinstance(bad_resp, dict)
    assert bad_resp["result"]["isError"] is True


def test_mcp_server_inspect_receipt_and_merkle_proof(tmp_path: Path) -> None:
    log_path = tmp_path / "receipts.jsonl"
    pub_path = tmp_path / "mcp.pub.json"
    signer = load_or_create_mcp_signer(pubkey_out=pub_path)
    writer = ReceiptLogWriter(log_path, signer)
    for call_id, tool_name, path_val in (
        ("c-0", "write_file", "a.txt"),
        ("c-1", "send_email", "ops@example.com"),
        ("c-2", "write_file", "b.txt"),
    ):
        writer.append(
            ToolReceipt(
                call_id=call_id,
                tool=tool_name,
                ok=True,
                executed=True,
                args={"target": path_val},
                output="done",
            )
        )

    srv = EpistemicGuardMcpServer()
    single_resp = srv.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 20,
            "method": "tools/call",
            "params": {
                "name": "inspect_receipt",
                "arguments": {
                    "log_path": str(log_path),
                    "pubkey_file": str(pub_path),
                    "seq": 1,
                    "include_aer1": True,
                },
            },
        }
    )
    assert isinstance(single_resp, dict)
    single_res = single_resp["result"]
    assert single_res["isError"] is False
    body = json.loads(single_res["content"][0]["text"])
    assert single_res["structuredContent"] == body
    assert body["log_valid"] is True
    assert body["entry_count"] == 3
    assert body["match_count"] == 1
    entry = body["entries"][0]
    assert entry["seq"] == 1
    assert entry["receipt"]["call_id"] == "c-1"
    assert entry["receipt"]["tool"] == "send_email"

    restored_aer1 = ToolReceipt.from_aer1(entry["aer1"])
    assert restored_aer1.tool == "send_email"

    proof_raw = entry["merkle_inclusion_proof"]
    proof = MerkleInclusionProof(
        leaf_index=proof_raw["leaf_index"],
        tree_size=proof_raw["tree_size"],
        leaf_sha256=proof_raw["leaf_sha256"],
        siblings=tuple((s["side"], s["sha256"]) for s in proof_raw["siblings"]),
        root_sha256=proof_raw["root_sha256"],
    )
    rcpt_obj = ToolReceipt.model_validate(entry["receipt"])
    assert MerkleReceiptLog.verify_inclusion_proof(rcpt_obj, proof) is True

    filtered_resp = srv.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 21,
            "method": "tools/call",
            "params": {
                "name": "inspect_receipt",
                "arguments": {
                    "log_path": str(log_path),
                    "tool": "write_file",
                },
            },
        }
    )
    assert isinstance(filtered_resp, dict)
    filtered_body = json.loads(filtered_resp["result"]["content"][0]["text"])
    assert filtered_body["match_count"] == 2
    assert [e["seq"] for e in filtered_body["entries"]] == [0, 2]

    oob_resp = srv.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 22,
            "method": "tools/call",
            "params": {
                "name": "inspect_receipt",
                "arguments": {"log_path": str(log_path), "seq": 99},
            },
        }
    )
    assert isinstance(oob_resp, dict)
    assert oob_resp["result"]["isError"] is True
    assert "structuredContent" not in oob_resp["result"]


def test_mcp_server_verify_narration_signature_pinning_session_and_freshness(
    tmp_path: Path,
) -> None:
    pub_path = tmp_path / "pinned.pub.json"
    signer = load_or_create_mcp_signer(pubkey_out=pub_path)
    pub_hex = signer.public_key_bytes.hex()

    signed_rcpt = signer.sign_receipt(
        ToolReceipt(
            call_id="c-live-1",
            tool="write_file",
            ok=True,
            executed=True,
            args={"path": "report.md"},
            output="wrote report.md",
            session_id="sess-live-1",
        )
    )

    srv = EpistemicGuardMcpServer()

    # 1. Validly signed, matching session, fresh timestamp -> is_valid=True, signature_verified=True
    ok_resp = srv.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 30,
            "method": "tools/call",
            "params": {
                "name": "verify_narration",
                "arguments": {
                    "narration": "I saved the report to report.md.",
                    "receipts": [signed_rcpt.model_dump(mode="json")],
                    "expected_pubkey_hex": pub_hex,
                    "expected_session_id": "sess-live-1",
                    "max_receipt_age_seconds": 300,
                },
            },
        }
    )
    assert isinstance(ok_resp, dict)
    assert ok_resp["result"]["isError"] is False
    ok_body = ok_resp["result"]["structuredContent"]
    assert ok_body["is_valid"] is True
    assert ok_body["signature_verified"] is True

    # 2. Unsigned receipt when expected_pubkey_hex is pinned -> isError=True
    unsigned_resp = srv.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 31,
            "method": "tools/call",
            "params": {
                "name": "verify_narration",
                "arguments": {
                    "narration": "I saved the report to report.md.",
                    "receipts": [{"tool": "write_file", "ok": True, "args": {"path": "report.md"}}],
                    "expected_pubkey_hex": pub_hex,
                },
            },
        }
    )
    assert isinstance(unsigned_resp, dict)
    assert unsigned_resp["result"]["isError"] is True

    # 3. Wrong session_id -> isError=True
    wrong_sess_resp = srv.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 32,
            "method": "tools/call",
            "params": {
                "name": "verify_narration",
                "arguments": {
                    "narration": "I saved the report to report.md.",
                    "receipts": [signed_rcpt.model_dump(mode="json")],
                    "expected_pubkey_hex": pub_hex,
                    "expected_session_id": "sess-other",
                },
            },
        }
    )
    assert isinstance(wrong_sess_resp, dict)
    assert wrong_sess_resp["result"]["isError"] is True

    # 4. Stale replayed receipt (expired timestamp_utc) -> isError=True
    stale_rcpt = signer.sign_receipt(
        ToolReceipt(
            call_id="c-old",
            tool="write_file",
            ok=True,
            args={"path": "report.md"},
            output="ok",
            session_id="sess-live-1",
            timestamp_utc="2020-01-01T00:00:00+00:00",
        )
    )
    stale_resp = srv.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 33,
            "method": "tools/call",
            "params": {
                "name": "verify_narration",
                "arguments": {
                    "narration": "I saved the report to report.md.",
                    "receipts": [stale_rcpt.model_dump(mode="json")],
                    "expected_pubkey_hex": pub_hex,
                    "max_receipt_age_seconds": 60,
                },
            },
        }
    )
    assert isinstance(stale_resp, dict)
    assert stale_resp["result"]["isError"] is True

    # 5. inspect_receipt with since_seq + expected_session_id slices only current turn
    turn_log = tmp_path / "turns.jsonl"
    writer = ReceiptLogWriter(turn_log, signer, session_id="sess-turns")
    writer.append(ToolReceipt(call_id="t0", tool="delete_file", ok=True, output="deleted"))
    writer.append(ToolReceipt(call_id="t1", tool="write_file", ok=True, output="wrote"))

    slice_resp = srv.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 34,
            "method": "tools/call",
            "params": {
                "name": "inspect_receipt",
                "arguments": {
                    "log_path": str(turn_log),
                    "pubkey_file": str(pub_path),
                    "expected_session_id": "sess-turns",
                    "since_seq": 1,
                },
            },
        }
    )
    assert isinstance(slice_resp, dict)
    assert slice_resp["result"]["isError"] is False
    slice_body = slice_resp["result"]["structuredContent"]
    assert slice_body["match_count"] == 1
    assert slice_body["entries"][0]["seq"] == 1
    assert slice_body["entries"][0]["receipt"]["call_id"] == "t1"
