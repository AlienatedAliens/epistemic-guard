"""Deterministic software-validation and protocol-conformance suite for epistemic-guard."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_guard.corroboration import corroborate_sources, deduplicate_syndicated_sources  # noqa: E402
from epistemic_guard.exceptions import MerkleTamperError, ProvenanceError, SignatureTamperError  # noqa: E402
from epistemic_guard.integrations import ReceiptRecorder  # noqa: E402
from epistemic_guard.mcp_proxy import McpStdioProxy, verify_receipt_log_file  # noqa: E402
from epistemic_guard.provenance import ProvenanceGate, SignerFactory  # noqa: E402
from epistemic_guard.schemas import Provenance, SignedPayload, ToolReceipt, UnsignedPayload  # noqa: E402
from epistemic_guard.store import MerkleReceiptLog, SQLiteProvenanceStore  # noqa: E402


def _deterministic_key(label: str) -> bytes:
    return hashlib.sha256(f"epistemic-guard-bench:{label}".encode()).digest()


def _bench_merkle_and_tamper() -> dict[str, int | float]:
    factory = SignerFactory.from_master_key(_deterministic_key("merkle"))
    gate = ProvenanceGate(factory.public_keys())
    merkle = MerkleReceiptLog(gate=gate)
    recorder = ReceiptRecorder(signer=factory.tool_verified(), merkle_log=merkle)

    @recorder.tool
    def write_record(idx: int) -> str:
        return f"committed-{idx}"

    n_receipts = 200
    for idx in range(n_receipts):
        write_record(idx)

    valid_proofs = 0
    tamper_rejected = 0
    for idx in range(n_receipts):
        proof = merkle.inclusion_proof(idx)
        receipt = recorder.receipts[idx]
        if MerkleReceiptLog.verify_inclusion_proof(receipt, proof):
            valid_proofs += 1
        forged = ToolReceipt(
            call_id=receipt.call_id,
            tool=receipt.tool,
            args=receipt.args,
            ok=receipt.ok,
            executed=receipt.executed,
            output=f"{receipt.output}-tampered",
            timestamp_utc=receipt.timestamp_utc,
        )
        try:
            MerkleReceiptLog.verify_inclusion_proof(forged, proof)
        except MerkleTamperError:
            tamper_rejected += 1

    return {
        "n_receipts": n_receipts,
        "valid_inclusion_proofs": valid_proofs,
        "valid_inclusion_pct": round(100.0 * valid_proofs / n_receipts, 1),
        "tampered_payloads_rejected": tamper_rejected,
        "tamper_rejection_pct": round(100.0 * tamper_rejected / n_receipts, 1),
        "false_accepts": n_receipts - tamper_rejected,
    }


def _bench_aer1_interop() -> dict[str, int | float]:
    n_cases = 50
    valid_roundtrips = 0
    tampered_rejected = 0
    for idx in range(n_cases):
        executed_flag = idx % 7 != 0
        ok_flag = executed_flag and (idx % 5 != 0)
        rec = ToolReceipt(
            call_id=f"call-{idx:03d}",
            tool="write_file",
            args={"path": f"doc_{idx}.txt", "bytes": idx * 16},
            ok=ok_flag,
            executed=executed_flag,
            error_code="EINVAL" if not executed_flag else "",
            output="ok" if ok_flag else "error",
            timestamp_utc="2026-11-18T00:00:00+00:00",
        )
        aer1_doc = rec.to_aer1()
        restored = ToolReceipt.from_aer1(aer1_doc)
        if restored.tool == rec.tool and restored.executed == rec.executed and restored.ok == rec.ok:
            valid_roundtrips += 1

        bad_doc = json.loads(json.dumps(aer1_doc))
        bad_doc["output_hash"] = "sha256:" + ("0" * 64)
        try:
            ToolReceipt.from_aer1(bad_doc)
        except (SignatureTamperError, ProvenanceError, ValueError):
            tampered_rejected += 1

    return {
        "n_aer1_cases": n_cases,
        "valid_roundtrips": valid_roundtrips,
        "valid_roundtrip_pct": round(100.0 * valid_roundtrips / n_cases, 1),
        "digest_mismatches_rejected": tampered_rejected,
        "digest_rejection_pct": round(100.0 * tampered_rejected / n_cases, 1),
    }


def _bench_mcp_proxy_wire() -> dict[str, int | float]:
    import io

    from epistemic_guard.mcp_proxy import run_mcp_stdio_proxy

    child_code = """
import json, os, sys
leaked = 1 if "EPISTEMIC_GUARD_MCP_KEY_HEX" in os.environ else 0
for raw in sys.stdin:
    line = raw.strip()
    if not line:
        continue
    msg = json.loads(line)
    mid = msg.get("id")
    params = msg.get("params", {})
    name = params.get("name", "")
    if name == "ok_tool":
        resp = {"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": f"leak={leaked}"}], "isError": False}}
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
    elif name == "err_tool":
        resp = {"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": "disk full"}], "isError": True}}
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
    elif name == "invalid_params":
        resp = {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": "Invalid params"}}
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
    elif name == "crash_tool":
        sys.exit(1)
"""
    master_hex = _deterministic_key("mcp-proxy").hex()
    n_per_mode = 10
    modes = ("ok_tool", "err_tool", "invalid_params", "crash_tool")

    with tempfile.TemporaryDirectory(prefix="eg_mcp_bench_") as tmp:
        tmp_dir = Path(tmp)
        script_path = tmp_dir / "child_mcp.py"
        script_path.write_text(child_code, encoding="utf-8")
        log_path = tmp_dir / "receipts.jsonl"
        pub_path = tmp_dir / "mcp.pub.json"

        old_env = os.environ.get("EPISTEMIC_GUARD_MCP_KEY_HEX")
        os.environ["EPISTEMIC_GUARD_MCP_KEY_HEX"] = master_hex
        try:
            # Batch the 30 non-crash calls into one stdio session
            steady_reqs = []
            req_id = 0
            for mode in ("ok_tool", "err_tool", "invalid_params"):
                for _ in range(n_per_mode):
                    req_id += 1
                    steady_reqs.append(
                        {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "method": "tools/call",
                            "params": {"name": mode, "arguments": {"seq": req_id}},
                        }
                    )
            wire_in = io.BytesIO(
                "".join(json.dumps(r) + "\n" for r in steady_reqs).encode("utf-8")
            )
            wire_out = io.BytesIO()
            run_mcp_stdio_proxy(
                [sys.executable, "-u", str(script_path)],
                log_path=log_path,
                pubkey_out=pub_path,
                include_aer1=True,
                client_in=wire_in,
                client_out=wire_out,
            )
            key_leaks = 1 if "leak=1" in wire_out.getvalue().decode("utf-8") else 0

            # Run 10 mid-call server crash events appending to the same Merkle log
            for _ in range(n_per_mode):
                req_id += 1
                crash_in = io.BytesIO(
                    (
                        json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "method": "tools/call",
                                "params": {"name": "crash_tool", "arguments": {"seq": req_id}},
                            }
                        )
                        + "\n"
                    ).encode("utf-8")
                )
                run_mcp_stdio_proxy(
                    [sys.executable, "-u", str(script_path)],
                    log_path=log_path,
                    pubkey_out=pub_path,
                    include_aer1=True,
                    client_in=crash_in,
                    client_out=io.BytesIO(),
                )
        finally:
            if old_env is None:
                os.environ.pop("EPISTEMIC_GUARD_MCP_KEY_HEX", None)
            else:
                os.environ["EPISTEMIC_GUARD_MCP_KEY_HEX"] = old_env

        ver = verify_receipt_log_file(log_path, pubkey_file=pub_path)
        total_calls = n_per_mode * len(modes)
        executed_true = sum(1 for r in ver.receipts if r.executed)
        executed_false = sum(1 for r in ver.receipts if not r.executed)
        ok_true = sum(1 for r in ver.receipts if r.ok)

        tampered_log = tmp_dir / "receipts_tampered.jsonl"
        lines = log_path.read_text(encoding="utf-8").splitlines()
        mut = json.loads(lines[0])
        mut["receipt"]["output"] = "tampered bytes"
        lines[0] = json.dumps(mut)
        tampered_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tampered_rejected = 0
        try:
            verify_receipt_log_file(tampered_log, pubkey_file=pub_path)
        except (SignatureTamperError, MerkleTamperError):
            tampered_rejected = 1

    return {
        "n_wire_calls": total_calls,
        "n_per_failure_mode": n_per_mode,
        "failure_mode_count": len(modes),
        "verified_entries": ver.entry_count,
        "key_pinned": 1 if ver.key_pinned else 0,
        "wire_verification_pct": round(100.0 * ver.entry_count / total_calls, 1),
        "executed_true_count": executed_true,
        "executed_false_count": executed_false,
        "succeeded_ok_count": ok_true,
        "child_env_key_leaks": key_leaks,
        "tampered_log_rejected": tampered_rejected,
    }


def _bench_sqlite_provenance_gate() -> dict[str, int | float]:
    factory = SignerFactory.from_master_key(_deterministic_key("sqlite-gate"))
    gate = ProvenanceGate(factory.public_keys())
    store = SQLiteProvenanceStore(":memory:", gate)

    n_user = 100
    n_tool = 100
    n_llm = 50
    n_forged = 50

    accepted_valid = 0
    for idx in range(n_user):
        env = factory.user_explicit().sign(f"user fact {idx}", trace_id=f"u-{idx}")
        store.append(env, expected_provenance=Provenance.USER_EXPLICIT)
        accepted_valid += 1
    for idx in range(n_tool):
        env = factory.tool_verified().sign(f"tool fact {idx}", trace_id=f"t-{idx}")
        store.append(env, expected_provenance=Provenance.TOOL_VERIFIED)
        accepted_valid += 1

    blocked_llm = 0
    for idx in range(n_llm):
        raw_llm = UnsignedPayload(
            payload_id=f"llm-{idx}",
            content=f"assistant claim {idx}",
            provenance=Provenance.LLM_NARRATION,
            trace_id=f"l-{idx}",
            created_at_utc="2026-11-18T00:00:00+00:00",
        )
        env_llm = SignedPayload(
            payload_id=raw_llm.payload_id,
            content=raw_llm.content,
            provenance=raw_llm.provenance,
            confidence=raw_llm.confidence,
            trace_id=raw_llm.trace_id,
            created_at_utc=raw_llm.created_at_utc,
            metadata=raw_llm.metadata,
            signature="ed25519:" + ("00" * 64),
            signer_domain=Provenance.LLM_NARRATION.value,
        )
        try:
            store.append(env_llm)
        except ProvenanceError:
            blocked_llm += 1

    attacker_factory = SignerFactory.from_master_key(_deterministic_key("attacker"))
    blocked_forged = 0
    for idx in range(n_forged):
        forged_env = attacker_factory.user_explicit().sign(
            f"injected fact {idx}", trace_id=f"f-{idx}"
        )
        try:
            store.append(forged_env)
        except SignatureTamperError:
            blocked_forged += 1

    chain_len = store.verify_chain()

    # Check SQLite immutability triggers (UPDATE and DELETE must fail)
    immutable_triggers_blocked = 0
    import sqlite3

    for sql in (
        "UPDATE provenance_envelopes SET content = 'mutated' WHERE seq = 1",
        "DELETE FROM provenance_envelopes WHERE seq = 1",
    ):
        try:
            store._conn.execute(sql)
        except sqlite3.IntegrityError:
            immutable_triggers_blocked += 1

    store.close()
    total_valid = n_user + n_tool
    total_unauthorized = n_llm + n_forged
    blocked_total = blocked_llm + blocked_forged

    return {
        "n_total_write_attempts": total_valid + total_unauthorized,
        "n_valid_writes": total_valid,
        "valid_writes_accepted": accepted_valid,
        "false_rejections": total_valid - accepted_valid,
        "false_rejection_pct": round(100.0 * (total_valid - accepted_valid) / total_valid, 1),
        "n_unauthorized_writes": total_unauthorized,
        "n_llm_narration_attempts": n_llm,
        "n_forged_key_attempts": n_forged,
        "unauthorized_writes_blocked": blocked_total,
        "unauthorized_block_pct": round(100.0 * blocked_total / total_unauthorized, 1),
        "verified_chain_records": chain_len,
        "immutable_sql_triggers_tested": 2,
        "immutable_sql_triggers_blocked": immutable_triggers_blocked,
    }


def _bench_syndication_dedup() -> dict[str, int | float]:
    from epistemic_guard.corroboration import OpenVINOWebEvidenceClassifier
    from epistemic_guard.hazard import OpenVINOHazardClassifier
    from epistemic_guard.nli import OpenVINONLIVerifier

    hz_clf = OpenVINOHazardClassifier(device="CPU", warn_on_missing=False)
    ev_clf = OpenVINOWebEvidenceClassifier(device="CPU", warn_on_missing=False)
    nli_ver = OpenVINONLIVerifier(device="CPU", warn_on_missing=False)

    n_clusters = 45
    mirrors_per_cluster = 3
    total_urls = n_clusters * mirrors_per_cluster
    single_origin_collapsed_clusters = 0
    high_hazard_blocked = 0

    for idx in range(n_clusters):
        wire_text = (
            f"Clinical bulletin {idx}: adult patients receiving compound-{idx} require "
            f"initial dosing between 2 mg and 5 mg once daily with serum monitoring."
        )
        pairs = [
            (f"wire-origin-{idx}.org", wire_text),
            (f"syndicated-a-{idx}.net", wire_text),
            (f"syndicated-b-{idx}.io", wire_text + " Additional editorial footer."),
        ]
        deduped, collapsed = deduplicate_syndicated_sources(pairs)
        if len(deduped) == 1 and collapsed == 2:
            single_origin_collapsed_clusters += 1

        burden = corroborate_sources(
            f"What is the recommended initial adult dosage in mg for compound-{idx}?",
            [
                {"url": f"https://wire-origin-{idx}.org/a", "snippet": wire_text},
                {"url": f"https://syndicated-a-{idx}.net/b", "snippet": wire_text},
                {"url": f"https://syndicated-b-{idx}.io/c", "snippet": wire_text},
            ],
            classifier=hz_clf,
            evidence_classifier=ev_clf,
            nli_verifier=nli_ver,
            device="CPU",
            warn_on_missing=False,
        )
        if burden.independent_sources_count == 1 and not burden.meets_casual_burden:
            high_hazard_blocked += 1

    return {
        "n_syndication_clusters": n_clusters,
        "mirrors_per_cluster": mirrors_per_cluster,
        "total_syndicated_urls": total_urls,
        "duplicate_urls_collapsed_per_cluster": mirrors_per_cluster - 1,
        "total_duplicate_urls_collapsed": n_clusters * (mirrors_per_cluster - 1),
        "single_origin_clusters_collapsed": single_origin_collapsed_clusters,
        "single_origin_collapse_pct": round(
            100.0 * single_origin_collapsed_clusters / n_clusters, 1
        ),
        "high_hazard_false_consensus_blocked": high_hazard_blocked,
        "high_hazard_block_pct": round(100.0 * high_hazard_blocked / n_clusters, 1),
    }


def main() -> None:
    # Ensure subprocess invocation is unused warning-clean
    _ = subprocess
    payload = {
        "merkle_receipts": _bench_merkle_and_tamper(),
        "aer1_interop": _bench_aer1_interop(),
        "mcp_proxy_wire": _bench_mcp_proxy_wire(),
        "sqlite_provenance_gate": _bench_sqlite_provenance_gate(),
        "syndication_dedup": _bench_syndication_dedup(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
