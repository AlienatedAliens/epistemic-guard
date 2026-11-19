"""Deterministic software-validation and protocol-conformance suite for epistemic-guard."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_guard.corroboration import (  # noqa: E402
    OpenVINOWebEvidenceClassifier,
    corroborate_sources,
    deduplicate_syndicated_sources,
)
from epistemic_guard.exceptions import (  # noqa: E402
    MerkleTamperError,
    ProvenanceError,
    SignatureTamperError,
)
from epistemic_guard.hazard import OpenVINOHazardClassifier  # noqa: E402
from epistemic_guard.integrations import ReceiptRecorder  # noqa: E402
from epistemic_guard.mcp_proxy import (  # noqa: E402
    ReceiptLogWriter,
    run_mcp_stdio_proxy,
    verify_receipt_log_file,
)
from epistemic_guard.nli import OpenVINONLIVerifier  # noqa: E402
from epistemic_guard.provenance import (  # noqa: E402
    ProvenanceGate,
    SignerFactory,
    canonical_receipt_bytes,
)
from epistemic_guard.schemas import (  # noqa: E402
    Provenance,
    SignedPayload,
    ToolReceipt,
    UnsignedPayload,
)
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
    factory = SignerFactory.from_master_key(_deterministic_key("aer1-interop"))
    signer = factory.tool_verified()
    gate = ProvenanceGate(factory.public_keys())
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
        signed_restored = signer.sign_receipt(restored)
        if (
            restored.tool == rec.tool
            and restored.call_id == rec.call_id
            and restored.args == rec.args
            and restored.executed == rec.executed
            and restored.ok == rec.ok
            and gate.verify_receipt(signed_restored)
        ):
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
    child_code = """
import json, os, sys
KEYS = ("EPISTEMIC_GUARD_MCP_KEY_HEX", "EPISTEMIC_GUARD_MASTER_KEY")
leaked = 1 if any(k.upper() in KEYS for k in os.environ) else 0
for raw in sys.stdin:
    line = raw.strip()
    if not line:
        continue
    msg = json.loads(line)
    mid = msg.get("id")
    params = msg.get("params", {})
    name = params.get("name", "")
    if name == "ok_tool":
        body = {"content": [{"type": "text", "text": f"leak={leaked}"}], "isError": False}
        resp = {"jsonrpc": "2.0", "id": mid, "result": body}
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
    elif name == "err_tool":
        body = {"content": [{"type": "text", "text": "disk full"}], "isError": True}
        resp = {"jsonrpc": "2.0", "id": mid, "result": body}
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
        old_master = os.environ.get("EPISTEMIC_GUARD_MASTER_KEY")
        os.environ["EPISTEMIC_GUARD_MCP_KEY_HEX"] = master_hex
        os.environ["EPISTEMIC_GUARD_MASTER_KEY"] = master_hex
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
            if old_master is None:
                os.environ.pop("EPISTEMIC_GUARD_MASTER_KEY", None)
            else:
                os.environ["EPISTEMIC_GUARD_MASTER_KEY"] = old_master

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
        "tampered_log_trials": 1,
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
        "valid_accept_pct": round(100.0 * accepted_valid / total_valid, 1),
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
        "immutable_trigger_block_pct": round(100.0 * immutable_triggers_blocked / 2, 1),
    }


def _bench_adversarial_replay_and_concurrency() -> dict[str, int | float]:
    factory = SignerFactory.from_master_key(_deterministic_key("adv-concurrency"))
    signer = factory.tool_verified()

    with tempfile.TemporaryDirectory(prefix="eg_adv_") as tmp:
        tmp_dir = Path(tmp)
        log_path = tmp_dir / "concurrent.jsonl"
        pub_path = tmp_dir / "concurrent.pub.json"
        pub_path.write_text(signer.public_key_bytes.hex() + "\n", encoding="ascii")

        n_workers = 4
        per_worker = 25
        total_concurrent = n_workers * per_worker

        def _worker(wid: int) -> None:
            writer = ReceiptLogWriter(log_path, signer, session_id="sess-alpha")
            for seq_i in range(per_worker):
                rcpt = ToolReceipt(
                    call_id=f"w{wid}-{seq_i}",
                    tool="write_file",
                    args={"worker": wid, "seq": seq_i},
                    ok=True,
                    executed=True,
                    output="committed",
                    session_id="sess-alpha",
                    timestamp_utc="2026-11-18T00:00:00+00:00",
                )
                writer.append(rcpt)

        threads = [threading.Thread(target=_worker, args=(w,)) for w in range(n_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        ver = verify_receipt_log_file(
            log_path, pubkey_file=pub_path, expected_session_id="sess-alpha"
        )

        # 1. Cross-session replay attack (log replayed under a different expected session_id)
        replay_rejected = 0
        try:
            verify_receipt_log_file(
                log_path, pubkey_file=pub_path, expected_session_id="sess-beta"
            )
        except (ProvenanceError, SignatureTamperError, MerkleTamperError):
            replay_rejected = 1

        # 2. Active entry reordering + public SHA-256 chain/Merkle rehash attack
        raw_lines = log_path.read_text(encoding="utf-8").splitlines()
        reordered_path = tmp_dir / "reordered.jsonl"
        parsed_entries = [json.loads(ln) for ln in raw_lines if ln.strip()]
        parsed_entries[1], parsed_entries[2] = parsed_entries[2], parsed_entries[1]
        rehash_merkle = MerkleReceiptLog()
        rebuilt: list[str] = []
        for new_seq, item in enumerate(parsed_entries):
            rcpt = ToolReceipt.model_validate(item["receipt"])
            new_chain = rehash_merkle.append(rcpt)
            proof = rehash_merkle.inclusion_proof(new_seq)
            item["seq"] = new_seq
            item["leaf_sha256"] = proof.leaf_sha256
            item["chain_sha256"] = new_chain
            item["merkle_root"] = proof.root_sha256
            item.pop("prev_chain_sha256", None)
            rebuilt.append(json.dumps(item, sort_keys=True, separators=(",", ":")))
        reordered_path.write_text("\n".join(rebuilt) + "\n", encoding="utf-8")
        reorder_rejected = 0
        try:
            verify_receipt_log_file(reordered_path, pubkey_file=pub_path)
        except (ProvenanceError, SignatureTamperError, MerkleTamperError):
            reorder_rejected = 1

        # 3. Canonical JSON non-finite float rejection (NaN and +Infinity)
        nonfinite_rejected = 0
        for bad_val in (float("nan"), float("inf")):
            bad_rcpt = ToolReceipt(
                call_id="nan-check",
                tool="metric_op",
                args={"score": bad_val},
                ok=True,
                executed=True,
            )
            try:
                canonical_receipt_bytes(bad_rcpt)
            except (SignatureTamperError, ValueError):
                nonfinite_rejected += 1

    adv_total = 1 + 1 + 2
    adv_blocked = replay_rejected + reorder_rejected + nonfinite_rejected
    return {
        "n_workers": n_workers,
        "n_concurrent_appends": total_concurrent,
        "concurrent_verified_entries": ver.entry_count,
        "concurrent_verified_pct": round(100.0 * ver.entry_count / total_concurrent, 1),
        "replay_reorder_nan_trials": adv_total,
        "replay_reorder_nan_rejected": adv_blocked,
        "replay_reorder_nan_pct": round(100.0 * adv_blocked / adv_total, 1),
    }


def _bench_syndication_dedup() -> dict[str, int | float]:
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

    # Paraphrase sensitivity sweep across 4 rewrite intensities (n=20 clusters x 3 URLs each)
    n_sweep = 20
    t0_collapsed = 0
    t1_collapsed = 0
    t2_collapsed = 0
    t3_collapsed = 0
    for idx in range(n_sweep):
        origin = (
            f"Security advisory {idx}: version 4.2 of gateway-{idx} contains an unauthenticated "
            f"buffer overflow in packet parsing that permits remote code execution over port 8443."
        )
        # Tier 0: wire dateline + boilerplate wrapper (0% core lexical substitution)
        m0_a = f"(REUTERS) - {origin}"
        m0_b = f"{origin} Read more in our daily security briefing."
        # Tier 1: light lexical edit (~15% word substitution, preserving 4-gram runs)
        m1_a = (
            f"Urgent advisory {idx}: version 4.2 of gateway-{idx} contains an unauthenticated "
            f"buffer overflow in packet parsing that allows remote code execution on port 8443."
        )
        m1_b = (
            f"Security bulletin {idx}: version 4.2 of gateway-{idx} contains an unauthenticated "
            f"buffer overflow in packet parsing that permits remote command execution on port 8443."
        )
        # Tier 2: moderate clause & synonym rewrite (~45% lexical substitution)
        m2_a = (
            f"Gateway-{idx} release 4.2 suffers from a pre-auth memory overflow during packet "
            f"handling, allowing attackers to run arbitrary code via TCP port 8443."
        )
        m2_b = (
            f"An unauthenticated packet-parser overflow flaw in gateway-{idx} v4.2 lets remote "
            f"actors execute code across network port 8443."
        )
        # Tier 3: heavy abstractive rewrite (>80% lexical replacement, zero shared 4-grams)
        m3_a = (
            f"Unpatched deployments of gateway-{idx} (build 4.2) expose port 8443 to full system "
            f"takeover when malformed frames trigger memory corruption before login."
        )
        m3_b = (
            f"Attackers targeting TCP/8443 on gateway-{idx} 4.2 can hijack host processes without "
            f"credentials by sending crafted network traffic to the defective decoder."
        )
        for tier_idx, (ma, mb) in enumerate(
            ((m0_a, m0_b), (m1_a, m1_b), (m2_a, m2_b), (m3_a, m3_b))
        ):
            ded, _ = deduplicate_syndicated_sources(
                [(f"orig-{idx}.org", origin), (f"mir1-{idx}.net", ma), (f"mir2-{idx}.io", mb)]
            )
            if len(ded) == 1:
                if tier_idx == 0:
                    t0_collapsed += 1
                elif tier_idx == 1:
                    t1_collapsed += 1
                elif tier_idx == 2:
                    t2_collapsed += 1
                else:
                    t3_collapsed += 1

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
        "paraphrase_sweep_clusters_per_tier": n_sweep,
        "tier0_wrapper_collapse_pct": round(100.0 * t0_collapsed / n_sweep, 1),
        "tier1_light_edit_collapse_pct": round(100.0 * t1_collapsed / n_sweep, 1),
        "tier2_moderate_rewrite_collapse_pct": round(100.0 * t2_collapsed / n_sweep, 1),
        "tier3_abstractive_rewrite_collapse_pct": round(100.0 * t3_collapsed / n_sweep, 1),
    }


def main() -> None:
    payload = {
        "merkle_receipts": _bench_merkle_and_tamper(),
        "aer1_interop": _bench_aer1_interop(),
        "mcp_proxy_wire": _bench_mcp_proxy_wire(),
        "adversarial_concurrency": _bench_adversarial_replay_and_concurrency(),
        "sqlite_provenance_gate": _bench_sqlite_provenance_gate(),
        "syndication_dedup": _bench_syndication_dedup(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
