"""AER-1 single-receipt core and binding conformance tests for `ToolReceipt.from_aer1`."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from epistemic_guard import (
    SignatureTamperError,
    ToolReceipt,
    aer1_anchor_report,
    aer1_merkle_root,
    export_receipts_to_aer1_chain_v07,
    load_aer1_json,
    verify_aer1_chain,
    verify_aer1_chain_commitment,
    verify_aer1_chain_v07,
    verify_aer1_receipt,
    verify_aer1_unicode_vector,
    verify_aer1_workflow,
    verify_aer1_workflow_merkle_vector,
)


def _make_valid_aer1(
    *,
    inner_overrides: dict[str, Any] | None = None,
    outer_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inner: dict[str, Any] = {
        "inputs": {"b": 2, "a": 1},
        "issuer_claims": {"provenance_class": "EXECUTED BY ZAMBO"},
        "output": {"text": "ok"},
        "receipt_schema_version": "0.4",
        "tool": {"name": "live_price", "scope": "public", "version": "4.1.4"},
    }
    if inner_overrides:
        inner.update(inner_overrides)
    raw = json.dumps(inner, sort_keys=True, separators=(",", ":")).encode("utf-8")
    inputs_raw = json.dumps(inner["inputs"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    rec: dict[str, Any] = {
        "id": "8869c25f-a500-4502-ac54-6b208f052a33",
        "receipt_schema_version": "0.4",
        "created_at": "2024-02-29T12:30:09.264Z",
        "tool": {"name": "live_price", "scope": "public", "version": "4.1.4"},
        "provenance_class": "EXECUTED BY ZAMBO",
        "canonical_bytes": base64.b64encode(raw).decode("ascii"),
        "output_hash": f"sha256:{hashlib.sha256(raw).hexdigest()}",
        "input_hash": f"sha256:{hashlib.sha256(inputs_raw).hexdigest()}",
        "verification_status": "verified",
    }
    if outer_overrides:
        rec.update(outer_overrides)
    return rec


def test_aer1_valid_core_and_legacy_shapes() -> None:
    rec = _make_valid_aer1()
    parsed = ToolReceipt.from_aer1(rec)
    assert parsed.tool == "live_price"
    assert parsed.args == {"a": 1, "b": 2}
    assert parsed.ok is True

    # Profile-boundary payload without `inputs` MUST still pass core interop check
    inner_no_inputs = {
        "issuer_claims": {"provenance_class": "EXECUTED BY ZAMBO"},
        "output": {"text": "ok"},
        "receipt_schema_version": "0.4",
        "tool": {"name": "live_price", "scope": "public", "version": "4.1.4"},
    }
    raw_no_in = json.dumps(inner_no_inputs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    rec_no_in = _make_valid_aer1(
        outer_overrides={
            "canonical_bytes": base64.b64encode(raw_no_in).decode("ascii"),
            "output_hash": f"sha256:{hashlib.sha256(raw_no_in).hexdigest()}",
        }
    )
    del rec_no_in["input_hash"]
    assert ToolReceipt.from_aer1(rec_no_in).tool == "live_price"


def test_aer1_structural_and_calendar_rejections() -> None:
    missing_oh = _make_valid_aer1()
    del missing_oh["output_hash"]
    with pytest.raises(SignatureTamperError, match="missing required member: output_hash"):
        ToolReceipt.from_aer1(missing_oh)

    for bad_date in (
        "1900-02-29T00:00:00Z",
        "2023-02-29T00:00:00Z",
        "2026-02-30T09:00:00Z",
        "2026-06-30T23:59:60Z",
        "2026-06-15T12:00:00+25:00",
    ):
        with pytest.raises(
            SignatureTamperError, match="created_at is not a valid calendar date/time"
        ):
            ToolReceipt.from_aer1(_make_valid_aer1(outer_overrides={"created_at": bad_date}))

    for non_rfc in ("2026-06-15t12:00:00z", "2026-06-15T12:00:00", "not-a-date"):
        with pytest.raises(SignatureTamperError, match="created_at is not RFC 3339"):
            ToolReceipt.from_aer1(_make_valid_aer1(outer_overrides={"created_at": non_rfc}))

    with pytest.raises(SignatureTamperError, match="id is the nil UUID"):
        ToolReceipt.from_aer1(
            _make_valid_aer1(outer_overrides={"id": "00000000-0000-0000-0000-000000000000"})
        )

    for bad_id in (
        "8869C25F-A500-4502-AC54-6B208F052A33",
        "8869c25f-a500-4502-ac54-6b208f052a33\n",
        "not-a-uuid",
    ):
        with pytest.raises(SignatureTamperError, match="id is not a lowercase UUID"):
            ToolReceipt.from_aer1(_make_valid_aer1(outer_overrides={"id": bad_id}))


def test_aer1_base64_utf8_duplicate_and_binding_rejections() -> None:
    # Non-canonical base64 pad bit for 1-byte payload b"{" -> canonical "ew==", non-canonical "ex=="
    non_canon_b64 = "ex=="
    raw_brace = base64.b64decode(non_canon_b64)
    with pytest.raises(
        SignatureTamperError, match="canonical_bytes is not the canonical base64 spelling"
    ):
        ToolReceipt.from_aer1(
            _make_valid_aer1(
                outer_overrides={
                    "canonical_bytes": non_canon_b64,
                    "output_hash": f"sha256:{hashlib.sha256(raw_brace).hexdigest()}",
                }
            )
        )

    # Invalid UTF-8 inside canonical_bytes
    bad_utf8 = b"\xff\xfe"
    with pytest.raises(SignatureTamperError, match="canonical_bytes is not valid UTF-8"):
        ToolReceipt.from_aer1(
            _make_valid_aer1(
                outer_overrides={
                    "canonical_bytes": base64.b64encode(bad_utf8).decode("ascii"),
                    "output_hash": f"sha256:{hashlib.sha256(bad_utf8).hexdigest()}",
                }
            )
        )

    # Duplicate JSON member name when parsed from raw JSON text/bytes
    valid = _make_valid_aer1()
    raw_dup = (
        json.dumps(valid)[:-1]
        + f', "output_hash": "{valid["output_hash"]}"}}'
    )
    with pytest.raises(SignatureTamperError, match="duplicate member name: output_hash"):
        ToolReceipt.from_aer1(raw_dup)

    # Input-hash non-canonical key order rejection
    wrong_order = b'{"b":2,"a":1}'
    with pytest.raises(SignatureTamperError, match="input_hash does not match"):
        ToolReceipt.from_aer1(
            _make_valid_aer1(
                outer_overrides={"input_hash": f"sha256:{hashlib.sha256(wrong_order).hexdigest()}"}
            )
        )

    # Version downgrade & tampered provenance binding
    with pytest.raises(SignatureTamperError, match="receipt_schema_version: outer"):
        ToolReceipt.from_aer1(_make_valid_aer1(outer_overrides={"receipt_schema_version": "0.3"}))

    with pytest.raises(SignatureTamperError, match="provenance binding: outer"):
        ToolReceipt.from_aer1(
            _make_valid_aer1(outer_overrides={"provenance_class": "LOGGED BY AGENT"})
        )

    # Control char / trailing newline in provenance_class
    for bad_pc in ("EXECUTED BY ZAMBO\x01", "EXECUTED BY ZAMBO\n", "UNKNOWN CLASS"):
        with pytest.raises(SignatureTamperError, match="provenance_class is not a known class"):
            ToolReceipt.from_aer1(_make_valid_aer1(outer_overrides={"provenance_class": bad_pc}))

    # Optional external ID binding check
    with pytest.raises(SignatureTamperError, match="id mismatch"):
        ToolReceipt.from_aer1(
            valid,
            expected_id="11111111-2222-4333-8444-555555555555",
        )


def test_aer1_full_146_vector_conformance_corpus() -> None:
    """Run the pinned `draft-zambo-aer1-26` corpus: 146 checks counted by this test plus 8 metadata-probe checks (154 total). The 146 is this repo's own tally, not the published kit's count."""
    corpus_path = Path(__file__).resolve().parents[1] / "benchmarks" / "aer1_full_corpus.json"
    bundle = json.loads(corpus_path.read_text(encoding="utf-8"))
    fb: dict[str, str] = bundle["files_b64"]

    passed = 0
    total = 0

    def _assert_check(ok: bool, msg: str) -> None:
        nonlocal passed, total
        total += 1
        assert ok, msg
        passed += 1

    for k, b64 in sorted(fb.items()):
        raw = base64.b64decode(b64)
        if k.startswith("test-vectors/valid/"):
            rep = verify_aer1_receipt(raw)
            _assert_check(rep.valid, f"{k}: {rep.failures}")
        elif k.startswith("test-vectors/invalid/"):
            rep = verify_aer1_receipt(raw)
            _assert_check(not rep.valid, f"{k} unexpectedly passed")
        elif k.startswith("test-vectors/invalid-profile/"):
            rep = verify_aer1_receipt(raw)
            _assert_check(
                rep.valid and not rep.reference_profile_valid,
                f"{k}: valid={rep.valid} profile={rep.reference_profile_valid}",
            )
        elif k.startswith("test-vectors/anchored/"):
            rep = verify_aer1_receipt(raw)
            fname = k.rsplit("/", 1)[1]
            exp_out = (
                "verified"
                if fname == "anchored-valid.json"
                else (
                    "rejected: unresolvable record"
                    if fname == "anchored-unresolvable-anchor.json"
                    else "verification failed"
                )
            )
            _assert_check(
                rep.valid and rep.anchor.outcome == exp_out,
                f"{k}: got {rep.anchor.outcome}, expected {exp_out}",
            )
        elif k.startswith("tests/metadata-probes/") and not k.endswith("laurent-receipt.json"):
            rep = verify_aer1_receipt(raw)
            _assert_check(
                not rep.valid and any("metadata_hash" in f for f in rep.failures),
                f"metadata probe {k} not rejected: {rep.failures}",
            )

    merkle_vecs = json.loads(base64.b64decode(fb["merkle-vectors.json"]).decode("utf-8"))["vectors"]
    for v in merkle_vecs:
        name = v["name"]
        if "receipt_ids" in v:
            got = aer1_merkle_root(v["receipt_ids"])
            _assert_check(got == v["expected_root"], f"merkle {name}: {got}")
        elif "leaves" in v:
            leaves = [
                hashlib.sha256(lf["preimage"].encode("utf-8")).digest()
                for lf in sorted(v["leaves"], key=lambda x: x["seq"])
            ]
            lvl = list(leaves)
            while len(lvl) > 1:
                if len(lvl) % 2:
                    lvl.append(lvl[-1])
                lvl = [
                    hashlib.sha256(lvl[i] + lvl[i + 1]).digest()
                    for i in range(0, len(lvl), 2)
                ]
            got = lvl[0].hex() if lvl else hashlib.sha256(b"").hexdigest()
            _assert_check(got == v["expected_root"], f"legacy merkle {name}: {got}")
        if v.get("expected_workflow_verdict") is not None:
            wf = {
                "type": "verifiable-workflow-receipt",
                "version": "1",
                "workflow_id": "11111111-1111-1111-1111-111111111111",
                "receipt_id": "22222222-2222-2222-2222-222222222222",
                "session_id": "test-session",
                "goal": "test workflow",
                "status": "ok",
                "steps": [
                    {
                        "seq": i + 1,
                        "tool": "test-tool",
                        "receipt_id": rid,
                        "receipt_hash": "a" * 64,
                        "started_at": "2026-01-01T00:00:00Z",
                        "ended_at": "2026-01-01T00:00:01Z",
                        "status": "ok",
                    }
                    for i, rid in enumerate(v["receipt_ids"])
                ],
                "merkle_root": v["expected_root"],
                "output_hash": "b" * 64,
                "verify_url": "https://example.com/workflow/test",
            }
            wf_fails = verify_aer1_workflow(wf)
            verdict = "valid" if not wf_fails else "invalid"
            ok = verdict == v["expected_workflow_verdict"]
            if ok and verdict == "invalid":
                rb = v.get("rejected_by", "")
                ok = any(rb in f for f in wf_fails)
            _assert_check(ok, f"workflow {name}: {wf_fails}")

    for v in json.loads(base64.b64decode(fb["chain-vectors.json"]).decode("utf-8"))["vectors"]:
        tl = v.get("timeline") if v.get("timeline") is not None else v.get("entries")
        fs = verify_aer1_chain(tl)
        _assert_check((not fs) == (v["expected_verdict"] == "valid"), f"chain {v['name']}: {fs}")

    for v in json.loads(base64.b64decode(fb["chain-vectors-v07.json"]).decode("utf-8"))["vectors"]:
        tl = v.get("timeline") if v.get("timeline") is not None else v.get("entries")
        fs = verify_aer1_chain_v07(tl)
        _assert_check(
            (not fs) == (v["expected_verdict"] == "valid"), f"chain-v07 {v['name']}: {fs}"
        )

    for v in json.loads(base64.b64decode(fb["commitment-vectors.json"]).decode("utf-8"))[
        "vectors"
    ]:
        fs = verify_aer1_chain_commitment(v.get("timeline"), v.get("commitment"))
        _assert_check(
            (not fs) == (v["expected_verdict"] == "valid"), f"commitment {v['name']}: {fs}"
        )

    for v in json.loads(base64.b64decode(fb["unicode-vectors.json"]).decode("utf-8"))["vectors"]:
        fs = verify_aer1_unicode_vector(v)
        _assert_check(not fs, f"unicode {v['name']}: {fs}")

    for v in json.loads(base64.b64decode(fb["id-binding-vectors.json"]).decode("utf-8"))["vectors"]:
        rep = verify_aer1_receipt(
            v["receipt"],
            expected_id=v.get("expected_id"),
            resolved_record=v.get("resolved_record"),
        )
        ok = (
            rep.digest_valid == (v["expected_digest"] == "pass")
            and rep.id_state == v["expected_id_state"]
            and rep.id_verdict == v["expected_id_verdict"]
            and rep.valid == (v["expected_verdict"] == "pass")
        )
        _assert_check(ok, f"id-binding {v['name']}: {rep}")

    for v in json.loads(base64.b64decode(fb["anchor-vectors.json"]).decode("utf-8"))["vectors"]:
        rec_raw = (
            v["receipt"]
            if isinstance(v.get("receipt"), dict)
            else base64.b64decode(fb[v["file"]])
        )
        loaded, _ = load_aer1_json(rec_raw)
        ar = aer1_anchor_report(loaded, evaluate=v.get("evaluate", True))
        ok = (
            ar.present == v["expected_present"]
            and ar.evaluated == v["expected_evaluated"]
            and ar.outcome == v["expected_anchor_outcome"]
        )
        _assert_check(ok, f"anchor {v['name']}: {ar}")

    for v in json.loads(
        base64.b64decode(fb["workflow-merkle-seq-output-v1.json"]).decode("utf-8")
    )["vectors"]:
        fs = verify_aer1_workflow_merkle_vector(v)
        _assert_check(not fs, f"workflow-merkle {v['name']}: {fs}")

    # Verify exact v1.5.7 evaluated_set_digest across all 72 official corpus files
    v157_entries: list[str] = []
    for k, b64 in sorted(fb.items()):
        if k.startswith("tests/metadata-probes/"):
            continue
        rel = k if k == "workflow-merkle-seq-output-v1.json" else f"base/{k}"
        digest_hex = hashlib.sha256(base64.b64decode(b64)).hexdigest()
        v157_entries.append(f"{rel}\0{digest_hex}\n")
    v157_entries.sort()
    v157_set_digest = "sha256:" + hashlib.sha256("".join(v157_entries).encode("utf-8")).hexdigest()
    assert v157_set_digest == "sha256:60bf73f325793e16843370cfa24903bb0d017525cb297e42bebc35857e2db0d2"

    # 146 official v1.5.7 checks + 8 metadata-binding tamper probes = 154 checks
    assert total == 154
    assert passed == 154

    # Verify native ToolReceipt -> -07 chain + Section 7.3 commitment roundtrip
    sample_receipts = [
        ToolReceipt(tool="write_file", args={"path": "a.txt"}, output="wrote 10B"),
        ToolReceipt(tool="delete_file", args={"path": "b.tmp"}, ok=False, error_code="EACCES"),
    ]
    tl_out, commit_out = export_receipts_to_aer1_chain_v07(sample_receipts, job_id="job-42")
    assert verify_aer1_chain_v07(tl_out) == []
    assert verify_aer1_chain_commitment(tl_out, commit_out) == []
