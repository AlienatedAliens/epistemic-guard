"""AER-1 single-receipt core and binding conformance tests for `ToolReceipt.from_aer1`."""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

import pytest

from epistemic_guard import SignatureTamperError, ToolReceipt


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
