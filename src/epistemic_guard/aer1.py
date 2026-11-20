"""Self-contained interop codec & full conformance verifier for IETF `draft-zambo-aer1-26`.

Covers Sections 3-10: single-receipt core + Section 3.1 `metadata_hash`, reference-producer
profile (`inputs` member), Section 3.3/6 transparency-log anchor reporting (5 states),
Section 4 raw-byte Unicode commitment (no NFC/NFD normalization), Section 6/13 ID-binding
verdicts, Section 7 chains (`-06` and hardened `-07` with `close: true` suffix-truncation
kill), Section 7.3 external chain commitments, and Section 8.1/8.2 workflow receipts (`MM-1`
duplicate-ID rejection), plus lossless export from native `ToolReceipt` sequences.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from os import PathLike
from pathlib import Path
from typing import Any, Final

from epistemic_guard.exceptions import SignatureTamperError
from epistemic_guard.schemas import (
    ToolReceipt,
    _aer1_metadata_block,
    _aer1_metadata_digest,
    _is_aer1_legacy_production_shape,
    _is_strict_rfc3339,
    _json_quote,
    _parse_json_with_duplicate_check,
)

_CORE_FIELDS: Final[tuple[str, ...]] = (
    "id",
    "receipt_schema_version",
    "created_at",
    "tool",
    "provenance_class",
    "canonical_bytes",
    "output_hash",
    "verification_status",
)
_UUID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z"
)
_STRICT_UUID_V4_RE: Final[re.Pattern[str]] = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"
)
_RFC3339_RE: Final[re.Pattern[str]] = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?(Z|[+-][0-9]{2}:[0-9]{2})\Z"
)
_HASH_RE: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}\Z")
_HEX64_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}\Z")
_PROV_RE: Final[re.Pattern[str]] = re.compile(
    r"^(EXECUTED BY [A-Z0-9][A-Z0-9 ._-]*|OBSERVED VIA GATEWAY|LOGGED BY AGENT)\Z"
)
_B64_RE: Final[re.Pattern[str]] = re.compile(r"\A[A-Za-z0-9+/]*={0,2}\Z")
_NIL_UUID: Final[str] = "00000000-0000-0000-0000-000000000000"
_ANCHOR_PROFILE_MEMBERS: Final[frozenset[str]] = frozenset(
    {"log", "leaf_hash", "anchored_at", "proof"}
)


@dataclass(frozen=True)
class Aer1AnchorReport:
    """5-state anchor verification report (`draft-zambo-aer1` Section 3.3 & Section 6 step 5)."""

    present: bool
    evaluated: bool
    outcome: str

    def as_dict(self) -> dict[str, bool | str]:
        return {
            "present": self.present,
            "evaluated": self.evaluated,
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class Aer1ReceiptReport:
    """Comprehensive verification verdict for a single AER-1 receipt artifact."""

    valid: bool
    digest_valid: bool
    is_legacy: bool
    id_state: str
    id_verdict: str
    anchor: Aer1AnchorReport
    reference_profile_valid: bool
    failures: tuple[str, ...]


def load_aer1_json(
    payload: Mapping[str, Any] | str | bytes | PathLike[str],
) -> tuple[object, str | None]:
    """Load an AER-1 JSON document while detecting RFC 8259 duplicate member names."""
    if isinstance(payload, Mapping):
        return dict(payload), None
    if isinstance(payload, PathLike):
        raw_text = Path(payload).read_text(encoding="utf-8")
    elif isinstance(payload, bytes):
        try:
            raw_text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SignatureTamperError("receipt is not valid UTF-8") from exc
    elif isinstance(payload, str):
        raw_text = payload
    else:
        return payload, None
    return _parse_json_with_duplicate_check(raw_text)


def check_aer1_core(
    receipt: object,
    *,
    duplicate_member: str | None = None,
) -> list[str]:
    """Verify Section 3-6 core shape, base64/UTF-8 canonicalization, `output_hash`,
    `metadata_hash`, and conditional `input_hash` rules. Returns failure strings (`[]` = pass).
    """
    failures: list[str] = []
    if not isinstance(receipt, Mapping):
        return ["receipt is not a JSON object"]
    legacy_prod = _is_aer1_legacy_production_shape(receipt)
    required: Sequence[str] = (
        tuple(f for f in _CORE_FIELDS if f != "created_at") + ("timestamp", "tool_context")
        if legacy_prod
        else _CORE_FIELDS
    )
    for field_name in required:
        if field_name not in receipt:
            failures.append(f"missing required member: {field_name}")
    if failures:
        if duplicate_member is not None:
            failures.append(f"duplicate member name: {duplicate_member}")
        return failures

    rid = receipt["id"]
    if not isinstance(rid, str) or not _UUID_RE.match(rid):
        failures.append("id is not a lowercase UUID")
    elif rid == _NIL_UUID:
        failures.append("id is the nil UUID")

    rsv = receipt["receipt_schema_version"]
    if not isinstance(rsv, str) or not rsv:
        failures.append("receipt_schema_version is not a non-empty string")

    created_val = receipt["timestamp"] if legacy_prod else receipt["created_at"]
    if not isinstance(created_val, str):
        failures.append("created_at is not a string")
    elif not _RFC3339_RE.match(created_val):
        failures.append("created_at is not RFC 3339")
    elif not _is_strict_rfc3339(created_val):
        failures.append("created_at is not a valid calendar date/time")

    tool = receipt["tool_context"] if legacy_prod else receipt["tool"]
    if not isinstance(tool, Mapping) or not all(
        isinstance(tool.get(k), str) and bool(tool[k]) for k in ("name", "version", "scope")
    ):
        failures.append("tool is not an object with name/version/scope strings")

    prov = receipt["provenance_class"]
    if not isinstance(prov, str) or not _PROV_RE.match(prov):
        failures.append("provenance_class is not a known class")

    cb = receipt["canonical_bytes"]
    raw: bytes | None = None
    if not isinstance(cb, str) or not _B64_RE.match(cb):
        failures.append("canonical_bytes is not valid base64")
    else:
        try:
            raw = base64.b64decode(cb, validate=True)
        except (binascii.Error, ValueError):
            failures.append("canonical_bytes is not valid base64")
        else:
            if base64.b64encode(raw).decode("ascii") != cb:
                failures.append("canonical_bytes is not the canonical base64 spelling")
            try:
                canon_text = raw.decode("utf-8")
            except UnicodeDecodeError:
                failures.append("canonical_bytes is not valid UTF-8")
                raw = None
            else:
                try:
                    _inner, inner_dup = _parse_json_with_duplicate_check(canon_text)
                    if inner_dup is not None:
                        failures.append(f"duplicate member name: {inner_dup}")
                except (json.JSONDecodeError, SignatureTamperError):
                    pass

    oh = receipt["output_hash"]
    if not isinstance(oh, str) or not _HASH_RE.match(oh):
        failures.append("output_hash is not sha256: + lowercase hex digest")
    elif raw is not None:
        want_oh = f"sha256:{hashlib.sha256(raw).hexdigest()}"
        if oh != want_oh:
            failures.append("output_hash does not match sha256(canonical_bytes)")

    mh = receipt.get("metadata_hash")
    if mh is not None:
        if not isinstance(mh, str) or not _HASH_RE.match(mh):
            failures.append("metadata_hash is not sha256: + lowercase hex digest")
        else:
            block = _aer1_metadata_block(receipt)
            if block is None:
                failures.append(
                    "metadata block cannot be recomputed from the stored member values"
                )
            elif mh != _aer1_metadata_digest(block):
                failures.append("metadata_hash does not match the recomputed metadata block")

    ih = receipt.get("input_hash")
    if ih is not None:
        if not isinstance(ih, str) or not _HASH_RE.match(ih):
            failures.append("input_hash is not sha256: + lowercase hex digest")
        elif raw is not None:
            try:
                payload_obj = json.loads(raw.decode("utf-8"))
            except ValueError:
                payload_obj = None
            if isinstance(payload_obj, Mapping) and "inputs" in payload_obj:
                blob = json.dumps(
                    payload_obj["inputs"], sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
                want_ih = f"sha256:{hashlib.sha256(blob).hexdigest()}"
                if ih != want_ih:
                    failures.append(
                        "input_hash does not match sha256(canonical JSON of the recorded inputs)"
                    )

    ver_status = receipt["verification_status"]
    if legacy_prod:
        if not isinstance(ver_status, str) or not ver_status:
            failures.append("verification_status is not a non-empty string")
    elif ver_status != "verified":
        failures.append("verification_status is not verified")

    if duplicate_member is not None:
        failures.append(f"duplicate member name: {duplicate_member}")
    return failures


def check_aer1_version_and_binding(receipt: object) -> tuple[bool, list[str]]:
    """Verify Section 6 version agreement & provenance binding (`(is_legacy, failures)`)."""
    if not isinstance(receipt, Mapping):
        return (False, ["receipt is not a JSON object"])
    inner_obj: object = None
    inner_version: object = None
    try:
        cb = receipt.get("canonical_bytes", "")
        if isinstance(cb, str):
            raw = base64.b64decode(cb, validate=True)
            inner_obj = json.loads(raw.decode("utf-8"))
    except Exception:
        inner_obj = None
    if isinstance(inner_obj, Mapping):
        inner_version = inner_obj.get("receipt_schema_version")
    outer_version = receipt.get("receipt_schema_version")
    legacy_version = inner_version if inner_version is not None else outer_version
    if legacy_version is not None and str(legacy_version) in {"0.2", "0.3"}:
        return (True, [])

    failures: list[str] = []
    if inner_version is not None and outer_version is not None:
        if str(inner_version) != str(outer_version):
            failures.append(
                f"receipt_schema_version: outer {_json_quote(outer_version)} "
                f"does not match committed {_json_quote(inner_version)}"
            )
    else:
        failures.append(
            f"receipt_schema_version: one side missing "
            f"(outer={_json_quote(outer_version)} committed={_json_quote(inner_version)})"
        )

    if isinstance(inner_obj, Mapping):
        issuer_claims = inner_obj.get("issuer_claims")
        committed = (
            issuer_claims.get("provenance_class")
            if isinstance(issuer_claims, Mapping)
            else None
        )
        outer_pc = receipt.get("provenance_class")
        if committed is not None and outer_pc is not None:
            if committed != outer_pc:
                failures.append(
                    f"provenance binding: outer {_json_quote(outer_pc)} "
                    f"does not match committed {_json_quote(committed)}"
                )
        else:
            failures.append(
                f"provenance binding: one side missing "
                f"(outer={_json_quote(outer_pc)} committed={_json_quote(committed)})"
            )
    else:
        failures.append("provenance binding: canonical payload did not decode to an object")
    return (False, failures)


def check_aer1_reference_profile(receipt: object) -> list[str]:
    """Reference-producer profile check: canonical JSON object must carry `'inputs'`."""
    if not isinstance(receipt, Mapping):
        return ["receipt is not a JSON object"]
    cb = receipt.get("canonical_bytes")
    if not isinstance(cb, str):
        return ["canonical_bytes is not a string"]
    try:
        raw = base64.b64decode(cb, validate=True)
    except (binascii.Error, ValueError):
        return ["canonical_bytes is not valid base64"]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return ["canonical_bytes is not valid UTF-8"]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return ["canonical payload is not JSON"]
    if not isinstance(payload, Mapping):
        return ["canonical payload is not a JSON object"]
    if "inputs" not in payload:
        return ["reference-producer payload omits the 'inputs' member"]
    return []


def check_aer1_disinterested_tier(receipt: object) -> list[str]:
    """Verify optional Section 3.3 transparency-log anchor profile & strict UUIDv4."""
    if not isinstance(receipt, Mapping):
        return ["anchor is not an object"]
    failures: list[str] = []
    rid = receipt.get("id")
    if not isinstance(rid, str) or not _STRICT_UUID_V4_RE.match(rid):
        failures.append("id is not a UUID v4 (strict profile)")
    anchor = receipt.get("anchor")
    if not isinstance(anchor, Mapping):
        return ["anchor is not an object"]
    if set(anchor.keys()) != _ANCHOR_PROFILE_MEMBERS:
        failures.append("anchor has members other than log/leaf_hash/anchored_at/proof")
    log_uri = anchor.get("log")
    if not isinstance(log_uri, str) or not log_uri:
        failures.append("anchor.log is not a non-empty string")
    leaf = anchor.get("leaf_hash")
    if not isinstance(leaf, str) or not _HASH_RE.match(leaf):
        failures.append("anchor.leaf_hash is not sha256: + lowercase hex digest")
    else:
        cb = receipt.get("canonical_bytes")
        raw: bytes | None = None
        if not isinstance(cb, str) or not _B64_RE.match(cb):
            failures.append("canonical_bytes is not valid base64")
        else:
            try:
                raw = base64.b64decode(cb, validate=True)
                raw.decode("utf-8")
            except (binascii.Error, ValueError, UnicodeDecodeError):
                raw = None
                failures.append("canonical_bytes is not valid UTF-8 base64")
        if raw is not None:
            want_leaf = f"sha256:{hashlib.sha256(raw).hexdigest()}"
            if leaf != want_leaf:
                failures.append("anchor.leaf_hash does not match canonical_bytes")
    anchored_at = anchor.get("anchored_at")
    if not isinstance(anchored_at, str) or not _RFC3339_RE.match(anchored_at):
        failures.append("anchor.anchored_at is not RFC 3339")
    elif not _is_strict_rfc3339(anchored_at):
        failures.append("anchor.anchored_at is not a valid calendar date/time")
    if not isinstance(anchor.get("proof"), Mapping):
        failures.append("anchor.proof is not an object")
    return failures


def _uri_host(uri: str) -> str | None:
    m = re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://([^/?#]+)", uri)
    if not m:
        return None
    host = m.group(1).rsplit("@", 1)[-1].split(":", 1)[0].lower()
    return host or None


def _anchor_resolvable(receipt: object) -> bool:
    if not isinstance(receipt, Mapping):
        return True
    anchor = receipt.get("anchor")
    if not isinstance(anchor, Mapping):
        return True
    log_uri = anchor.get("log")
    if not isinstance(log_uri, str) or not log_uri:
        return True
    host = _uri_host(log_uri)
    if host is None:
        return True
    return not host.endswith(".invalid")


def aer1_anchor_report(receipt: object, *, evaluate: bool = True) -> Aer1AnchorReport:
    """Compute the 5-state machine-visible anchor report (`draft-zambo-aer1` Section 6 step 5)."""
    present = bool(
        isinstance(receipt, Mapping)
        and isinstance(receipt.get("anchor"), Mapping)
        and bool(set(receipt["anchor"].keys()) & _ANCHOR_PROFILE_MEMBERS)
    )
    if not present:
        return Aer1AnchorReport(present=False, evaluated=False, outcome="absent")
    if not evaluate:
        return Aer1AnchorReport(present=True, evaluated=False, outcome="not checked")
    tier_fails = check_aer1_disinterested_tier(receipt)
    if not _anchor_resolvable(receipt):
        outcome = "rejected: unresolvable record"
    elif tier_fails:
        outcome = "verification failed"
    else:
        outcome = "verified"
    return Aer1AnchorReport(present=True, evaluated=True, outcome=outcome)


def evaluate_aer1_id_binding(
    receipt: object,
    *,
    expected_id: str | None = None,
    resolved_record: Mapping[str, Any] | None = None,
) -> tuple[str, str, list[str]]:
    """Evaluate Section 6/13 ID binding; returns `(id_state, id_verdict, failures)`."""
    rid = receipt.get("id") if isinstance(receipt, Mapping) else None
    if expected_id is None and resolved_record is None:
        id_verdict = "absent" if rid is None else "unverified"
        return ("unverified", id_verdict, [])

    failures: list[str] = []
    state: str | None = None
    if expected_id is not None:
        if rid != expected_id:
            failures.append(
                f"id mismatch: artifact id {rid!r} does not match "
                f"the externally supplied expected id {expected_id!r}"
            )
            state = "mismatch"
        else:
            state = "verified-expected"

    if resolved_record is not None:
        rrec: object = resolved_record
        if isinstance(rrec, Mapping) and isinstance(rrec.get("receipt"), Mapping):
            rrec = rrec["receipt"]
        if not isinstance(rrec, Mapping):
            failures.append("resolved record is not a JSON object")
            state = "mismatch"
        else:
            r_digest = rrec.get("output_hash")
            r_id = rrec.get("id")
            art_digest = receipt.get("output_hash") if isinstance(receipt, Mapping) else None
            if r_digest != art_digest:
                failures.append(
                    f"id mismatch: resolved record digest {r_digest!r} disagrees "
                    f"with the artifact's committed digest {art_digest!r}"
                )
                state = "mismatch"
            elif r_id is not None and r_id != rid:
                failures.append(
                    f"id mismatch: resolved record carries id {r_id!r}, "
                    f"not the artifact's id {rid!r}"
                )
                state = "mismatch"
            elif not failures:
                state = "verified-resolved"

    final_state = state if state is not None else "unverified"
    if not isinstance(receipt, Mapping) or rid is None:
        verdict = "absent"
    else:
        verdict = {
            "unverified": "unverified",
            "verified-expected": "matched",
            "verified-resolved": "resolved",
            "mismatch": "mismatched",
        }.get(final_state, "unverified")
    return (final_state, verdict, failures)


def verify_aer1_receipt(
    payload: Mapping[str, Any] | str | bytes | PathLike[str],
    *,
    expected_id: str | None = None,
    resolved_record: Mapping[str, Any] | None = None,
    evaluate_anchor: bool = True,
) -> Aer1ReceiptReport:
    """Verify a single AER-1 receipt (core, binding, ID state, anchor report, and profile)."""
    try:
        loaded, dup_member = load_aer1_json(payload)
    except (json.JSONDecodeError, SignatureTamperError, OSError) as exc:
        return Aer1ReceiptReport(
            valid=False,
            digest_valid=False,
            is_legacy=False,
            id_state="unverified",
            id_verdict="absent",
            anchor=Aer1AnchorReport(present=False, evaluated=False, outcome="absent"),
            reference_profile_valid=False,
            failures=(str(exc),),
        )

    core_fails = check_aer1_core(loaded, duplicate_member=dup_member)
    is_legacy, bind_fails = check_aer1_version_and_binding(loaded)
    digest_ok = not (core_fails or bind_fails)
    id_state, id_verdict, id_fails = evaluate_aer1_id_binding(
        loaded, expected_id=expected_id, resolved_record=resolved_record
    )
    anchor_rep = aer1_anchor_report(loaded, evaluate=evaluate_anchor)
    prof_fails = check_aer1_reference_profile(loaded)
    all_failures = tuple(core_fails + bind_fails + id_fails)
    return Aer1ReceiptReport(
        valid=digest_ok and not id_fails,
        digest_valid=digest_ok,
        is_legacy=is_legacy,
        id_state=id_state,
        id_verdict=id_verdict,
        anchor=anchor_rep,
        reference_profile_valid=not prof_fails,
        failures=all_failures,
    )


def verify_aer1_unicode_vector(vector: Mapping[str, Any]) -> list[str]:
    """Section 4 raw-byte Unicode check (NFC vs NFD payloads must never be normalized)."""
    expected = vector.get("expected")
    if expected not in {"match", "mismatch"}:
        return [f"unknown expected outcome: {expected!r}"]
    b64 = vector.get("payload_base64")
    if not isinstance(b64, str):
        return ["payload_base64 is not a string"]
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        return ["payload_base64 is not valid base64"]
    failures: list[str] = []
    if raw.hex() != vector.get("payload_utf8_hex"):
        failures.append("decoded payload bytes do not match payload_utf8_hex")
    digest = hashlib.sha256(raw).hexdigest()
    if expected == "match" and digest != vector.get("sha256"):
        failures.append("sha256(payload) does not match the stated commitment")
    if expected == "mismatch" and digest == vector.get("sha256"):
        failures.append("sha256(payload) matches a commitment it MUST NOT match")
    return failures


def aer1_merkle_root(receipt_ids: Sequence[str]) -> str:
    """Section 8.1 canonical Merkle root over UTF-8 `receipt_id` strings (odd leaf duplicated)."""
    if not receipt_ids:
        return hashlib.sha256(b"").hexdigest()
    level = [hashlib.sha256(rid.encode("utf-8")).digest() for rid in receipt_ids]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        level = [
            hashlib.sha256(level[i] + level[i + 1]).digest()
            for i in range(0, len(level), 2)
        ]
    return level[0].hex()


def verify_aer1_workflow(workflow: object) -> list[str]:
    """Section 8.2 offline workflow receipt verification (including `MM-1` duplicate-ID rule)."""
    if not isinstance(workflow, Mapping):
        return ["workflow is not a JSON object"]
    failures: list[str] = []
    if workflow.get("type") != "verifiable-workflow-receipt":
        failures.append('workflow type is not "verifiable-workflow-receipt"')
    ver = workflow.get("version")
    if not isinstance(ver, str) or not ver:
        failures.append("workflow version is not a non-empty string")
    wid = workflow.get("workflow_id")
    if not isinstance(wid, str) or not _UUID_RE.match(wid):
        failures.append("workflow workflow_id is not a lowercase UUID")
    wrid = workflow.get("receipt_id")
    if not isinstance(wrid, str) or not _UUID_RE.match(wrid):
        failures.append("workflow receipt_id is not a lowercase UUID")
    if not isinstance(workflow.get("session_id"), str) or not workflow.get("session_id"):
        failures.append("workflow session_id is not a non-empty string")
    if not isinstance(workflow.get("goal"), str) or not workflow.get("goal"):
        failures.append("workflow goal is not a non-empty string")
    if not isinstance(workflow.get("status"), str) or not workflow.get("status"):
        failures.append("workflow status is not a non-empty string")
    oh = workflow.get("output_hash")
    if not isinstance(oh, str) or not _HEX64_RE.match(oh):
        failures.append("workflow output_hash is not a 64-char lowercase hex SHA-256 digest")
    vu = workflow.get("verify_url")
    if not isinstance(vu, str) or not vu.startswith(("http://", "https://")):
        failures.append("workflow verify_url is not an http(s) URL string")

    steps = workflow.get("steps")
    if not isinstance(steps, list) or not steps:
        failures.append("workflow steps is not a non-empty list")
        return failures

    for idx, step in enumerate(steps):
        if not isinstance(step, Mapping):
            failures.append(f"step {idx + 1} is not an object")
            continue
        seq = step.get("seq")
        if isinstance(seq, bool) or not (
            isinstance(seq, int) or (isinstance(seq, float) and seq.is_integer())
        ):
            failures.append(f"step {idx + 1} seq is not an integer")
        if not isinstance(step.get("receipt_id"), str):
            failures.append(f"step {idx + 1} receipt_id is not a string")
        if not isinstance(step.get("tool"), str) or not step.get("tool"):
            failures.append(f"step {idx + 1} tool is not a non-empty string")
        rh = step.get("receipt_hash")
        if not isinstance(rh, str) or not _HEX64_RE.match(rh):
            failures.append(
                f"step {idx + 1} receipt_hash is not a 64-char lowercase hex SHA-256 digest"
            )
        sa = step.get("started_at")
        if not isinstance(sa, str) or not _is_strict_rfc3339(sa):
            failures.append(f"step {idx + 1} started_at is not a valid RFC 3339 timestamp")
        ea = step.get("ended_at")
        if not isinstance(ea, str) or not _is_strict_rfc3339(ea):
            failures.append(f"step {idx + 1} ended_at is not a valid RFC 3339 timestamp")
        if not isinstance(step.get("status"), str) or not step.get("status"):
            failures.append(f"step {idx + 1} status is not a non-empty string")

    if failures:
        return failures

    n = len(steps)
    for idx, step in enumerate(steps):
        if int(step["seq"]) != idx + 1:
            failures.append(
                f"step seq values are not 1..{n} in order (index {idx} carries seq {step['seq']})"
            )
            break

    seen: set[str] = set()
    ordered_ids: list[str] = []
    for step in steps:
        rid = str(step["receipt_id"])
        if rid in seen:
            failures.append(f"duplicate receipt_id: {rid}")
            break
        seen.add(rid)
        ordered_ids.append(rid)

    root = aer1_merkle_root(ordered_ids)
    mr = workflow.get("merkle_root")
    if not isinstance(mr, str) or mr != root:
        failures.append("merkle_root does not match the recomputed Section 8.1 root")
    return failures


def verify_aer1_chain(timeline: object) -> list[str]:
    """Section 7 (`-06`) chain mechanics (`prev_digest` links + `close: true` at tail)."""
    if not isinstance(timeline, list) or not timeline:
        return ["timeline is not a non-empty list"]
    failures: list[str] = []
    prev: str | None = None
    for idx, entry in enumerate(timeline):
        if not isinstance(entry, Mapping):
            failures.append(f"entry {idx} is not an object")
            break
        cb = entry.get("canonical_bytes")
        if not isinstance(cb, str) or not _B64_RE.match(cb):
            failures.append(f"entry {idx} canonical_bytes is not valid base64")
            break
        try:
            raw = base64.b64decode(cb, validate=True)
        except (binascii.Error, ValueError):
            failures.append(f"entry {idx} canonical_bytes is not valid base64")
            break
        digest = hashlib.sha256(raw).hexdigest()
        pd = entry.get("prev_digest")
        if idx == 0:
            if not isinstance(pd, str) or pd != "0" * 64:
                failures.append("entry 0 prev_digest is not 64 zero characters")
        elif not isinstance(pd, str) or pd != prev:
            failures.append(f"entry {idx} prev_digest does not match the digest of entry {idx - 1}")
        prev = digest

    last = timeline[-1]
    if isinstance(last, Mapping) and last.get("close") is not True:
        failures.append("last entry does not carry close: true")
    return failures


def aer1_chain_entry_digest_v07(entry: Mapping[str, Any]) -> str:
    """Section 7 (`-07`) hardened chain entry SHA-256 digest."""
    raw = base64.b64decode(str(entry["canonical_bytes"]), validate=True)
    output_hash = hashlib.sha256(raw).hexdigest()
    seq_val = entry["seq"]
    seq_int = int(seq_val) if isinstance(seq_val, float) and seq_val.is_integer() else seq_val
    payload = {
        "prev_digest": entry["prev_digest"],
        "seq": seq_int,
        "job_id": entry["job_id"],
        "close": entry.get("close", False),
        "id": entry["id"],
        "tool": entry["tool"],
        "provenance_class": entry["provenance_class"],
        "output_hash": output_hash,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(blob).hexdigest()


def verify_aer1_chain_v07(timeline: object) -> list[str]:
    """Section 7 (`-07`) hardened chain verification (`seq=1..n`, `job_id`, `close: true`)."""
    if not isinstance(timeline, list) or not timeline:
        return ["timeline is not a non-empty list"]
    failures: list[str] = []
    job: str | None = None
    prev: str | None = None
    for idx, entry in enumerate(timeline):
        if not isinstance(entry, Mapping):
            return [f"entry {idx} is not an object"]
        cb = entry.get("canonical_bytes")
        if not isinstance(cb, str) or not _B64_RE.match(cb):
            return [f"entry {idx} canonical_bytes is not valid base64"]
        try:
            base64.b64decode(cb, validate=True)
        except (binascii.Error, ValueError):
            return [f"entry {idx} canonical_bytes is not valid base64"]
        for member in ("id", "tool", "provenance_class", "job_id", "prev_digest"):
            val = entry.get(member)
            if not isinstance(val, str) or not val:
                return [f"entry {idx} missing {member}"]
        seq_raw = entry.get("seq")
        if isinstance(seq_raw, bool) or not (
            isinstance(seq_raw, int) or (isinstance(seq_raw, float) and seq_raw.is_integer())
        ):
            return [f"entry {idx} seq is not an integer"]
        seq = int(seq_raw)
        if seq != idx + 1:
            failures.append(f"entry {idx} seq {seq} breaks contiguity (expected {idx + 1})")
        if job is None:
            job = str(entry["job_id"])
        elif entry["job_id"] != job:
            failures.append(f"entry {idx} job_id mixes jobs")
        pd = entry["prev_digest"]
        if idx == 0:
            if pd != "0" * 64:
                failures.append("entry 0 prev_digest is not 64 zero characters")
        elif pd != prev:
            failures.append(f"entry {idx} prev_digest does not match the digest of entry {idx - 1}")
        if "close" in entry and not isinstance(entry["close"], bool):
            failures.append(f"entry {idx} close is not a boolean")
        elif idx < len(timeline) - 1 and entry.get("close") is True:
            failures.append(f"entry {idx} carries close:true before the last entry")
        prev = aer1_chain_entry_digest_v07(entry)

    last = timeline[-1]
    if not isinstance(last, Mapping) or last.get("close") is not True:
        failures.append("last entry does not carry close: true")
    return failures


def verify_aer1_chain_commitment(timeline: object, commitment: object) -> list[str]:
    """Section 7.3 external chain commitment verification."""
    if not isinstance(timeline, list) or not timeline:
        return ["timeline is not a non-empty list"]
    if not isinstance(commitment, Mapping):
        return ["commitment is not a JSON object"]
    failures: list[str] = []
    job_id = commitment.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        failures.append("commitment job_id is not a non-empty string")
    digest = commitment.get("final_entry_digest")
    if not isinstance(digest, str) or not _HEX64_RE.match(digest):
        failures.append("commitment final_entry_digest is not 64 lowercase hex characters")
    count = commitment.get("entry_count")
    if isinstance(count, bool) or not isinstance(count, int):
        failures.append("commitment entry_count is not an integer")
    published = commitment.get("published_at")
    if not isinstance(published, str) or not _RFC3339_RE.match(published):
        failures.append("commitment published_at is not RFC 3339")
    elif not _is_strict_rfc3339(published):
        failures.append("commitment published_at is not a valid calendar date/time")
    if failures:
        return failures

    assert isinstance(count, int)  # noqa: S101
    if count != len(timeline):
        failures.append(
            f"commitment entry_count {count} does not match the timeline length {len(timeline)}"
        )
    for idx, entry in enumerate(timeline):
        if (
            not isinstance(entry, Mapping)
            or not isinstance(entry.get("job_id"), str)
            or not entry["job_id"]
        ):
            failures.append(f"entry {idx} has no usable job_id")
        elif entry["job_id"] != job_id:
            failures.append(f"entry {idx} job_id does not match the commitment job_id")
    try:
        last_entry = timeline[-1]
        if not isinstance(last_entry, Mapping):
            return [*failures, "final entry is not an object"]
        want = aer1_chain_entry_digest_v07(last_entry)
    except (KeyError, ValueError, TypeError, binascii.Error) as exc:
        failures.append(f"final entry digest could not be recomputed: {exc}")
    else:
        if digest != want:
            failures.append(
                "commitment final_entry_digest does not match "
                "the recomputed digest of the last entry"
            )
    return failures


def export_receipts_to_aer1_chain_v07(
    receipts: Sequence[ToolReceipt],
    *,
    job_id: str | None = None,
    close: bool = True,
    published_at: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Export an ordered `ToolReceipt` sequence as a `-07` AER-1 chain + Section 7.3 commitment."""
    if not receipts:
        raise ValueError("receipts must be non-empty to build an AER-1 chain.")
    resolved_job = job_id or receipts[0].session_id or "epistemic-guard-job"
    timeline: list[dict[str, Any]] = []
    prev_digest = "0" * 64
    n = len(receipts)
    for idx, rcpt in enumerate(receipts):
        aer1_doc = rcpt.to_aer1()
        is_last = close and (idx == n - 1)
        entry: dict[str, Any] = {
            "id": str(aer1_doc["id"]),
            "seq": idx + 1,
            "job_id": resolved_job,
            "tool": rcpt.tool,
            "provenance_class": str(aer1_doc["provenance_class"]),
            "canonical_bytes": str(aer1_doc["canonical_bytes"]),
            "prev_digest": prev_digest,
            "close": is_last,
        }
        prev_digest = aer1_chain_entry_digest_v07(entry)
        timeline.append(entry)

    pub_ts = published_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    commitment = {
        "job_id": resolved_job,
        "final_entry_digest": prev_digest,
        "entry_count": n,
        "published_at": pub_ts,
    }
    return timeline, commitment
