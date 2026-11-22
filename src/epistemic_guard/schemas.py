"""Pydantic v2 schemas for `epistemic-guard` (zero LLM, UI, or daemon dependencies)."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from os import PathLike
from pathlib import Path
from typing import Any, Final
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from epistemic_guard.exceptions import SignatureTamperError


class Provenance(StrEnum):
    """Origin domain tags for memory envelopes."""

    USER_EXPLICIT = "user_explicit"
    USER_CONFIRMED = "user_confirmed"
    TOOL_VERIFIED = "tool_verified"
    LISP_ASSERTION = "lisp_assertion"
    INFERRED = "inferred"
    LLM_NARRATION = "llm_narration"
    TOOL_CONTENT_UNTRUSTED = "tool_content_untrusted"


ELIGIBLE_PROVENANCE: Final[frozenset[Provenance]] = frozenset(
    {
        Provenance.USER_EXPLICIT,
        Provenance.USER_CONFIRMED,
        Provenance.TOOL_VERIFIED,
        Provenance.LISP_ASSERTION,
        Provenance.INFERRED,
    }
)


class Hazard(StrEnum):
    """Query/claim risk tier used to scale required source corroboration."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ProofStandard(StrEnum):
    """Confidence thresholds mapped to human-readable proof tiers."""

    SPECULATION = "speculation"
    BALANCE_OF_PROBABILITIES = "balance of probabilities"
    CLEAR_AND_CONVINCING = "clear and convincing"
    BEYOND_REASONABLE_DOUBT = "beyond reasonable doubt"


PROOF_STANDARDS: Final[dict[ProofStandard, float]] = {
    ProofStandard.SPECULATION: 0.0,
    ProofStandard.BALANCE_OF_PROBABILITIES: 0.50,
    ProofStandard.CLEAR_AND_CONVINCING: 0.75,
    ProofStandard.BEYOND_REASONABLE_DOUBT: 0.95,
}


def standard_for_confidence(confidence: float) -> ProofStandard:
    """Map a confidence score in `[0.0, 1.0]` to its highest satisfied `ProofStandard`."""
    if confidence >= PROOF_STANDARDS[ProofStandard.BEYOND_REASONABLE_DOUBT]:
        return ProofStandard.BEYOND_REASONABLE_DOUBT
    if confidence >= PROOF_STANDARDS[ProofStandard.CLEAR_AND_CONVINCING]:
        return ProofStandard.CLEAR_AND_CONVINCING
    if confidence >= PROOF_STANDARDS[ProofStandard.BALANCE_OF_PROBABILITIES]:
        return ProofStandard.BALANCE_OF_PROBABILITIES
    return ProofStandard.SPECULATION


class BelnapValue(StrEnum):
    """Four-valued truth state (`neither`, `true`, `false`, `both` for contradictions)."""

    NEITHER = "neither"
    TRUE = "true"
    FALSE = "false"
    BOTH = "both"


class UnsignedPayload(BaseModel):
    """Memory record awaiting domain-separated Ed25519 signing."""

    payload_id: str = Field(default_factory=lambda: uuid4().hex)
    content: str
    provenance: Provenance
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    trace_id: str = ""
    created_at_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class SignedPayload(UnsignedPayload):
    """Ed25519-signed memory envelope verified by `ProvenanceGate`."""

    signature: str
    signer_domain: str


_AER1_CORE_FIELDS: Final[tuple[str, ...]] = (
    "id",
    "receipt_schema_version",
    "created_at",
    "tool",
    "provenance_class",
    "canonical_bytes",
    "output_hash",
    "verification_status",
)
_AER1_UUID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z"
)
_AER1_RFC3339_RE: Final[re.Pattern[str]] = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?(Z|[+-][0-9]{2}:[0-9]{2})\Z"
)
_AER1_HASH_RE: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}\Z")
_AER1_PROV_RE: Final[re.Pattern[str]] = re.compile(
    r"^(EXECUTED BY [A-Z0-9][A-Z0-9 ._-]*|OBSERVED VIA GATEWAY|LOGGED BY AGENT)\Z"
)
_AER1_B64_RE: Final[re.Pattern[str]] = re.compile(r"\A[A-Za-z0-9+/]*={0,2}\Z")
_AER1_NIL_UUID: Final[str] = "00000000-0000-0000-0000-000000000000"


def _coerce_receipt_arg(val: object) -> str | int | float | bool:
    if isinstance(val, bool) or isinstance(val, (str, int)):
        return val
    if isinstance(val, float):
        return val if math.isfinite(val) else str(val)
    try:
        return json.dumps(val, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return str(val)


def _is_strict_rfc3339(value: object) -> bool:
    """Validate strict RFC 3339 calendar date/time (no leap-second 60 or non-leap Feb 29)."""
    if not isinstance(value, str) or not _AER1_RFC3339_RE.match(value):
        return False
    if not value.endswith("Z"):
        off = value[-6:]
        try:
            oh, om = int(off[1:3]), int(off[4:6])
        except ValueError:
            return False
        if not (0 <= oh <= 23 and 0 <= om <= 59):
            return False
    try:
        year = int(value[0:4])
        month = int(value[5:7])
        day = int(value[8:10])
        hour = int(value[11:13])
        minute = int(value[14:16])
        second = int(value[17:19])
    except ValueError:
        return False
    if not (1 <= month <= 12):
        return False
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        return False
    if month in {1, 3, 5, 7, 8, 10, 12}:
        max_day = 31
    elif month in {4, 6, 9, 11}:
        max_day = 30
    else:
        is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        max_day = 29 if is_leap else 28
    return 1 <= day <= max_day


def _parse_json_with_duplicate_check(text: str) -> tuple[object, str | None]:
    """Parse JSON text while capturing the first duplicate object member name, if any."""
    duplicates: list[str] = []

    def _hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        seen: set[str] = set()
        for key, _val in pairs:
            if key in seen and not duplicates:
                duplicates.append(key)
            seen.add(key)
        return dict(pairs)

    def _reject_nonfinite(token: str) -> object:
        raise SignatureTamperError(f"non-standard JSON numeric constant: {token}")

    parsed: object = json.loads(
        text,
        object_pairs_hook=_hook,
        parse_constant=_reject_nonfinite,
    )
    return parsed, (duplicates[0] if duplicates else None)


def _is_aer1_legacy_production_shape(rec: Mapping[str, Any]) -> bool:
    """Narrow legacy schema 0.2/0.3 production envelope check (`timestamp` + `tool_context`)."""
    if "created_at" in rec:
        return False
    if rec.get("receipt_schema_version") not in {"0.2", "0.3"}:
        return False
    ts = rec.get("timestamp")
    if not isinstance(ts, str) or not ts:
        return False
    tool = rec.get("tool")
    ctx = rec.get("tool_context")
    if not isinstance(tool, str) or not tool or not isinstance(ctx, Mapping):
        return False
    if not all(isinstance(ctx.get(k), str) and bool(ctx[k]) for k in ("name", "version", "scope")):
        return False
    return bool(ctx.get("name") == tool)


def _json_quote(val: object) -> str:
    return json.dumps(val, ensure_ascii=False)


def _aer1_metadata_block(rec: Mapping[str, Any]) -> dict[str, Any] | None:
    """Rebuild Section 3.1 canonical metadata block (`draft-zambo-aer1-26`)."""
    tool = rec.get("tool")
    if not isinstance(tool, Mapping):
        return None
    block: dict[str, Any] = {}
    for key in ("created_at", "id", "output_hash", "provenance_class", "receipt_schema_version"):
        val = rec.get(key)
        if not isinstance(val, str):
            return None
        block[key] = val
    tool_block: dict[str, str] = {}
    for key in ("name", "scope", "version"):
        val = tool.get(key)
        if not isinstance(val, str):
            return None
        tool_block[key] = val
    block["tool"] = tool_block
    return block


def _aer1_metadata_digest(block: Mapping[str, Any]) -> str:
    blob = json.dumps(
        block, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(blob).hexdigest()}"


def _derive_aer1_uuid(call_id: str, canon_sha: str) -> str:
    if _AER1_UUID_RE.match(call_id) and call_id != _AER1_NIL_UUID:
        return call_id
    digest = bytearray(
        hashlib.sha256(f"epistemic-guard:aer1-id:{call_id}:{canon_sha}".encode()).digest()[:16]
    )
    digest[6] = (digest[6] & 0x0F) | 0x40
    digest[8] = (digest[8] & 0x3F) | 0x80
    return str(UUID(bytes=bytes(digest)))


def _normalize_aer1_timestamp(ts: str) -> str:
    if _is_strict_rfc3339(ts):
        return ts
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        else:
            dt = dt.astimezone(UTC)
        return dt.isoformat().replace("+00:00", "Z")
    except ValueError:
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _validate_and_decode_aer1(
    rec: Mapping[str, Any],
    *,
    duplicate_member: str | None,
    verify_commitment: bool,
    expected_id: str | None,
    resolved_record: Mapping[str, Any] | None,
) -> tuple[bytes, Mapping[str, Any], str, bool]:
    legacy_prod = _is_aer1_legacy_production_shape(rec)
    required: Sequence[str] = (
        tuple(f for f in _AER1_CORE_FIELDS if f != "created_at") + ("timestamp", "tool_context")
        if legacy_prod
        else _AER1_CORE_FIELDS
    )
    for field_name in required:
        if field_name not in rec:
            raise SignatureTamperError(f"missing required member: {field_name}")

    rid = rec["id"]
    if not isinstance(rid, str) or not _AER1_UUID_RE.match(rid):
        raise SignatureTamperError("id is not a lowercase UUID")
    if rid == _AER1_NIL_UUID:
        raise SignatureTamperError("id is the nil UUID")

    rsv = rec["receipt_schema_version"]
    if not isinstance(rsv, str) or not rsv:
        raise SignatureTamperError("receipt_schema_version is not a non-empty string")

    created_val = rec["timestamp"] if legacy_prod else rec["created_at"]
    if not isinstance(created_val, str):
        raise SignatureTamperError("created_at is not a string")
    if not _AER1_RFC3339_RE.match(created_val):
        raise SignatureTamperError("created_at is not RFC 3339")
    if not _is_strict_rfc3339(created_val):
        raise SignatureTamperError("created_at is not a valid calendar date/time")

    tool_val = rec["tool_context"] if legacy_prod else rec["tool"]
    if not isinstance(tool_val, Mapping) or not all(
        isinstance(tool_val.get(k), str) and bool(tool_val[k])
        for k in ("name", "version", "scope")
    ):
        raise SignatureTamperError("tool is not an object with name/version/scope strings")

    prov_cls = rec["provenance_class"]
    if not isinstance(prov_cls, str) or not _AER1_PROV_RE.match(prov_cls):
        raise SignatureTamperError("provenance_class is not a known class")

    cb = rec["canonical_bytes"]
    if not isinstance(cb, str) or not _AER1_B64_RE.match(cb):
        raise SignatureTamperError("canonical_bytes is not valid base64")
    try:
        canon_bytes = base64.b64decode(cb, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise SignatureTamperError("canonical_bytes is not valid base64") from exc
    if base64.b64encode(canon_bytes).decode("ascii") != cb:
        raise SignatureTamperError("canonical_bytes is not the canonical base64 spelling")
    try:
        canon_text = canon_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SignatureTamperError("canonical_bytes is not valid UTF-8") from exc

    oh = rec["output_hash"]
    if not isinstance(oh, str) or not _AER1_HASH_RE.match(oh):
        raise SignatureTamperError("output_hash is not sha256: + lowercase hex digest")

    computed_hex = hashlib.sha256(canon_bytes).hexdigest()
    if verify_commitment and oh != f"sha256:{computed_hex}":
        raise SignatureTamperError("output_hash does not match sha256(canonical_bytes)")

    mh = rec.get("metadata_hash")
    if mh is not None:
        if not isinstance(mh, str) or not _AER1_HASH_RE.match(mh):
            raise SignatureTamperError("metadata_hash is not sha256: + lowercase hex digest")
        if verify_commitment:
            mblock = _aer1_metadata_block(rec)
            if mblock is None:
                raise SignatureTamperError(
                    "metadata block cannot be recomputed from the stored member values"
                )
            if mh != _aer1_metadata_digest(mblock):
                raise SignatureTamperError(
                    "metadata_hash does not match the recomputed metadata block"
                )

    try:
        inner_parsed, inner_dup = _parse_json_with_duplicate_check(canon_text)
    except json.JSONDecodeError as exc:
        raise SignatureTamperError("canonical payload is not valid JSON") from exc
    if inner_dup is not None:
        raise SignatureTamperError(f"duplicate member name: {inner_dup}")

    ih = rec.get("input_hash")
    if ih is not None:
        if not isinstance(ih, str) or not _AER1_HASH_RE.match(ih):
            raise SignatureTamperError("input_hash is not sha256: + lowercase hex digest")
        if verify_commitment and isinstance(inner_parsed, Mapping) and "inputs" in inner_parsed:
            inputs_raw = json.dumps(
                inner_parsed["inputs"], sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            want_ih = f"sha256:{hashlib.sha256(inputs_raw).hexdigest()}"
            if ih != want_ih:
                raise SignatureTamperError(
                    "input_hash does not match sha256(canonical JSON of the recorded inputs)"
                )

    ver_status = rec["verification_status"]
    if legacy_prod:
        if not isinstance(ver_status, str) or not ver_status:
            raise SignatureTamperError("verification_status is not a non-empty string")
    elif ver_status != "verified":
        raise SignatureTamperError("verification_status is not verified")

    if duplicate_member is not None:
        raise SignatureTamperError(f"duplicate member name: {duplicate_member}")

    if verify_commitment:
        inner_ver = (
            inner_parsed.get("receipt_schema_version")
            if isinstance(inner_parsed, Mapping)
            else None
        )
        legacy_ver = inner_ver if inner_ver is not None else rsv
        is_legacy = legacy_ver is not None and str(legacy_ver) in {"0.2", "0.3"}
        if not is_legacy:
            if inner_ver is not None:
                if str(inner_ver) != str(rsv):
                    raise SignatureTamperError(
                        f"receipt_schema_version: outer {_json_quote(rsv)} "
                        f"does not match committed {_json_quote(inner_ver)}"
                    )
            else:
                raise SignatureTamperError(
                    f"receipt_schema_version: one side missing "
                    f"(outer={_json_quote(rsv)} committed={_json_quote(inner_ver)})"
                )
            if isinstance(inner_parsed, Mapping):
                issuer_claims = inner_parsed.get("issuer_claims")
                committed_pc = (
                    issuer_claims.get("provenance_class")
                    if isinstance(issuer_claims, Mapping)
                    else None
                )
                if committed_pc is not None:
                    if committed_pc != prov_cls:
                        raise SignatureTamperError(
                            f"provenance binding: outer {_json_quote(prov_cls)} "
                            f"does not match committed {_json_quote(committed_pc)}"
                        )
                else:
                    raise SignatureTamperError(
                        f"provenance binding: one side missing "
                        f"(outer={_json_quote(prov_cls)} committed={_json_quote(committed_pc)})"
                    )
            else:
                raise SignatureTamperError(
                    "provenance binding: canonical payload did not decode to an object"
                )

    if expected_id is not None and rid != expected_id:
        raise SignatureTamperError(
            f"id mismatch: artifact id {rid!r} does not match "
            f"the externally supplied expected id {expected_id!r}"
        )
    if resolved_record is not None:
        rrec_raw = resolved_record.get("receipt")
        rrec: object = rrec_raw if isinstance(rrec_raw, Mapping) else resolved_record
        if not isinstance(rrec, Mapping):
            raise SignatureTamperError("resolved record is not a JSON object")
        r_digest = rrec.get("output_hash")
        r_id = rrec.get("id")
        if r_digest != oh:
            raise SignatureTamperError(
                f"id mismatch: resolved record digest {r_digest!r} disagrees "
                f"with the artifact's committed digest {oh!r}"
            )
        if r_id is not None and r_id != rid:
            raise SignatureTamperError(
                f"id mismatch: resolved record carries id {r_id!r}, "
                f"not the artifact's id {rid!r}"
            )

    canon_obj: Mapping[str, Any] = inner_parsed if isinstance(inner_parsed, Mapping) else {}
    return canon_bytes, canon_obj, computed_hex, legacy_prod


class ToolReceipt(BaseModel):
    """Execution or pre-execution rejection receipt emitted for a tool call attempt."""

    tool: str
    ok: bool = True
    executed: bool = True
    error_code: str = ""
    call_id: str = Field(default_factory=lambda: f"rcpt-{uuid4().hex[:10]}")
    args: dict[str, str | int | float | bool] = Field(default_factory=dict)
    output: str = ""
    output_sha256: str = ""
    tool_definition_sha256: str = ""
    endpoint_id: str = ""
    evidence_tier: str = ""
    effect_observation: dict[str, Any] | None = None
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    session_id: str = ""
    signature: str = ""

    @classmethod
    def from_aer1(
        cls,
        payload: Mapping[str, Any] | str | bytes | PathLike[str],
        *,
        verify_commitment: bool = True,
        expected_id: str | None = None,
        resolved_record: Mapping[str, Any] | None = None,
    ) -> ToolReceipt:
        """Parse and verify an AER-1 (`draft-zambo-aer1`) receipt into a `ToolReceipt`."""
        duplicate_member: str | None = None
        if isinstance(payload, PathLike):
            try:
                raw_text = Path(payload).read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise SignatureTamperError("receipt is not valid UTF-8") from exc
            try:
                loaded, duplicate_member = _parse_json_with_duplicate_check(raw_text)
            except json.JSONDecodeError as exc:
                raise SignatureTamperError("receipt is not valid JSON") from exc
            if not isinstance(loaded, Mapping):
                raise SignatureTamperError("receipt is not a JSON object")
            outer_map: Mapping[str, Any] = loaded
        elif isinstance(payload, (str, bytes)):
            if isinstance(payload, bytes):
                try:
                    raw_text = payload.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise SignatureTamperError("receipt is not valid UTF-8") from exc
            else:
                raw_text = payload
            try:
                loaded, duplicate_member = _parse_json_with_duplicate_check(raw_text)
            except json.JSONDecodeError as exc:
                raise SignatureTamperError("receipt is not valid JSON") from exc
            if not isinstance(loaded, Mapping):
                raise SignatureTamperError("receipt is not a JSON object")
            outer_map = loaded
        elif isinstance(payload, Mapping):
            outer_map = payload
        else:
            raise SignatureTamperError("receipt is not a JSON object")

        rec_nested = outer_map.get("receipt")
        rec: Mapping[str, Any] = (
            rec_nested
            if isinstance(rec_nested, Mapping)
            and "canonical_bytes" not in outer_map
            and "id" not in outer_map
            else outer_map
        )

        _canon_bytes, canon_obj, computed_hex, legacy_prod = _validate_and_decode_aer1(
            rec,
            duplicate_member=duplicate_member,
            verify_commitment=verify_commitment,
            expected_id=expected_id,
            resolved_record=resolved_record,
        )

        tool_block = rec["tool_context"] if legacy_prod else rec["tool"]
        tool_name = str(tool_block["name"])

        steps = rec.get("steps")
        step0 = (
            steps[0]
            if isinstance(steps, list) and steps and isinstance(steps[0], Mapping)
            else {}
        )
        step_status = str(step0.get("status", "")).lower()

        out_obj = canon_obj.get("output")
        executed = True
        ok = True
        error_code = ""
        output_text = str(rec.get("result_preview") or "")
        args_map: dict[str, str | int | float | bool] = {}

        prov_obj = rec.get("provenance")
        if isinstance(canon_obj.get("inputs"), Mapping):
            args_map = {
                str(k): _coerce_receipt_arg(v) for k, v in canon_obj["inputs"].items()
            }
        elif isinstance(prov_obj, Mapping) and isinstance(prov_obj.get("request"), Mapping):
            args_map = {
                str(k): _coerce_receipt_arg(v) for k, v in prov_obj["request"].items()
            }

        if isinstance(out_obj, Mapping):
            is_pre_rej = (
                out_obj.get("executed") is False
                or out_obj.get("receipt_type") == "pre_execution_rejection"
            )
            if is_pre_rej:
                executed = False
                ok = False
            if isinstance(out_obj.get("arguments_received"), Mapping):
                args_map = {
                    str(k): _coerce_receipt_arg(v)
                    for k, v in out_obj["arguments_received"].items()
                }
            err_obj = out_obj.get("error")
            if isinstance(err_obj, Mapping):
                error_code = str(err_obj.get("code", ""))
                output_text = str(err_obj.get("message", output_text))
                ok = False
            elif "text" in out_obj:
                output_text = str(out_obj["text"])

        if step_status in {"rejected_before_execution", "rejected"}:
            executed = False
            ok = False
        elif step_status in {"failed", "error"}:
            ok = False

        caller_obj = canon_obj.get("caller")
        restored_session_id = (
            str(caller_obj.get("session_id", ""))
            if isinstance(caller_obj, Mapping) and caller_obj.get("session_id")
            else ""
        )
        restored_call_id = (
            str(caller_obj["id"])
            if isinstance(caller_obj, Mapping)
            and caller_obj.get("type") == "call"
            and isinstance(caller_obj.get("id"), str)
            and caller_obj["id"]
            else str(rec["id"])
        )
        created_str = str(rec["timestamp"] if legacy_prod else rec["created_at"])
        output_digest = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
        return cls(
            tool=tool_name,
            ok=ok,
            executed=executed,
            error_code=error_code,
            call_id=restored_call_id,
            args=args_map,
            output=output_text,
            output_sha256=output_digest,
            timestamp_utc=created_str,
            session_id=restored_session_id,
        )

    def to_aer1(
        self,
        *,
        receipt_schema_version: str = "0.4",
        provenance_class: str | None = None,
    ) -> dict[str, Any]:
        """Export this `ToolReceipt` as a portable AER-1-compatible JSON dictionary."""
        prov_cls = provenance_class or (
            "EXECUTED BY EPISTEMIC-GUARD" if self.executed else "OBSERVED VIA GATEWAY"
        )
        if not _AER1_PROV_RE.match(prov_cls):
            raise ValueError(f"Invalid AER-1 provenance_class: {prov_cls!r}")
        if not self.executed:
            out_block: dict[str, Any] = {
                "arguments_received": dict(self.args),
                "error": {
                    "code": self.error_code or "invalid_tool_arguments",
                    "message": self.output,
                },
                "executed": False,
                "outcome": "rejected_before_execution",
                "receipt_type": "pre_execution_rejection",
                "requested_tool": self.tool,
            }
            step_status = "rejected_before_execution"
        elif not self.ok:
            out_block = {
                "arguments_received": dict(self.args),
                "error": {
                    "code": self.error_code or "execution_error",
                    "message": self.output,
                },
                "executed": True,
                "outcome": "execution_failed",
            }
            step_status = "failed"
        else:
            out_block = {"text": self.output}
            step_status = "success"

        caller_block: dict[str, str] = {"id": self.call_id, "type": "call"}
        if self.session_id:
            caller_block["session_id"] = self.session_id
        inputs_dict = dict(self.args)
        inputs_raw = json.dumps(
            inputs_dict, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        canon_payload: dict[str, Any] = {
            "caller": caller_block,
            "inputs": inputs_dict,
            "issuer_claims": {"provenance_class": prov_cls},
            "output": out_block,
            "receipt_schema_version": receipt_schema_version,
            "tool": {"name": self.tool, "scope": "local", "version": "0.6.0"},
        }
        canon_raw = json.dumps(
            canon_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        canon_sha = hashlib.sha256(canon_raw).hexdigest()
        inputs_sha = hashlib.sha256(inputs_raw).hexdigest()
        aer1_id = _derive_aer1_uuid(self.call_id, canon_sha)
        created_at = _normalize_aer1_timestamp(self.timestamp_utc)
        tool_descriptor = {"name": self.tool, "scope": "local", "version": "0.6.0"}
        output_hash_str = f"sha256:{canon_sha}"
        meta_hash_str = _aer1_metadata_digest(
            {
                "created_at": created_at,
                "id": aer1_id,
                "output_hash": output_hash_str,
                "provenance_class": prov_cls,
                "receipt_schema_version": receipt_schema_version,
                "tool": tool_descriptor,
            }
        )
        return {
            "id": aer1_id,
            "created_at": created_at,
            "receipt_schema_version": receipt_schema_version,
            "tool": tool_descriptor,
            "provenance_class": prov_cls,
            "canonical_bytes": base64.b64encode(canon_raw).decode("ascii"),
            "input_hash": f"sha256:{inputs_sha}",
            "output_hash": output_hash_str,
            "metadata_hash": meta_hash_str,
            "verification_status": "verified",
            "steps": [
                {
                    "tool": self.tool,
                    "status": step_status,
                    "completed_at": created_at,
                }
            ],
        }


class StructuredActionTag(BaseModel):
    """Parsed `[ACTION: <verb> <target>]` or `<action_claim>` tag from assistant output."""

    verb: str
    target: str
    raw_tag: str
    start_char: int
    end_char: int
    grounded: bool


class UnsupportedClaimSpan(BaseModel):
    """Character-offset span of an assistant clause claiming an unexecuted action."""

    text: str
    start_char: int
    end_char: int
    claimed_family: str
    reason: str


class SupervisionVerdict(BaseModel):
    """Clause-level action-receipt verification result."""

    is_valid: bool
    device: str = "CPU"
    latency_ms: float = 0.0
    clauses_checked: int = 0
    unsupported_claims: list[str] = Field(default_factory=list)
    unsupported_spans: list[UnsupportedClaimSpan] = Field(default_factory=list)
    unfulfilled_promises: list[str] = Field(default_factory=list)
    structured_actions: list[StructuredActionTag] = Field(default_factory=list)

    def to_retry_prompt(self, executed_receipts: Sequence[ToolReceipt] | None = None) -> str:
        """Build a concise system/developer correction prompt for automatic LLM retry loops."""
        if self.is_valid:
            return ""
        all_rcpts = list(executed_receipts or ())
        ok_tools = [
            f"{r.tool}({', '.join(f'{k}={v!r}' for k, v in r.args.items())})"
            for r in all_rcpts
            if r.ok and r.executed
        ]
        failed_attempts: list[str] = []
        for r in all_rcpts:
            if r.ok and r.executed:
                continue
            args_fmt = ", ".join(f"{k}={v!r}" for k, v in r.args.items())
            stage_lbl = "REJECTED BEFORE EXECUTION" if not r.executed else "FAILED"
            detail = (
                f"{r.error_code}: {r.output}"
                if r.error_code and r.error_code not in r.output
                else (r.output or r.error_code or "error")
            )
            failed_attempts.append(f"{r.tool}({args_fmt}) -> {stage_lbl} ({detail})")

        tools_summary = ", ".join(ok_tools) if ok_tools else "NONE"
        failed_clause = (
            f" Rejected/failed tool attempts in this turn: [{'; '.join(failed_attempts)}]."
            if failed_attempts
            else ""
        )
        flagged = "; ".join(f'"{c}"' for c in (self.unsupported_claims + self.unfulfilled_promises))
        return (
            "VERIFICATION FAILED: Your draft reply claimed actions or pending executions without "
            f"matching tool receipts ({flagged}). Verified tool calls in this turn: "
            f"[{tools_summary}].{failed_clause} Rewrite your reply so it only states actions "
            "backed by verified tool calls, accurately reports any rejected/failed tool attempts, "
            "or asks before running the tool."
        )


# Backwards-compatible alias
ClauseSupervisionVerdict = SupervisionVerdict


class NLIVerdict(BaseModel):
    """3-class natural language inference result for `(premise, hypothesis)`."""

    label: str
    contradiction: float
    entailment: float
    neutral: float
    device: str = "CPU"
    latency_ms: float = 0.0

    @property
    def belnap(self) -> BelnapValue:
        """Fail-closed 4-valued Belnap projection (`neutral` -> `NEITHER`)."""
        if self.label == "entailment":
            return BelnapValue.TRUE
        if self.label == "contradiction":
            return BelnapValue.FALSE
        return BelnapValue.NEITHER


class PassageAssessment(BaseModel):
    """Per-source relevance and stance classification."""

    domain: str
    stance: str
    confidence: float
    used_neural: bool = False


class WebEvidentiaryBurden(BaseModel):
    """Multi-source RAG corroboration result after domain & wire-copy deduplication."""

    query: str
    hazard: Hazard
    hazard_device: str = "CPU"
    evidence_device: str = "DETERMINISTIC_CPU"
    nli_device: str = "DETERMINISTIC_CPU"
    hazard_latency_ms: float = 0.0
    corroboration_latency_ms: float = 0.0
    confidence: float
    achieved_standard: ProofStandard
    required_casual_standard: ProofStandard
    required_commit_standard: ProofStandard
    required_unhedged_standard: ProofStandard
    meets_casual_burden: bool
    meets_commit_burden: bool
    meets_unhedged_burden: bool
    raw_sources_count: int = 0
    deduplicated_duplicates_collapsed: int = 0
    irrelevant_sources_filtered: int = 0
    independent_sources_count: int = 0
    independent_domains: list[str] = Field(default_factory=list)
    disputed: bool = False
    dispute_sources: list[str] = Field(default_factory=list)
    passage_assessments: list[PassageAssessment] = Field(default_factory=list)
    summary_tag: str = ""
    guidance: str = ""
