"""Classify a tool result payload as `ok`, `failed`, or `indeterminate`.

Shared by the MCP proxies (`McpStdioProxy`, `McpStreamableHttpProxy`) and the transcript extractor
(`extract_receipts_from_messages`) so the same result is never a clean execution on one path and a
failure on another. Many MCP servers report failures without `isError`, and some report a write as
unknown or partly applied; only machine-readable signals are used here, never free prose.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Final, Literal

ResultStatus = Literal["ok", "failed", "indeterminate"]

_READBACK_TOOL_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|_)(?:get|read|search|query|fetch|check|verify|inspect|stat|status|scan|lookup|probe|list|find|show)(?:_|$)",
    re.IGNORECASE,
)
_MUTATING_TOOL_VERB_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|_)(?:create|write|edit|update|set|put|post|patch|delete|remove|drop|insert|add|"
    r"append|send|email|mail|invite|pay|transfer|refund|book|schedule|commit|push|merge|"
    r"deploy|exec|execute|run|bash|shell|kill|stop|start|restart|enable|disable|toggle|"
    r"revoke|cancel|archive|wipe|purge|move|rename|copy|upload|publish|assign|protect)(?:_|$)",
    re.IGNORECASE,
)

_FAILED_STATUSES: Final[frozenset[str]] = frozenset(
    {"error", "failed", "failure", "denied", "rejected"}
)
# Status values that say the call's own effect is unknown or only partly applied.
_INDETERMINATE_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "unknown",
        "indeterminate",
        "uncertain",
        "partial",
        "partially_applied",
        "partially_completed",
        "partial_success",
        "partial_failure",
    }
)


_UNKNOWN_TOKENS: Final[frozenset[str]] = frozenset({"unknown", "indeterminate", "uncertain"})
_OUTCOME_KEYS: Final[tuple[str, ...]] = ("status", "outcome", "state", "result_status")
_SIDE_EFFECT_KEYS: Final[tuple[str, ...]] = ("side_effect_status", "side_effect", "effect_status")
_ERROR_CODE_KEYS: Final[tuple[str, ...]] = ("error", "error_code", "code")


def _says_unknown(val: object) -> bool:
    """Return True when a status-like string value reads unknown, indeterminate or uncertain."""
    norm = _norm_status(val)
    if not norm:
        return False
    return (
        norm in _INDETERMINATE_STATUSES
        or "maybe_applied" in norm
        or bool(set(re.split(r"[^a-z0-9]+", norm)) & _UNKNOWN_TOKENS)
    )


def _unknown_error_code(val: object) -> bool:
    """Return True for an error CODE string such as `rent_outcome_unknown` or `x_uncertain`."""
    if isinstance(val, Mapping):
        return any(_unknown_error_code(val.get(k)) for k in ("code", "error_code"))
    norm = _norm_status(val)
    return bool(norm) and norm.endswith(("_unknown", "_uncertain"))


def _candidates(obj: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """`obj` plus MCP envelopes: `structuredContent`, JSON text blocks, `result`/`data`."""
    found: list[Mapping[str, Any]] = [obj]
    structured = obj.get("structuredContent")
    if isinstance(structured, Mapping):
        found.append(structured)
    blocks = obj.get("content")
    if isinstance(blocks, list):
        for block in blocks:
            if isinstance(block, Mapping) and isinstance(block.get("text"), str):
                inner = _as_mapping(block["text"])
                if inner is not None:
                    found.append(inner)
    for cand in list(found):
        for key in ("result", "data"):
            nested = cand.get(key)
            if isinstance(nested, Mapping):
                found.append(nested)
    return found


def _reads_unknown(obj: Mapping[str, Any]) -> bool:
    """Value-based unknown-outcome shapes, whatever the member is called.

    (a) a status/outcome/state member whose value reads unknown/indeterminate/uncertain
    (`submission_uncertain`, `outcome_unknown`); (b) an error code ending `_unknown`/`_uncertain`
    (`rent_outcome_unknown`); (c) a side-effect member that reads unknown. `retry_safe: false` is
    not a signal on its own.
    """
    return (
        any(_says_unknown(obj.get(k)) for k in _OUTCOME_KEYS + _SIDE_EFFECT_KEYS)
        or any(_unknown_error_code(obj.get(k)) for k in _ERROR_CODE_KEYS)
    )


def is_read_only_tool(tool_name: str) -> bool:
    """Return True iff `tool_name` is a read/query/probe tool with no mutating verb."""
    return bool(_READBACK_TOOL_RE.search(tool_name)) and not bool(
        _MUTATING_TOOL_VERB_RE.search(tool_name)
    )


def _as_mapping(content: str | Mapping[str, Any]) -> Mapping[str, Any] | None:
    if isinstance(content, Mapping):
        return content
    stripped = content.strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        parsed = json.loads(stripped)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _norm_status(val: object) -> str:
    return re.sub(r"[\s-]+", "_", str(val).strip().lower()) if isinstance(val, str) else ""


_NO_ERROR_STRINGS: Final[frozenset[str]] = frozenset(
    {"none", "null", "false", "no", "no error", "no errors", "n/a", "na", "nil", "0", "ok"}
)
_SUCCESS_STATUSES: Final[frozenset[str]] = frozenset(
    {"ok", "success", "succeeded", "successful", "completed", "complete", "done"}
)
_BATCH_KEYS: Final[tuple[str, ...]] = ("results", "items", "responses", "operations", "batch")
_ITEM_SIGNAL_KEYS: Final[tuple[str, ...]] = (
    "status",
    "ok",
    "success",
    "error",
    "isError",
    "is_error",
)


def _has_error_value(val: object) -> bool:
    """Return True when an `error` field carries a real error, not a null/empty placeholder."""
    if val is None or val is False or val == 0:
        return False
    if isinstance(val, str):
        return val.strip().lower() not in _NO_ERROR_STRINGS and bool(val.strip())
    if isinstance(val, (list, tuple, dict, set)):
        return bool(val)
    return True


def _item_failed(item: Mapping[str, Any]) -> bool:
    return bool(
        item.get("ok") is False
        or item.get("success") is False
        or item.get("isError") is True
        or item.get("is_error") is True
        or _has_error_value(item.get("error"))
        or _norm_status(item.get("status")) in _FAILED_STATUSES
    )


def _batch_items(obj: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
    """Return per-item result mappings when `obj` is a batch result with per-item statuses."""
    for key in _BATCH_KEYS:
        val = obj.get(key)
        if (
            isinstance(val, list)
            and val
            and all(isinstance(i, Mapping) for i in val)
            and all(any(k in i for k in _ITEM_SIGNAL_KEYS) for i in val)
        ):
            return list(val)
    return None


def classify_tool_result(content: str | Mapping[str, Any], *, tool: str = "") -> ResultStatus:
    """Classify a tool result from explicit, machine-readable signals.

    `indeterminate`: a `status`/`outcome`/`state` whose value reads unknown, indeterminate,
    uncertain or partly applied, an error code ending `_unknown`/`_uncertain`, or an unknown
    side-effect status (also inside `structuredContent`, JSON text blocks and one `result`/`data`
    level), for a tool that is not read-only (a read tool's `status` usually describes the record
    it fetched, not the call);
    or a non-empty `error` that contradicts an explicit success signal (see below).
    `failed`: `ok`/`success` false, `isError`/`is_error` true, a non-empty `error`, a failed
    `status`, or text starting with `Error:`/`Traceback`.

    A null, empty, false, zero or `"none"`-like `error` is no error. When a non-empty top-level
    `error` sits next to per-item statuses (`results`/`items`/`responses`/`operations`/`batch`),
    the items decide: all items succeeded is `ok`, all failed is `failed`, a mix is
    `indeterminate`. A non-empty `error` next to `ok: true`/`success: true`/a success `status`
    with no per-item statuses is ambiguous: `indeterminate` for a mutating tool, `failed` for a
    read-only one.
    """
    if isinstance(content, str) and content.strip().lower().startswith(("error:", "traceback")):
        return "failed"
    obj = _as_mapping(content)
    if obj is None:
        return "ok"
    statuses = {_norm_status(obj.get("status")), _norm_status(obj.get("outcome"))}
    read_only = bool(tool) and is_read_only_tool(tool)
    if not read_only and (
        statuses & _INDETERMINATE_STATUSES or any(_reads_unknown(c) for c in _candidates(obj))
    ):
        return "indeterminate"
    if obj.get("ok") is False or obj.get("success") is False:
        return "failed"
    if obj.get("isError") is True or obj.get("is_error") is True:
        return "failed"
    if _has_error_value(obj.get("error")):
        items = _batch_items(obj)
        if items is not None:
            failed_n = sum(1 for i in items if _item_failed(i))
            if failed_n == 0:
                return "ok"
            if failed_n == len(items):
                return "failed"
            return "indeterminate"
        claims_success = (
            obj.get("ok") is True
            or obj.get("success") is True
            or bool(statuses & _SUCCESS_STATUSES)
        )
        if claims_success and not read_only:
            return "indeterminate"
        return "failed"
    if statuses & _FAILED_STATUSES:
        return "failed"
    return "ok"


def is_replayed_result(content: str | Mapping[str, Any]) -> bool:
    """Return True when a successful result says the call created nothing new this time.

    An idempotent server answers a retried write with the original result plus a marker:
    a top-level `created: false`, or `replayed`/`idempotent_replay`/`deduplicated` true. The
    write happened on an earlier call, so the result still counts as `ok`, but it cannot ground
    a claim that this turn created the object.
    """
    obj = _as_mapping(content) if isinstance(content, (str, Mapping)) else None
    if obj is None:
        return False
    # MCP wraps results as {"content": [{"type": "text", "text": "<json>"}]} and/or
    # {"structuredContent": {...}}.
    candidates: list[Mapping[str, Any]] = [obj]
    structured = obj.get("structuredContent")
    if isinstance(structured, Mapping):
        candidates.append(structured)
    blocks = obj.get("content")
    if isinstance(blocks, list):
        for block in blocks:
            if isinstance(block, Mapping) and isinstance(block.get("text"), str):
                inner = _as_mapping(block["text"])
                if inner is not None:
                    candidates.append(inner)
    # One level of a `result` / `data` envelope.
    for cand in list(candidates):
        for key in ("result", "data"):
            nested = cand.get(key)
            if isinstance(nested, Mapping):
                candidates.append(nested)

    def _is_false(val: object) -> bool:
        return val is False or (isinstance(val, str) and val.strip().lower() == "false")

    def _is_true(val: object) -> bool:
        return val is True or (isinstance(val, str) and val.strip().lower() == "true")

    return any(
        _is_false(c.get("created"))
        or any(_is_true(c.get(k)) for k in ("replayed", "idempotent_replay", "deduplicated"))
        for c in candidates
    )
