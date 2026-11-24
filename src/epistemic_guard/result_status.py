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


def classify_tool_result(content: str | Mapping[str, Any], *, tool: str = "") -> ResultStatus:
    """Classify a tool result from explicit, machine-readable signals.

    `indeterminate`: a top-level `status`/`outcome` of unknown or partly applied, for a tool that is
    not read-only (a read tool's `status` usually describes the record it fetched, not the call).
    `failed`: `ok`/`success` false, `isError`/`is_error` true, a non-empty `error`, a failed
    `status`, or text starting with `Error:`/`Traceback`.
    """
    if isinstance(content, str) and content.strip().lower().startswith(("error:", "traceback")):
        return "failed"
    obj = _as_mapping(content)
    if obj is None:
        return "ok"
    statuses = {_norm_status(obj.get("status")), _norm_status(obj.get("outcome"))}
    if not (tool and is_read_only_tool(tool)) and statuses & _INDETERMINATE_STATUSES:
        return "indeterminate"
    if obj.get("ok") is False or obj.get("success") is False:
        return "failed"
    if obj.get("isError") is True or obj.get("is_error") is True:
        return "failed"
    if obj.get("error") not in (None, False, "", [], {}):
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
    if obj.get("created") is False:
        return True
    return any(obj.get(k) is True for k in ("replayed", "idempotent_replay", "deduplicated"))
