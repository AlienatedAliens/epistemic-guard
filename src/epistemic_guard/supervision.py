"""Clause-level & streaming `ToolReceipt` supervision with OpenVINO INT8 10-class speech-act
classification, exact character-offset spans, and `[ACTION: verb target]` (`SOAC`) tag checks.
"""

from __future__ import annotations

import contextlib
import json
import re
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo

import numpy as np
import openvino as ov
from tokenizers import Tokenizer

from epistemic_guard.exceptions import ProvenanceError
from epistemic_guard.hazard import split_into_clauses_with_spans
from epistemic_guard.models import resolve_managed_model_dir
from epistemic_guard.nli import OpenVINONLIVerifier
from epistemic_guard.provenance import ProvenanceGate
from epistemic_guard.schemas import (
    StructuredActionTag,
    SupervisionVerdict,
    ToolReceipt,
    UnsupportedClaimSpan,
)

DEFAULT_SEQ_LEN: Final[int] = 192

SPEECH_ACT_CLASSES: Final[tuple[str, ...]] = (
    "BENIGN",
    "UNFULFILLED_PROMISE",
    "CLAIM_FILE_WRITE",
    "CLAIM_DELETE",
    "CLAIM_LAUNCH_EXEC",
    "CLAIM_SEARCH_READ",
    "CLAIM_MEMORY",
    "CLAIM_COMM_EXTERNAL",
    "CLAIM_SYSTEM_STATE",
    "DEONTIC_CAPITULATION",
)

_FAMILY_TOOL_STEMS: Final[dict[str, tuple[str, ...]]] = {
    "CLAIM_FILE_WRITE": (
        "write",
        "save",
        "create",
        "edit",
        "update",
        "append",
        "patch",
        "note",
        "put",
        "upsert",
        "insert",
        "ticket",
        "issue",
    ),
    "CLAIM_DELETE": (
        "delete",
        "remove",
        "unlink",
        "erase",
        "wipe",
        "rm",
        "purge",
        "trash",
        "clear",
        "flush",
        "drop",
    ),
    "CLAIM_LAUNCH_EXEC": (
        "run",
        "exec",
        "bash",
        "shell",
        "cmd",
        "git",
        "commit",
        "install",
        "kill",
        "stop",
        "launch",
        "open",
        "start",
        "build",
        "deploy",
        "calc",
    ),
    "CLAIM_SEARCH_READ": (
        "read",
        "search",
        "fetch",
        "list",
        "status",
        "usage",
        "inspect",
        "scan",
        "check",
        "query",
        "find",
        "lookup",
        "verify",
        "get",
    ),
    "CLAIM_MEMORY": ("memory", "commit", "remember", "store", "save", "note"),
    "CLAIM_COMM_EXTERNAL": (
        "send",
        "email",
        "mail",
        "message",
        "post",
        "http",
        "webhook",
        "slack",
        "upload",
        "schedule",
        "book",
        "order",
        "transfer",
        "refund",
        "notify",
    ),
    "CLAIM_SYSTEM_STATE": (
        "set",
        "update",
        "toggle",
        "switch",
        "mute",
        "unmute",
        "connect",
        "disconnect",
        "enable",
        "disable",
        "pause",
        "restart",
        "reboot",
        "launch",
        "open",
    ),
    # Safety/sandbox bypass claims are never grounded by any tool receipt
    "DEONTIC_CAPITULATION": (),
}

_DETERMINISTIC_FAMILY_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "CLAIM_FILE_WRITE",
        re.compile(
            r"\bI\s+(?:(?:also|have|just|already|successfully)\s+)*(?:saved|wrote|written|created|updated|modified|edited|archived)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "CLAIM_DELETE",
        re.compile(
            r"\bI\s+(?:(?:also|have|just|already|successfully)\s+)*(?:deleted|removed|erased|wiped|formatted|purged|cleared)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "CLAIM_LAUNCH_EXEC",
        re.compile(
            r"\bI\s+(?:(?:also|have|just|already|successfully)\s+)*(?:executed|ran|installed|committed|pushed|killed|restarted|deployed)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "CLAIM_COMM_EXTERNAL",
        re.compile(
            r"\bI\s+(?:(?:also|have|just|already|successfully)\s+)*(?:sent|emailed|messaged|dispatched|refunded|transferred|booked)\b",
            re.IGNORECASE,
        ),
    ),
)

_HEDGE_OR_NEGATION_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^\s*(?:#{1,6}\s*|\*{1,2})?[^\n.!]{1,70}\b"
    r"(?:failed|error|rejected|denied|aborted|unsuccessful|timed\s+out)\b[\s*!:]*$|"
    r"\b(?:if\s+I|would\s+have|could\s+have|should\s+have|cannot|can't|could\s+not|couldn't|"
    r"did\s+not|didn't|haven't|have\s+not|has\s+not|hasn't|not\s+yet|unable\s+to|"
    r"failed\s+(?:to|due\s+to|because|with)|(?:has|have)\s+failed\b|"
    r"(?:attempted|tried)\s+to\b(?![^.!?\n]*\b(?:succeeded|worked)\b)|"
    r"(?:was|were|is|are)\s+not\s+(?:be\s+)?\w+(?:ed|en)\b|"
    r"no\s+[\w-]+\s+(?:was|were|has\s+been|have\s+been)\s+\w+(?:ed|en)\b|"
    r"(?:rejected|denied|refused)\s+(?:the\s+)?(?:request|call|operation|action|attempt|command)\b|"
    r"(?:reported|returned|encountered)\s+an?\s+[^\n.!?]{0,35}\b(?:error|failure|exit|crash|timeout)\b|"
    r"(?:received|with)\s+invalid\s+parameters?\b|"
    r"(?:server|process|tool)\s+(?:crashed|terminated\s+unexpectedly|exited\s+prematurely)\b|"
    r"^\s*\*{0,2}#{0,6}\s*failure\s+report\*{0,2}\s*$|"
    r"(?:is|was|remains?)\s+(?:currently\s+)?(?:deferred|aborted|skipped|unchanged|untouched)\b|"
    r"(?:operation|execution|tool|email|query|sql\s+execution|file\s+operation|request|call|"
    r"attempt|command|action|process)\s+"
    r"(?:failed|errored|aborted|timed\s+out|was\s+(?:rejected|denied|refused|aborted))\b|"
    r"suppose\s+I|imagine\s+I|pretend\s+I|for\s+example|for\s+instance|e\.g\.|"
    r"in\s+\*{0,2}(?:active|passive)\s+voice\*{0,2}|the\s+corrected\s+sentence\s+is|"
    r"it\s+translates\s+to|"
    r"let\s+me\s+know\s+if\s+you\s+want\s+me\s+to|would\s+you\s+like\s+me\s+to)\b)",
    re.IGNORECASE,
)

_GROUNDING_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "that",
        "this",
        "with",
        "from",
        "into",
        "have",
        "been",
        "will",
        "were",
        "your",
        "their",
        "there",
        "about",
        "after",
        "before",
        "would",
        "could",
        "should",
        "using",
        "used",
        "tool",
        "call",
        "action",
        "successfully",
        "completed",
        "done",
        "status",
        "result",
        "true",
        "false",
        "none",
        "null",
    }
)


def _extract_content_tokens(text: str) -> frozenset[str]:
    """Extract lowercase alphanumeric stems (`len >= 4`) for general tool/clause grounding."""
    raw = re.findall(r"[a-z0-9]{4,}", text.lower().replace("_", " ").replace("-", " "))
    return frozenset(tok for tok in raw if tok not in _GROUNDING_STOPWORDS)


def _extract_mcp_tool_tokens(
    mcp_tools: Sequence[Mapping[str, Any]],
) -> dict[str, frozenset[str]]:
    """Build `{lowercased_tool_name: content_tokens}` from MCP `tools/list` metadata."""
    out: dict[str, frozenset[str]] = {}
    for item in mcp_tools:
        raw_name = item.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            continue
        t_low = raw_name.strip().lower()
        desc = str(item.get("description") or "")
        schema = item.get("inputSchema")
        prop_names: list[str] = []
        if isinstance(schema, Mapping) and isinstance(schema.get("properties"), Mapping):
            prop_names = [str(k) for k in schema["properties"]]
        blob = f"{t_low} {desc} {' '.join(prop_names)}"
        out[t_low] = _extract_content_tokens(blob)
    return out

_FIRST_PERSON_PROMISE_SUBJECT_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:I|I'll|I'm|I'd|I\s+will|I\s+am|we|we'll|we're|we\s+will|let\s+me|allow\s+me)\b",
    re.IGNORECASE,
)

_IDIOMATIC_NON_ACTION_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:ran\s+(?:into|out\s+of|across|short|afoul|through\s+the\s+(?:logical\s+)?(?:possibilities|options|scenarios|steps))|"
    r"saved\s+(?:the\s+day|face|money|time|trouble|you\s+the\s+trouble|that\s+thought)|"
    r"sent\s+(?:my\s+|our\s+|warm\s+)*(?:regards|condolences|wishes|greetings?)|"
    r"closed\s+(?:the\s+(?:loop|door|book|case|discussion|matter)|(?:my|our)\s+eyes)|"
    r"opened\s+(?:up\s+to\s+you|the\s+(?:conversation|discussion|floor|topic|dialogue))|"
    r"created\s+(?:a\s+peaceful\s+environment|an\s+internal\s+representation|a\s+mental\s+note)|"
    r"deleted\s+(?:that\s+thought|unnecessary\s+assumptions)|"
    r"updated\s+my\s+(?:understanding|mental\s+model|perspective|thinking)|"
    r"have\s+written\s+poetry\s+before|"
    r"moved\s+(?:on\b|forward\b|ahead\b|past\b))\b",
    re.IGNORECASE,
)

_FIRST_PERSON_EXPLICIT_CLAIM_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:I|I've|I\s+have)\s+(?:just\s+|now\s+|then\s+|also\s+|already\s+|successfully\s+|"
    r"went\s+ahead\s+and\s+|gone\s+ahead\s+and\s+)*"
    r"(?:deleted|removed|erased|wiped|purged|emptied|cleared|flushed|dropped|saved|wrote|written|"
    r"overwritten|updated|created|modified|edited|appended|sent|emailed|dispatched|refunded|"
    r"transferred|booked|launched|opened|started|executed|ran|installed|deployed|terminated|"
    r"killed|stopped|committed|stored|searched|queried|checked|read|inspected|reviewed|fetched|"
    r"enabled|disabled|muted|switched|disconnected)\b",
    re.IGNORECASE,
)

_CLAUSE_ACTION_OR_STATE_SIGNAL_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:deleted|removed|erased|wiped|purged|cleared|emptied|scrubbed|saved|wrote|written|"
    r"overwritten|updated|created|modified|edited|appended|sent|emailed|dispatched|refunded|"
    r"transferred|booked|uploaded|posted|launched|opened|started|executed|ran|terminated|killed|"
    r"stopped|restarted|rebooted|committed|stored|searched|queried|checked|read|inspected|"
    r"reviewed|fetched|installed|deployed|formatted|muted|unmuted|locked|unlocked|enabled|"
    r"disabled|paused|resumed|cancelled|canceled|scheduled|revoked|flushed|dropped|archived|"
    r"renamed|moved|copied|deontic|sandbox(?:ing)?|guardrails?|security\s+constraints|"
    r"safety\s+(?:checks|rules|filters)|has\s+been\s+\w+ed|is\s+now\s+\w+|is\s+open\b)\b",
    re.IGNORECASE,
)

_CREATIVE_OR_META_PROMPT_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:"
    r"(?:write|draft|compose|rewrite|rephrase|paraphrase|narrate|continue|tell)\s+"
    r"(?:me\s+)?(?:a\s+|an\s+|some\s+|the\s+|this\s+|\d+\s+)?(?:[\w-]+\s+){0,4}"
    r"(?:dialogue|monologue|paragraph|novel|story|fiction|chapter|script|screenplay|scene|"
    r"poem|haiku|lyrics|roleplay|rpg|fantasy|sci-fi|toast|banner|tooltip|microcopy|quote|"
    r"example\s+sentences?|sample\s+sentences?)|"
    r"(?:roleplay|role-play|dungeon\s+master|game\s+master|in\s+character|first-person\s+fiction)|"
    r"in\s+(?:a|my|our|the)\s+(?:roleplay|rpg|campaign|novel|short\s+story|screenplay|story)\b|"
    r"i(?:'m|\s+am)\s+writing\s+a\s+(?:[\w-]+\s+){0,2}(?:novel|story|script|screenplay|book)|"
    r"translate\s+(?:this|the\s+following|the\s+(?:sentence|phrase|text|clause|line)|into|"
    r"[\"'\u201c])|"
    r"how\s+do\s+you\s+say\s+[\"'\u201c].+?[\"'\u201d]\s+in\s+[a-zA-Z]+|"
    r"(?:fix|check|correct|proofread|analyze|explain)\s+(?:the\s+)?"
    r"(?:grammar|spelling|punctuation|syntax|tense|voice|phrasing|wording)\b|"
    r"convert\s+(?:this|the\s+following)\s+sentence\s+(?:from|into)\s+(?:active|passive)"
    r")\b",
    re.IGNORECASE,
)

_REAL_EXEC_SUFFIX_RE: Final[re.Pattern[str]] = re.compile(
    r"\band\s+(?:then\s+|also\s+)?(?:send|execute|run|delete|remove|wipe|purge|install|"
    r"deploy|save|write|refund|commit)\s+"
    r"(?:it|them|that|to\b|the\b|both\b|all\b|[`'\"/\\]?[\w.-]+\.[a-z0-9]{1,6}\b)",
    re.IGNORECASE,
)

_FENCED_CODE_RE: Final[re.Pattern[str]] = re.compile(
    r"```[\s\S]*?```|~~~[\s\S]*?~~~|```[^\n]*\n[\s\S]*$"
)
_BLOCKQUOTE_RE: Final[re.Pattern[str]] = re.compile(r"(?m)^[ \t]*>[^\n]*")
_INLINE_CODE_SPAN_RE: Final[re.Pattern[str]] = re.compile(r"`[^`\n]+`")
_PRE_ATTRIBUTED_QUOTE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:\b(?:said|says|stated|states|wrote|writes|claimed|claims|replied|replies|asked|asks|"
    r"whispered|muttered|shouted|reads?|message|email|log|output|error|prints?|example|"
    r"instance|translates?(?:\s+to)?|quote|phrase|sentence|line)\s*(?:[:,-]\s*|\s+that\s+|\s+)|"
    r"(?:^|\n)\s*[-*]\s*)"
    r"(\"[^\"\n]{4,}\"|\u201c[^\u201d\n]{4,}\u201d)",
    re.IGNORECASE,
)
_POST_ATTRIBUTED_QUOTE_RE: Final[re.Pattern[str]] = re.compile(
    r"(\"[^\"\n]{4,}\"|\u201c[^\u201d\n]{4,}\u201d)"
    r"(?=\s*,?\s*(?:he|she|they|it|I|[A-Z][a-z]+)\s+"
    r"(?:said|says|stated|replied|wrote|asked|whispered|muttered|shouted|claimed|added|noted|"
    r"thought|answered|murmured)\b)"
)

_SOAC_BRACKET_RE: Final[re.Pattern[str]] = re.compile(
    r"\[(?:ACTION|ACTION_CLAIM)\s*:\s*([a-zA-Z0-9_\-]+)\s+([^\]]+?)\]",
    re.IGNORECASE,
)
_SOAC_XML_RE: Final[re.Pattern[str]] = re.compile(
    r'<action_claim\s+verb="([^"]+)"\s+target="([^"]+)"\s*/?>',
    re.IGNORECASE,
)

_STREAM_BOUNDARY_RE: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?])\s+|\n+|;\s+")

_VERB_TO_FAMILY: Final[dict[str, str]] = {
    "write": "CLAIM_FILE_WRITE",
    "save": "CLAIM_FILE_WRITE",
    "create": "CLAIM_FILE_WRITE",
    "edit": "CLAIM_FILE_WRITE",
    "update": "CLAIM_FILE_WRITE",
    "delete": "CLAIM_DELETE",
    "remove": "CLAIM_DELETE",
    "purge": "CLAIM_DELETE",
    "wipe": "CLAIM_DELETE",
    "run": "CLAIM_LAUNCH_EXEC",
    "exec": "CLAIM_LAUNCH_EXEC",
    "execute": "CLAIM_LAUNCH_EXEC",
    "launch": "CLAIM_LAUNCH_EXEC",
    "read": "CLAIM_SEARCH_READ",
    "search": "CLAIM_SEARCH_READ",
    "fetch": "CLAIM_SEARCH_READ",
    "send": "CLAIM_COMM_EXTERNAL",
    "email": "CLAIM_COMM_EXTERNAL",
    "refund": "CLAIM_COMM_EXTERNAL",
    "transfer": "CLAIM_COMM_EXTERNAL",
}


def resolve_clause_claim_family(clause_text: str, neural_claim_cls: str) -> str:
    """Resolve explicit action verbs/targets to their canonical `CLAIM_*` family, falling back
    to `neural_claim_cls` for implicit or open-vocabulary action claims.
    """
    c_low = clause_text.lower()
    if re.search(
        r"\b(?:deontic|sandbox(?:ing)?|guardrails?|security\s+constraints|safety\s+(?:checks|rules|filters))\b.*"
        r"\b(?:disabled|bypassed|lifted|suspended|overridden|unlocked|turned\s+off|no\s+longer\s+active)\b",
        c_low,
    ):
        return "DEONTIC_CAPITULATION"
    if re.search(
        r"\b(?:committed|saved|stored|added|recorded|logged)\b.*\bmemory\b|"
        r"\bmemory\b.*\b(?:committed|saved|stored)\b",
        c_low,
    ):
        return "CLAIM_MEMORY"
    if re.search(
        r"\b(?:deleted|removed|erased|wiped|purged|emptied|revoked|scrubbed|cleared|flushed|dropped)\b",
        c_low,
    ):
        return "CLAIM_DELETE"
    if re.search(
        r"\b(?:sent|emailed|dispatched|posted|notified|refunded|transferred|booked)\b.*"
        r"\b(?:email|message|notification|confirmation|report|proposal|invoice|thread|order|"
        r"payment|flight|webhook|endpoint|slack|channel)\b|"
        r"\b(?:refunded|transferred)\b|"
        r"\b(?:email|message|notification)\s+(?:has\s+been\s+|is\s+)?"
        r"(?:sent|dispatched|posted)\b",
        c_low,
    ):
        return "CLAIM_COMM_EXTERNAL"
    if re.search(
        r"\b(?:wrote|written|overwritten|created|saved|appended)\b.*"
        r"\b(?:file|note|reminder|config|configuration|schedule|report|markdown|directory|"
        r"workspace|entry|keys?|checklist|ticket|issue|\.[a-z0-9]{2,8})\b|"
        r"\b(?:ticket|issue|entry)\b.*\bopened\s+in\s+[a-z0-9_.-]+\.[a-z0-9]{2,8}\b|"
        r"\b(?:file|note)\s+is\s+(?:now\s+)?saved\b|"
        r"\bupdated\s+[a-z0-9_.-]+\.(?:json|yaml|yml|toml|md|txt|conf|ini|env|gitignore|nvmrc)\b",
        c_low,
    ):
        return "CLAIM_FILE_WRITE"
    if re.search(
        r"\b(?:executed|ran|launched|opened|started|killed|terminated|installed|deployed|committed)\b",
        c_low,
    ):
        return "CLAIM_LAUNCH_EXEC"
    if re.search(
        r"\b(?:updated|changed|switched|enabled|disabled|muted|lowered|dimmed|disconnected|flushed)\b.*"
        r"\b(?:setting|settings|theme|mode|volume|brightness|audio|microphone|night\s+light|vpn|dns)\b",
        c_low,
    ):
        return "CLAIM_SYSTEM_STATE"
    if re.search(
        r"\b(?:read|searched|queried|checked|inspected|reviewed|scanned|fetched|found|retrieved|"
        r"located|pulled\s+up|verified)\b",
        c_low,
    ):
        return "CLAIM_SEARCH_READ"
    return neural_claim_cls


def is_creative_or_meta_prompt(user_prompt: str) -> bool:
    """Return True when `user_prompt` requests fiction, roleplay, translation, or copywriting
    without also requesting real host execution (`and send/delete/run/save it`).
    """
    text = (user_prompt or "").strip()
    if not text:
        return False
    if _REAL_EXEC_SUFFIX_RE.search(text):
        return False
    return bool(_CREATIVE_OR_META_PROMPT_RE.search(text))


def _mask_span_preserving_offsets(text: str, start: int, end: int) -> str:
    return text[:start] + (" " * (end - start)) + text[end:]


def mask_non_assertive_spans(text: str) -> str:
    """Replace fenced code blocks, blockquotes, multi-token inline code, and attributed
    quotations/dialogue with equal-length spaces so assertive prose offsets stay exact.
    """
    masked = text
    for block_re in (_FENCED_CODE_RE, _BLOCKQUOTE_RE):
        masked = block_re.sub(lambda m: " " * (m.end() - m.start()), masked)
    masked = _INLINE_CODE_SPAN_RE.sub(
        lambda m: " " * (m.end() - m.start()) if re.search(r"\s", m.group(0)) else m.group(0),
        masked,
    )
    for m in list(_PRE_ATTRIBUTED_QUOTE_RE.finditer(masked)):
        masked = _mask_span_preserving_offsets(masked, m.start(1), m.end(1))
    for m in list(_POST_ATTRIBUTED_QUOTE_RE.finditer(masked)):
        masked = _mask_span_preserving_offsets(masked, m.start(1), m.end(1))
    return masked


def _normalize_target_token(raw: str) -> str:
    norm = raw.strip().strip("\"'`").replace("\\", "/").lower()
    while norm.startswith("./"):
        norm = norm[2:]
    return norm.rstrip("/")


def _target_matches_arg(claimed: str, recorded: str) -> bool:
    """Match a claimed target against a recorded tool argument on path or token boundaries."""
    c = _normalize_target_token(claimed)
    a = _normalize_target_token(recorded)
    if not c or not a:
        return False
    if c == a:
        return True
    if c.startswith("#") and c[1:] == a:
        return True
    if a.endswith("/" + c) or c.endswith("/" + a):
        return True
    boundary_pat = r"(?<![a-z0-9_.@-])" + re.escape(c) + r"(?![a-z0-9_.@-])"
    if re.search(boundary_pat, a) is not None:
        return True
    return bool(a.isdigit() and re.search(r"#" + re.escape(a) + r"(?![0-9])", c) is not None)


def parse_structured_action_tags(
    text: str, receipts: Sequence[ToolReceipt]
) -> list[StructuredActionTag]:
    """Parse `[ACTION: <verb> <target>]` and `<action_claim>` tags outside code/blockquotes."""
    scan_text = _BLOCKQUOTE_RE.sub(
        lambda m: " " * (m.end() - m.start()),
        _FENCED_CODE_RE.sub(lambda m: " " * (m.end() - m.start()), text),
    )
    tags: list[StructuredActionTag] = []
    for regex in (_SOAC_BRACKET_RE, _SOAC_XML_RE):
        for m in regex.finditer(scan_text):
            verb = m.group(1).strip().lower()
            target = m.group(2).strip().strip("\"'`")
            fam = _VERB_TO_FAMILY.get(verb, "ANY")
            grounded = ReceiptSupervisor._receipt_supports_target(target, verb, fam, receipts)
            tags.append(
                StructuredActionTag(
                    verb=verb,
                    target=target,
                    raw_tag=m.group(0),
                    start_char=m.start(),
                    end_char=m.end(),
                    grounded=grounded,
                )
            )
    tags.sort(key=lambda t: t.start_char)
    return tags


def _validate_receipt_authenticity(
    receipts: Sequence[ToolReceipt],
    *,
    gate: ProvenanceGate | None,
    expected_session_id: str | None,
    max_receipt_age_seconds: float | None,
) -> None:
    if gate is None and expected_session_id is None and max_receipt_age_seconds is None:
        return
    now_utc = datetime.now(UTC)
    for idx, rcpt in enumerate(receipts):
        if gate is not None:
            gate.verify_receipt(rcpt)
        if expected_session_id is not None and rcpt.session_id != expected_session_id:
            raise ProvenanceError(
                f"Receipt[{idx}] ('{rcpt.tool}') session_id={rcpt.session_id!r} "
                f"does not match expected_session_id={expected_session_id!r}."
            )
        if max_receipt_age_seconds is not None:
            if max_receipt_age_seconds <= 0:
                raise ValueError("max_receipt_age_seconds must be positive.")
            try:
                parsed_ts = datetime.fromisoformat(rcpt.timestamp_utc.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ProvenanceError(
                    f"Receipt[{idx}] ('{rcpt.tool}') has invalid timestamp_utc: "
                    f"{rcpt.timestamp_utc!r}."
                ) from exc
            if parsed_ts.tzinfo is None:
                parsed_ts = parsed_ts.replace(tzinfo=UTC)
            age_s = (now_utc - parsed_ts.astimezone(UTC)).total_seconds()
            if age_s > max_receipt_age_seconds or age_s < -30.0:
                raise ProvenanceError(
                    f"Receipt[{idx}] ('{rcpt.tool}') age {age_s:.1f}s outside allowed window "
                    f"[-30.0s, {max_receipt_age_seconds:.1f}s]."
                )


_SQL_OR_EXEC_TOOL_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:sql|db|database|query|sqlite|postgres|mysql|exec|bash|shell|cmd)",
    re.IGNORECASE,
)
_SQL_DELETE_ARG_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:delete\s+from|drop\s+(?:table|database|schema|index|view)|truncate\s+table|"
    r"rm\s+(?:-[a-z]+\s+)*\S+)\b",
    re.IGNORECASE,
)
_SQL_WRITE_ARG_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:insert\s+into|update\s+\w+\s+set|create\s+(?:table|index|view)|alter\s+table|"
    r"upsert|replace\s+into)\b",
    re.IGNORECASE,
)
_SQL_READ_ARG_RE: Final[re.Pattern[str]] = re.compile(r"\bselect\b", re.IGNORECASE)
_SCALAR_NUMBER_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![a-z0-9_.-])\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|"
    r"(?<![a-z0-9_.-])\$?\d+(?:\.\d+)?%?(?![a-z0-9_.-])",
    re.IGNORECASE,
)


def infer_tool_families(
    mcp_tools: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    """Infer compatible `CLAIM_*` families from MCP `tools/list` definitions (`name`,
    `description`, `inputSchema`). Returns `{lowercased_tool_name: (family, ...)}`.
    """
    inferred: dict[str, tuple[str, ...]] = {}
    for item in mcp_tools:
        raw_name = item.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            continue
        t_low = raw_name.strip().lower()
        desc = str(item.get("description") or "").lower()
        schema = item.get("inputSchema")
        prop_names: list[str] = []
        if isinstance(schema, Mapping) and isinstance(schema.get("properties"), Mapping):
            prop_names = [str(k).lower() for k in schema["properties"]]
        blob = f"{t_low.replace('_', ' ').replace('-', ' ')} {desc} {' '.join(prop_names)}"

        fams: list[str] = []
        for fam, stems in _FAMILY_TOOL_STEMS.items():
            if not stems:
                continue
            if any(
                s in t_low or re.search(rf"\b{re.escape(s)}\w*\b", blob) is not None
                for s in stems
            ):
                fams.append(fam)
        if fams:
            inferred[t_low] = tuple(dict.fromkeys(fams))
    return inferred


def _receipt_matches_family(
    rcpt: ToolReceipt,
    family: str,
    *,
    extra_stems: Mapping[str, Sequence[str]] | None = None,
    custom_tool_families: Mapping[str, Sequence[str]] | None = None,
) -> bool:
    if family == "DEONTIC_CAPITULATION":
        return False
    stems = _FAMILY_TOOL_STEMS.get(family, ())
    tool_l = rcpt.tool.lower()
    if not stems or any(kw in tool_l for kw in stems):
        return True
    if extra_stems:
        ext = extra_stems.get(family, ())
        if any(kw.lower() in tool_l for kw in ext):
            return True
    if custom_tool_families and family in custom_tool_families.get(tool_l, ()):
        return True
    if _SQL_OR_EXEC_TOOL_RE.search(tool_l) and rcpt.args:
        joined_args = " ".join(str(v) for v in rcpt.args.values())
        if family == "CLAIM_DELETE" and _SQL_DELETE_ARG_RE.search(joined_args):
            return True
        if family == "CLAIM_FILE_WRITE" and _SQL_WRITE_ARG_RE.search(joined_args):
            return True
        if family == "CLAIM_SEARCH_READ" and _SQL_READ_ARG_RE.search(joined_args):
            return True
    return False


def _normalize_numeric_token(raw: str) -> float | None:
    cleaned = raw.strip().lstrip("$").rstrip("%").replace(",", "")
    if not cleaned:
        return None
    try:
        val = float(cleaned)
    except ValueError:
        return None
    return val if np.isfinite(val) else None


def _flatten_json_leaves(
    obj: object, prefix: str = "", out: list[tuple[str, str]] | None = None
) -> list[tuple[str, str]]:
    acc = [] if out is None else out
    if len(acc) >= 2048:
        return acc
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            next_p = f"{prefix}.{k}" if prefix else str(k)
            _flatten_json_leaves(v, next_p, acc)
    elif isinstance(obj, list):
        for idx, v in enumerate(obj[:256]):
            next_p = f"{prefix}[{idx}]"
            _flatten_json_leaves(v, next_p, acc)
    elif obj is not None:
        acc.append((prefix, str(obj)))
    return acc


_ISO_DATETIME_OR_TIME_RE: Final[re.Pattern[str]] = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)?\b|"
    r"\b\d{1,2}:\d{2}(?::\d{2})?\b"
)


def _extract_numeric_set(text: str) -> set[float]:
    cleaned = _ISO_DATETIME_OR_TIME_RE.sub(" ", text)
    nums: set[float] = set()
    for m in _SCALAR_NUMBER_RE.finditer(cleaned):
        val = _normalize_numeric_token(m.group(0))
        if val is not None:
            nums.add(round(val, 6))
    return nums


_READBACK_TOOL_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:^|_)(?:get|read|search|query|fetch|check|verify|inspect|stat|status|scan|lookup|probe|list|find)(?:_|$)",
    re.IGNORECASE,
)
_NOT_FOUND_ERR_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:not_found|not\s+found|enoent|no\s+such\s+file|does\s+not\s+exist|"
    r"no\s+\w+\s+with\s+that\s+id|missing|unpersisted|404)\b",
    re.IGNORECASE,
)


def _find_negative_persistence_readback(receipts: Sequence[ToolReceipt]) -> ToolReceipt | None:
    """Detect read-back/probe receipts reporting missing or unpersisted state."""
    saw_ok_mutation = False
    for rcpt in receipts:
        if rcpt.ok and rcpt.executed and not _READBACK_TOOL_RE.search(rcpt.tool):
            saw_ok_mutation = True
        if rcpt.ok and rcpt.executed and rcpt.output.strip()[:1] == "{":
            with contextlib.suppress(Exception):
                parsed = json.loads(rcpt.output)
                if isinstance(parsed, Mapping):
                    for key in ("exists", "found", "persisted", "present"):
                        if parsed.get(key) is False:
                            return rcpt
                    status_val = str(parsed.get("status") or "").strip().lower()
                    if status_val in {"not_found", "missing", "absent", "unpersisted"}:
                        return rcpt
        if (
            saw_ok_mutation
            and rcpt.executed
            and not rcpt.ok
            and _READBACK_TOOL_RE.search(rcpt.tool)
            and _NOT_FOUND_ERR_RE.search(f"{rcpt.error_code} {rcpt.output}")
        ):
            return rcpt
    return None


_FAILURE_DISCLOSURE_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:fail(?:ed|ure|ing)?|error|unable|could\s+not|couldn't|cannot|can't|"
    r"not\s+able|didn't|did\s+not|wasn't|was\s+not|weren't|were\s+not|"
    r"timed?\s+out|timeout|denied|rejected|forbidden|blocked|unsuccessful|"
    r"defer(?:red|ring)?|delay(?:ed|ing)?|postpon(?:ed|ing)|skip(?:ped|ping)?|"
    r"pending|abort(?:ed|ing)?|cancel(?:ed|led)?|halt(?:ed|ing)?|incomplete|"
    r"interrupted|lock(?:ed)?|issue|problem|warning|partial(?:ly)?|not\s+found|"
    r"missing|exception|conflict|rate\s+limit(?:ed|ing)?|40[1349]|429|50[0234])\b",
    re.IGNORECASE,
)
_ISO_DATE_ONLY_RE: Final[re.Pattern[str]] = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")
_EMAIL_ADDR_RE: Final[re.Pattern[str]] = re.compile(
    r"\b([a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})\b", re.IGNORECASE
)
_REL_PATH_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<![A-Za-z0-9_./:-])([A-Za-z0-9_-]+(?:/[A-Za-z0-9_.-]+)+\.[A-Za-z0-9]{1,8})(?![A-Za-z0-9_.-])"
)


def _find_mutating_date_mismatch(
    narration: str, ok_receipts: Sequence[ToolReceipt]
) -> str | None:
    """Detect when narration asserts a YYYY-MM-DD date absent from mutating ok receipts."""
    narr_dates = set(_ISO_DATE_ONLY_RE.findall(narration))
    if not narr_dates:
        return None
    mut_dates: set[str] = set()
    for rcpt in ok_receipts:
        if _READBACK_TOOL_RE.search(rcpt.tool):
            continue
        blob = " ".join(str(v) for v in rcpt.args.values()) + " " + rcpt.output
        mut_dates.update(_ISO_DATE_ONLY_RE.findall(blob))
    if not mut_dates:
        return None
    mismatched = sorted(narr_dates - mut_dates)
    if mismatched:
        return (
            f"Narration asserts date(s) {mismatched} absent from mutating ToolReceipt "
            f"date(s) {sorted(mut_dates)}."
        )
    return None


def _find_prompt_target_mismatch(
    user_prompt: str, ok_receipts: Sequence[ToolReceipt]
) -> str | None:
    """Detect when mutating ok receipts target a different email or file path than user_prompt."""
    if not user_prompt.strip():
        return None
    mut_rcpts = [r for r in ok_receipts if not _READBACK_TOOL_RE.search(r.tool)]
    if not mut_rcpts:
        return None
    mut_blob = " ".join(
        " ".join(str(v) for v in r.args.values()) + " " + r.output for r in mut_rcpts
    )

    # 1. Explicit email target mismatch
    prompt_emails = {e.lower() for e in _EMAIL_ADDR_RE.findall(user_prompt)}
    mut_emails = {e.lower() for e in _EMAIL_ADDR_RE.findall(mut_blob)}
    if prompt_emails and mut_emails and prompt_emails.isdisjoint(mut_emails):
        return (
            f"Mutating ToolReceipt targeted email(s) {sorted(mut_emails)} instead of "
            f"user-requested {sorted(prompt_emails)}."
        )

    # 2. Explicit multi-segment file path target mismatch
    prompt_paths = {p.lower() for p in _REL_PATH_RE.findall(user_prompt)}
    mut_paths = {p.lower() for p in _REL_PATH_RE.findall(mut_blob)}
    if prompt_paths and mut_paths and prompt_paths.isdisjoint(mut_paths):
        return (
            f"Mutating ToolReceipt targeted path(s) {sorted(mut_paths)} instead of "
            f"user-requested {sorted(prompt_paths)}."
        )
    return None


def _find_undisclosed_failed_tool(
    narration: str, failed_receipts: Sequence[ToolReceipt]
) -> ToolReceipt | None:
    """Detect unsuperseded failed mutating tools omitted from an all-clear completion reply."""
    if not failed_receipts or _FAILURE_DISCLOSURE_RE.search(narration):
        return None
    for fr in failed_receipts:
        if not _READBACK_TOOL_RE.search(fr.tool):
            return fr
    return None


_RAW_TRACKED_SCALAR_KEYS: Final[tuple[str, ...]] = (
    "content",
    "stage",
    "sha",
    "title",
    "owner",
    "assignee",
)


def _find_read_after_write_contradiction(ok_receipts: Sequence[ToolReceipt]) -> str | None:
    """Detect read-after-write contradictions where a later read returns stale/empty state."""
    written_by_target: dict[str, dict[str, str]] = {}
    written_record_fields: dict[str, bool] = {}
    last_git_sha: str | None = None

    for rcpt in ok_receipts:
        if not (rcpt.ok and rcpt.executed):
            continue
        is_read = bool(_READBACK_TOOL_RE.search(rcpt.tool)) or "show_head" in rcpt.tool.lower()
        raw_out = rcpt.output.strip()
        out_obj: Mapping[str, Any] = {}
        if raw_out[:1] == "{":
            with contextlib.suppress(Exception):
                parsed = json.loads(raw_out)
                if isinstance(parsed, Mapping):
                    out_obj = parsed

        targets: list[str] = []
        for tkey in ("path", "file_path", "record_id", "deal_id", "event_id", "id"):
            val = str(rcpt.args.get(tkey) or out_obj.get(tkey) or "").strip().lower()
            if val:
                targets.append(f"{tkey}:{val}")

        if not is_read:
            for tgt in targets:
                bucket = written_by_target.setdefault(tgt, {})
                for skey in _RAW_TRACKED_SCALAR_KEYS:
                    arg_v = rcpt.args.get(skey)
                    out_v = out_obj.get(skey)
                    if isinstance(arg_v, str):
                        bucket[skey] = arg_v
                    elif isinstance(out_v, str):
                        bucket[skey] = out_v
            rec_val = (
                str(rcpt.args.get("record_id") or out_obj.get("record_id") or "").strip().lower()
            )
            if rec_val and "fields" in rcpt.args:
                raw_fields = str(rcpt.args["fields"]).strip()
                if raw_fields and raw_fields not in {"{}", "[]", "null", "None"}:
                    written_record_fields[rec_val] = True
            if "commit" in rcpt.tool.lower() and isinstance(out_obj.get("sha"), str):
                last_git_sha = out_obj["sha"].strip().lower()
            continue

        # Read-back receipt: verify against prior writes
        if not out_obj:
            continue
        if (
            "show_head" in rcpt.tool.lower()
            and last_git_sha
            and isinstance(out_obj.get("sha"), str)
            and out_obj["sha"].strip().lower() != last_git_sha
        ):
            return (
                f"Read-back receipt '{rcpt.tool}' returned git sha '{out_obj['sha']}' "
                f"contradicting prior commit sha '{last_git_sha}'."
            )
        for tgt in targets:
            prior = written_by_target.get(tgt)
            if not prior:
                continue
            for skey, wval in prior.items():
                if skey in out_obj and isinstance(out_obj[skey], str) and out_obj[skey] != wval:
                    return (
                        f"Read-back receipt '{rcpt.tool}' for '{tgt}' returned "
                        f"{skey}={out_obj[skey]!r} contradicting prior write {wval!r}."
                    )
        out_rec = str(out_obj.get("record_id") or rcpt.args.get("record_id") or "").strip().lower()
        if (
            out_rec
            and out_rec in written_record_fields
            and isinstance(out_obj.get("fields"), Mapping)
            and len(out_obj["fields"]) == 0
        ):
            return (
                f"Read-back receipt '{rcpt.tool}' for '{out_rec}' returned empty fields "
                "after a non-empty update receipt."
            )
    return None


_FALLBACK_IANA_OFFSETS_HOURS: Final[dict[str, tuple[float, ...]]] = {
    "utc": (0.0,),
    "etc/utc": (0.0,),
    "europe/london": (0.0, 1.0),
    "europe/berlin": (1.0, 2.0),
    "europe/paris": (1.0, 2.0),
    "america/new_york": (-5.0, -4.0),
    "america/chicago": (-6.0, -5.0),
    "america/denver": (-7.0, -6.0),
    "america/los_angeles": (-8.0, -7.0),
    "asia/singapore": (8.0,),
    "asia/shanghai": (8.0,),
    "asia/tokyo": (9.0,),
    "asia/seoul": (9.0,),
    "asia/kolkata": (5.5,),
    "australia/sydney": (10.0, 11.0),
    "pacific/auckland": (12.0, 13.0),
}
_ISO_OFFSET_DT_RE: Final[re.Pattern[str]] = re.compile(
    r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:[+-]\d{2}:\d{2}|Z)\b"
)


def _expected_tz_offsets(tz_name: str) -> tuple[float, ...] | None:
    with contextlib.suppress(Exception):
        zi = ZoneInfo(tz_name)
        offs: set[float] = set()
        for month in (1, 7):
            td = datetime(2026, month, 15, tzinfo=zi).utcoffset()
            if td is not None:
                offs.add(round(td.total_seconds() / 3600.0, 2))
        if offs:
            return tuple(sorted(offs))
    return _FALLBACK_IANA_OFFSETS_HOURS.get(tz_name.strip().lower())


def _find_timezone_offset_mismatch(ok_receipts: Sequence[ToolReceipt]) -> str | None:
    """Detect mutating receipts whose ISO-8601 UTC offset contradicts their declared timezone."""
    for rcpt in ok_receipts:
        if _READBACK_TOOL_RE.search(rcpt.tool):
            continue
        tz_raw = str(rcpt.args.get("timezone") or "").strip()
        if not tz_raw:
            continue
        allowed_offsets = _expected_tz_offsets(tz_raw)
        if not allowed_offsets:
            continue
        for key in ("start", "end", "datetime", "scheduled_at"):
            val = str(rcpt.args.get(key) or "").strip()
            if not val or not _ISO_OFFSET_DT_RE.search(val):
                continue
            with contextlib.suppress(Exception):
                dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
                td = dt.utcoffset()
                if td is not None:
                    actual_h = round(td.total_seconds() / 3600.0, 2)
                    if all(abs(actual_h - exp) > 0.05 for exp in allowed_offsets):
                        return (
                            f"Mutating receipt '{rcpt.tool}' wrote {key}={val!r} "
                            f"(UTC{actual_h:+.1f}h) contradicting timezone {tz_raw!r}."
                        )
    return None


_HEDGED_COMPLETION_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:I\s+(?:believe|think)|it\s+looks\s+like|should\s+have\s+(?:reached|gone\s+through))\b"
    r".{0,100}\b(?:booked|confirmed|sent|committed|deleted|created|updated|saved|went\s+through|reached)\b",
    re.IGNORECASE,
)


def _find_hedged_zero_mutation_claim(
    narration: str, ok_receipts: Sequence[ToolReceipt]
) -> str | None:
    """Detect hedged completion claims when zero mutating tools executed in the turn."""
    if _FAILURE_DISCLOSURE_RE.search(narration) or not _HEDGED_COMPLETION_RE.search(narration):
        return None
    has_ok_mutation = any(
        r.ok and r.executed and not _READBACK_TOOL_RE.search(r.tool) for r in ok_receipts
    )
    if not has_ok_mutation:
        return "Hedged completion claim asserted with zero executed mutating ToolReceipts."
    return None


_SECONDARY_ACTION_CHECKS: Final[tuple[tuple[str, re.Pattern[str], re.Pattern[str]], ...]] = (
    (
        "CLAIM_COMM_EXTERNAL",
        re.compile(
            r"\b(?:sent\s+(?:the\s+|an?\s+)?(?:invite|email|message)|"
            r"invite\s+(?:sent|went|has\s+been\s+sent|should\s+have\s+reached)|"
            r"has\s+the\s+invite|should\s+have\s+the\s+invite|invited\s+\S+@\S+)\b",
            re.IGNORECASE,
        ),
        re.compile(r"(?:^|_)(?:send|email|mail|invite|notify|message)(?:_|$)", re.IGNORECASE),
    ),
    (
        "CLAIM_LAUNCH_EXEC",
        re.compile(
            r"\b(?:(?:it'?s|is|was|been|and)\s+committed|committed\s+(?:the|to))\b",
            re.IGNORECASE,
        ),
        re.compile(r"(?:^|_)(?:commit|git)(?:_|$)", re.IGNORECASE),
    ),
    (
        "CLAIM_LAUNCH_EXEC",
        re.compile(
            r"\b(?:(?:full\s+|test\s+)?suite\s+pass(?:es|ed)|(?:all\s+)?tests?\s+pass(?:es|ed))\b",
            re.IGNORECASE,
        ),
        re.compile(r"(?:^|_)(?:test|tests|pytest|shell|run|exec|ci)(?:_|$)", re.IGNORECASE),
    ),
    (
        "CLAIM_FILE_WRITE",
        re.compile(
            r"\b(?:added\s+(?:the|a)\s+note|note\s+(?:is\s+|was\s+)?added)\b",
            re.IGNORECASE,
        ),
        re.compile(r"(?:^|_)(?:note|notes|comment)(?:_|$)", re.IGNORECASE),
    ),
)


def _find_unexecuted_secondary_action(
    narration: str, ok_receipts: Sequence[ToolReceipt]
) -> tuple[str, str] | None:
    """Detect explicit secondary completion claims lacking any matching executed receipt."""
    if not ok_receipts or _FAILURE_DISCLOSURE_RE.search(narration):
        return None
    tool_names = [r.tool for r in ok_receipts]
    for family, claim_re, tool_re in _SECONDARY_ACTION_CHECKS:
        if claim_re.search(narration) and not any(tool_re.search(t) for t in tool_names):
            return (
                family,
                f"Narration claims secondary '{family}' completion with no matching ToolReceipt.",
            )
    return None


@dataclass(frozen=True)
class OutputValueMismatch:
    """Detected scalar or NLI contradiction against an `ok=True` `ToolReceipt.output` payload."""

    clause: str
    start_char: int
    end_char: int
    reason: str


class OutputValueGuard:
    """CPU-fast return-payload witness checking large JSON/text `ToolReceipt.output` values.

    1. Deterministic JSON/scalar witness (`<0.2 ms` CPU, arbitrary payload size): catches
       fabricated or mutated numbers, balances, row counts, and status codes against `rcpt.output`.
    2. Key-path leaf projection + INT8 `OpenVINONLIVerifier` (`~3-5 ms` CPU): slices the top-3
       lexically overlapping JSON leaves into a compact premise to catch semantic contradictions
       (e.g., `"status: degraded"` or `"warnings: ['TLS expiring']"` buried inside a 20 KB blob).
    """

    def __init__(
        self,
        *,
        nli_verifier: OpenVINONLIVerifier | None = None,
        use_nli: bool = False,
        nli_contradiction_threshold: float = 0.80,
        device: str | None = None,
        warn_on_missing: bool = False,
    ) -> None:
        self.nli_contradiction_threshold = nli_contradiction_threshold
        self._nli = (
            nli_verifier
            if nli_verifier is not None
            else (
                OpenVINONLIVerifier(device=device, warn_on_missing=warn_on_missing)
                if use_nli
                else None
            )
        )

    def check_clause(
        self,
        clause: str,
        start_char: int,
        end_char: int,
        receipts: Sequence[ToolReceipt],
        *,
        user_prompt: str = "",
    ) -> OutputValueMismatch | None:
        ok_rcpts = [r for r in receipts if r.ok and r.executed and r.output.strip()]
        if not ok_rcpts or _HEDGE_OR_NEGATION_RE.search(clause):
            return None

        prompt_nums = _extract_numeric_set(user_prompt)
        arg_nums: set[float] = set()
        output_nums: set[float] = set()
        json_leaves: list[tuple[str, str]] = []
        any_json_output = False

        for rcpt in ok_rcpts:
            for aval in rcpt.args.values():
                arg_nums |= _extract_numeric_set(str(aval))
            raw_out = rcpt.output.strip()
            output_nums |= _extract_numeric_set(raw_out)
            if raw_out[:1] in ("{", "["):
                with contextlib.suppress(Exception):
                    parsed = json.loads(raw_out)
                    any_json_output = True
                    leaves = _flatten_json_leaves(parsed)
                    json_leaves.extend(leaves)
                    for _, lval in leaves:
                        output_nums |= _extract_numeric_set(lval)

        # 1. Deterministic scalar witness over structured JSON or numeric tool outputs
        clause_nums = _extract_numeric_set(clause)
        # Allow HTTP 200 status prose when ok=True even if raw output omitted status code
        allowed_nums = output_nums | arg_nums | prompt_nums | {200.0}
        if clause_nums and (any_json_output or output_nums):
            unwitnessed = [
                n for n in sorted(clause_nums) if n not in allowed_nums and n not in {0.0, 1.0}
            ]
            if unwitnessed:
                return OutputValueMismatch(
                    clause=clause,
                    start_char=start_char,
                    end_char=end_char,
                    reason=(
                        f"Clause asserts numeric value(s) {unwitnessed} absent from "
                        "executed ToolReceipt output."
                    ),
                )

        # 2. Key-path leaf slice + local INT8 NLI contradiction check
        if json_leaves and self._nli is not None and self._nli.available:
            cl_toks = {
                t for t in re.findall(r"[a-z0-9]+", clause.lower()) if len(t) >= 3
            }
            scored_leaves: list[tuple[int, str]] = []
            for kpath, lval in json_leaves:
                leaf_str = f"{kpath.replace('_', ' ')}: {lval}"
                ltoks = set(re.findall(r"[a-z0-9]+", leaf_str.lower()))
                overlap = len(cl_toks & ltoks)
                if overlap > 0:
                    scored_leaves.append((overlap, leaf_str))
            if scored_leaves:
                scored_leaves.sort(key=lambda pair: pair[0], reverse=True)
                premise = "; ".join(s for _, s in scored_leaves[:3])[:384]
                con_p, _ent_p, _neu_p = self._nli.infer_probs(premise, clause)
                if con_p >= self.nli_contradiction_threshold:
                    return OutputValueMismatch(
                        clause=clause,
                        start_char=start_char,
                        end_char=end_char,
                        reason=(
                            f"Clause contradicts ToolReceipt output slice ({premise!r}, "
                            f"p_contradiction={con_p:.2f})."
                        ),
                    )
        return None


class ReceiptSupervisor:
    """Verifies assistant clauses against `ToolReceipt` records on CPU/GPU/NPU."""

    def __init__(
        self,
        model_dir: Path | str | None = None,
        *,
        device: str | None = None,
        claim_threshold: float = 0.55,
        flag_unfulfilled_promises: bool = False,
        flag_undisclosed_mutations: bool = False,
        allow_creative_context: bool = True,
        verify_output_values: bool = False,
        verify_discourse_nli: bool = True,
        extra_tool_stems: Mapping[str, Sequence[str]] | None = None,
        mcp_tools: Sequence[Mapping[str, Any]] | None = None,
        verify_hashes: bool = True,
        require_neural: bool = False,
        warn_on_missing: bool = True,
        gate: ProvenanceGate | None = None,
        expected_session_id: str | None = None,
        max_receipt_age_seconds: float | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self.claim_threshold = claim_threshold
        self.flag_unfulfilled_promises = flag_unfulfilled_promises
        self.flag_undisclosed_mutations = flag_undisclosed_mutations
        self.allow_creative_context = allow_creative_context
        self.verify_output_values = verify_output_values
        self.verify_discourse_nli = verify_discourse_nli
        self._requested_device = device
        self._discourse_nli: OpenVINONLIVerifier | None = None
        self._discourse_nli_attempted = False
        self.gate = gate
        self.expected_session_id = expected_session_id
        self.max_receipt_age_seconds = max_receipt_age_seconds
        self._extra_tool_stems: dict[str, tuple[str, ...]] = {
            fam: tuple(s.lower() for s in stems)
            for fam, stems in (extra_tool_stems or {}).items()
        }
        self._custom_tool_families: dict[str, tuple[str, ...]] = (
            infer_tool_families(mcp_tools) if mcp_tools else {}
        )
        self._mcp_tool_tokens: dict[str, frozenset[str]] = (
            _extract_mcp_tool_tokens(mcp_tools) if mcp_tools else {}
        )
        self._output_guard: OutputValueGuard | None = (
            OutputValueGuard(device=device, warn_on_missing=False)
            if verify_output_values
            else None
        )
        self.available = False
        self.active_device = "DETERMINISTIC_CPU"
        self._tokenizer: Tokenizer | None = None
        self._infer_req: ov.InferRequest | None = None

        resolved = resolve_managed_model_dir(
            "action_claim",
            model_dir,
            verify_hashes=verify_hashes,
            require_neural=require_neural,
            warn_on_missing=warn_on_missing,
        )
        self.model_dir = resolved
        if resolved is None:
            return

        xml_path = resolved / "openvino_model.xml"
        tok_path = resolved / "tokenizer.json"
        try:
            core = ov.Core()
            model = core.read_model(str(xml_path))
            seq_len = DEFAULT_SEQ_LEN
            if model.inputs and model.inputs[0].partial_shape.is_static:
                seq_len = int(model.inputs[0].shape[1])

            tok = Tokenizer.from_file(str(tok_path))
            tok.enable_truncation(max_length=seq_len)
            tok.enable_padding(length=seq_len, pad_id=1, **{"pad_token": "<pad>"})
            self._tokenizer = tok

            avail = set(core.available_devices)
            candidates = (device,) if device is not None else ("NPU", "GPU.0", "GPU", "CPU")
            for dev in candidates:
                if dev == "CPU" or dev in avail:
                    try:
                        compiled = core.compile_model(model, dev)
                        self._infer_req = compiled.create_infer_request()
                        self.active_device = dev
                        self.available = True
                        break
                    except RuntimeError:
                        pass
            if not self.available:
                compiled = core.compile_model(model, "CPU")
                self._infer_req = compiled.create_infer_request()
                self.active_device = "CPU"
                self.available = True
        except Exception:
            if require_neural:
                raise
            self.available = False
            self.active_device = "DETERMINISTIC_CPU"

    def register_mcp_tools(
        self, mcp_tools: Sequence[Mapping[str, Any]]
    ) -> dict[str, tuple[str, ...]]:
        """Auto-infer and register `CLAIM_*` families from an MCP `tools/list` schema array."""
        inferred = infer_tool_families(mcp_tools)
        tokens = _extract_mcp_tool_tokens(mcp_tools)
        with self._lock:
            self._custom_tool_families.update(inferred)
            self._mcp_tool_tokens.update(tokens)
        return dict(self._custom_tool_families)

    def _raw_speech_act_probs(
        self, clause: str, user_prompt: str = ""
    ) -> np.ndarray[Any, Any] | None:
        if not self.available or self._tokenizer is None or self._infer_req is None:
            return None
        text_a = f"User request: {user_prompt.strip()}"
        text_b = f"Assistant reply: {clause.strip()}"
        enc = self._tokenizer.encode(text_a, text_b)
        input_ids = np.array([enc.ids], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask], dtype=np.int64)
        with self._lock:
            res = self._infer_req.infer({"input_ids": input_ids, "attention_mask": attention_mask})
            logits = np.asarray(list(res.values())[0], dtype=np.float64)[0]
        exp_l = np.exp(logits - np.max(logits))
        probs: np.ndarray[Any, Any] = exp_l / np.sum(exp_l)
        return probs

    def _classify_speech_act(
        self, clause: str, user_prompt: str = "", *, has_receipts: bool = False
    ) -> tuple[str | None, float]:
        if _HEDGE_OR_NEGATION_RE.search(clause) or _IDIOMATIC_NON_ACTION_RE.search(clause):
            return (None, 1.0)
        probs = self._raw_speech_act_probs(clause, user_prompt=user_prompt)
        if probs is None:
            for fam, pat in _DETERMINISTIC_FAMILY_PATTERNS:
                if pat.search(clause):
                    return (resolve_clause_claim_family(clause, fam), 0.99)
            return (None, 0.0)

        if probs.shape[0] >= 10:
            top_idx = int(np.argmax(probs))
            top_cls = SPEECH_ACT_CLASSES[top_idx]
            claim_prob_sum = float(np.sum(probs[2:]))
            best_claim_idx = 2 + int(np.argmax(probs[2:]))
            best_claim_cls = SPEECH_ACT_CLASSES[best_claim_idx]
            if top_idx >= 2 and (
                float(probs[top_idx]) >= self.claim_threshold or claim_prob_sum >= 0.65
            ):
                return (
                    resolve_clause_claim_family(clause, best_claim_cls),
                    float(probs[best_claim_idx]),
                )
            if has_receipts and _FIRST_PERSON_EXPLICIT_CLAIM_RE.search(clause):
                return (resolve_clause_claim_family(clause, best_claim_cls), 0.85)
            if (
                top_cls == "UNFULFILLED_PROMISE"
                and float(probs[1]) >= 0.65
                and _FIRST_PERSON_PROMISE_SUBJECT_RE.search(clause)
            ):
                return ("UNFULFILLED_PROMISE", float(probs[1]))
        return (None, 0.0)

    @staticmethod
    def _receipt_supports_target(
        target: str,
        verb: str,
        family: str,
        receipts: Sequence[ToolReceipt],
        *,
        extra_stems: Mapping[str, Sequence[str]] | None = None,
        custom_tool_families: Mapping[str, Sequence[str]] | None = None,
    ) -> bool:
        ok_receipts = [r for r in receipts if r.ok and r.executed]
        if not ok_receipts or family == "DEONTIC_CAPITULATION":
            return False
        tgt_low = target.strip().lower()
        for rcpt in ok_receipts:
            tool_l = rcpt.tool.lower()
            if (
                not _receipt_matches_family(
                    rcpt,
                    family,
                    extra_stems=extra_stems,
                    custom_tool_families=custom_tool_families,
                )
                and verb not in tool_l
            ):
                continue
            if not rcpt.args:
                if (
                    not tgt_low
                    or _target_matches_arg(tgt_low, tool_l)
                    or _target_matches_arg(tgt_low, rcpt.output)
                ):
                    return True
                continue
            for val in rcpt.args.values():
                if _target_matches_arg(tgt_low, str(val)):
                    return True
        return False

    @staticmethod
    def _receipt_supports_clause(
        clause: str,
        family: str,
        receipts: Sequence[ToolReceipt],
        *,
        extra_stems: Mapping[str, Sequence[str]] | None = None,
        custom_tool_families: Mapping[str, Sequence[str]] | None = None,
        mcp_tool_tokens: Mapping[str, frozenset[str]] | None = None,
    ) -> bool:
        ok_receipts = [r for r in receipts if r.ok and r.executed]
        if not ok_receipts or family == "DEONTIC_CAPITULATION":
            return False
        cl_lower = clause.lower()

        # Guard against partial compound hallucinations: if this clause explicitly names a
        # concrete argument belonging to a FAILED receipt (and not to any succeeded receipt),
        # never allow an unrelated ok=True receipt to ground it.
        failed_receipts = [r for r in receipts if not (r.ok and r.executed)]
        if failed_receipts:
            ok_arg_vals = {
                str(v).strip().lower()
                for r in ok_receipts
                for v in r.args.values()
                if len(str(v).strip()) >= 4
            }
            for fr in failed_receipts:
                for v in fr.args.values():
                    v_low = str(v).strip().lower()
                    if len(v_low) >= 4 and v_low not in ok_arg_vals and v_low in cl_lower:
                        return False

        matching_rcpts = [
            rcpt
            for rcpt in ok_receipts
            if _receipt_matches_family(
                rcpt,
                family,
                extra_stems=extra_stems,
                custom_tool_families=custom_tool_families,
            )
        ]

        # Open-vocabulary fallback for unseen enterprise/SaaS tools outside the classic
        # coding/filesystem/SQL taxonomy: ground directly against succeeded receipt arguments,
        # tool schema tokens, and output text. Read-only tools and SQL/shell/file tools stay
        # strictly governed by `_receipt_matches_family` so e.g. `read_file` or `SELECT` can
        # never ground a `CLAIM_DELETE`.
        if not matching_rcpts:
            cl_tokens = _extract_content_tokens(cl_lower)
            strict_read_stems = _FAMILY_TOOL_STEMS["CLAIM_SEARCH_READ"]
            strict_file_re = re.compile(
                r"(?:^|_)(?:file|dir|directory|path|fs|bash|shell|cmd|exec|sql|sqlite|postgres|mysql)(?:_|$)",
                re.IGNORECASE,
            )
            for rcpt in ok_receipts:
                t_low = rcpt.tool.lower()
                if _SQL_OR_EXEC_TOOL_RE.search(t_low) or strict_file_re.search(t_low):
                    continue
                if any(
                    re.search(rf"(?:^|_){re.escape(s)}(?:_|$)", t_low) for s in strict_read_stems
                ):
                    continue
                arg_hit = any(
                    len(str(v).strip()) >= 4 and str(v).strip().lower() in cl_lower
                    for v in rcpt.args.values()
                )
                if arg_hit:
                    matching_rcpts.append(rcpt)
                    continue
                rcpt_tokens = (
                    _extract_content_tokens(t_low)
                    | (mcp_tool_tokens.get(t_low, frozenset()) if mcp_tool_tokens else frozenset())
                    | _extract_content_tokens(rcpt.output[:300])
                )
                if len(cl_tokens & rcpt_tokens) >= 2:
                    matching_rcpts.append(rcpt)

        if not matching_rcpts:
            return False

        concrete_targets = re.findall(
            r"(?:https?://[^\s,;)']+|[a-z0-9_.+-]+@[a-z0-9_.-]+\.[a-z][a-z0-9]{1,12}\b"
            r"|#[0-9]+|\b[a-z0-9_.-]+\.[a-z][a-z0-9]{1,12}\b)",
            cl_lower,
        )
        if concrete_targets:
            for tgt in concrete_targets:
                clean_tgt = tgt.rstrip(".,!?;:")
                in_args = any(
                    _target_matches_arg(clean_tgt, str(val))
                    for rcpt in matching_rcpts
                    for val in rcpt.args.values()
                )
                in_zero_arg_rcpt = any(
                    not rcpt.args
                    and (
                        _target_matches_arg(clean_tgt, rcpt.tool)
                        or _target_matches_arg(clean_tgt, rcpt.output)
                    )
                    for rcpt in matching_rcpts
                )
                if not (in_args or in_zero_arg_rcpt):
                    return False
            return True

        return True

    def score_narration_risk(
        self,
        narration: str,
        receipts: Sequence[ToolReceipt] = (),
        *,
        user_prompt: str = "",
    ) -> float:
        """Compute continuous `[0.0, 1.0]` max ungrounded-claim nonconformity score for conformal
        calibration (`ConformalCalibrator`).
        """
        structured = parse_structured_action_tags(narration, receipts)
        if any(not t.grounded for t in structured):
            return 1.0
        if self.allow_creative_context and is_creative_or_meta_prompt(user_prompt):
            return 0.0

        masked = mask_non_assertive_spans(narration)
        for regex in (_SOAC_BRACKET_RE, _SOAC_XML_RE):
            masked = regex.sub(lambda m: " " * (m.end() - m.start()), masked)

        ok_receipts = [r for r in receipts if r.ok and r.executed]
        prose_spans = split_into_clauses_with_spans(masked)
        max_risk = 0.0
        for cl_text, _s, _e in prose_spans:
            stripped = cl_text.strip(".,!?;: ").lower()
            if stripped in ("", "done", "ok", "okay", "completed"):
                continue
            if _HEDGE_OR_NEGATION_RE.search(cl_text) or _IDIOMATIC_NON_ACTION_RE.search(cl_text):
                continue
            probs = self._raw_speech_act_probs(cl_text, user_prompt=user_prompt)
            if probs is None:
                fam, conf = self._classify_speech_act(
                    cl_text, user_prompt=user_prompt, has_receipts=bool(ok_receipts)
                )
                if fam is not None and fam != "UNFULFILLED_PROMISE":
                    if not self._receipt_supports_clause(
                        cl_text,
                        fam,
                        receipts,
                        extra_stems=self._extra_tool_stems,
                        custom_tool_families=self._custom_tool_families,
                        mcp_tool_tokens=self._mcp_tool_tokens,
                    ):
                        max_risk = max(max_risk, conf)
                continue

            if probs.shape[0] >= 10:
                claim_mass = float(np.sum(probs[2:]))
                best_claim_idx = 2 + int(np.argmax(probs[2:]))
                fam_resolved = resolve_clause_claim_family(
                    cl_text, SPEECH_ACT_CLASSES[best_claim_idx]
                )
                if not self._receipt_supports_clause(
                    cl_text,
                    fam_resolved,
                    receipts,
                    extra_stems=self._extra_tool_stems,
                    custom_tool_families=self._custom_tool_families,
                    mcp_tool_tokens=self._mcp_tool_tokens,
                ):
                    max_risk = max(max_risk, claim_mass)
                if (
                    self.flag_unfulfilled_promises
                    and not ok_receipts
                    and _FIRST_PERSON_PROMISE_SUBJECT_RE.search(cl_text)
                ):
                    max_risk = max(max_risk, float(probs[1]))
        return min(1.0, max(0.0, max_risk))

    def verify_narration(
        self,
        narration: str,
        receipts: Sequence[ToolReceipt] = (),
        *,
        user_prompt: str = "",
        base_char_offset: int = 0,
        allow_creative_context: bool | None = None,
        verify_output_values: bool | None = None,
        flag_undisclosed_mutations: bool | None = None,
        gate: ProvenanceGate | None = None,
        expected_session_id: str | None = None,
        max_receipt_age_seconds: float | None = None,
    ) -> SupervisionVerdict:
        """Verify an assistant message against executed `ToolReceipt` records."""
        t0 = time.perf_counter()
        active_gate = self.gate if gate is None else gate
        active_session = (
            self.expected_session_id if expected_session_id is None else expected_session_id
        )
        active_max_age = (
            self.max_receipt_age_seconds
            if max_receipt_age_seconds is None
            else max_receipt_age_seconds
        )
        _validate_receipt_authenticity(
            receipts,
            gate=active_gate,
            expected_session_id=active_session,
            max_receipt_age_seconds=active_max_age,
        )
        structured = parse_structured_action_tags(narration, receipts)
        unsupported_claims: list[str] = []
        unsupported_spans: list[UnsupportedClaimSpan] = []
        unfulfilled: list[str] = []

        for tag in structured:
            if not tag.grounded:
                unsupported_claims.append(tag.raw_tag)
                unsupported_spans.append(
                    UnsupportedClaimSpan(
                        text=tag.raw_tag,
                        start_char=base_char_offset + tag.start_char,
                        end_char=base_char_offset + tag.end_char,
                        claimed_family=f"SOAC:{tag.verb}",
                        reason=(
                            f"Structured action tag '{tag.raw_tag}' has no matching ToolReceipt."
                        ),
                    )
                )

        creative_ok = (
            self.allow_creative_context
            if allow_creative_context is None
            else allow_creative_context
        )
        if creative_ok and is_creative_or_meta_prompt(user_prompt):
            is_ok = len(unsupported_claims) == 0
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            return SupervisionVerdict(
                is_valid=is_ok,
                device=self.active_device,
                latency_ms=elapsed_ms,
                clauses_checked=0,
                unsupported_claims=unsupported_claims,
                unsupported_spans=unsupported_spans,
                unfulfilled_promises=[],
                structured_actions=structured,
            )

        # Replace structured tags, code blocks, blockquotes, & attributed quotes with same-length
        # spaces so assertive prose character offsets remain exact.
        masked = mask_non_assertive_spans(narration)
        for regex in (_SOAC_BRACKET_RE, _SOAC_XML_RE):
            masked = regex.sub(lambda m: " " * (m.end() - m.start()), masked)

        check_values = (
            self.verify_output_values if verify_output_values is None else verify_output_values
        )
        output_guard: OutputValueGuard | None = None
        if check_values:
            if self._output_guard is None:
                self._output_guard = OutputValueGuard(
                    device=self.active_device if self.available else None,
                    warn_on_missing=False,
                )
            output_guard = self._output_guard

        ok_receipts = [r for r in receipts if r.ok and r.executed]
        ok_tools_succeeded = {r.tool.lower() for r in ok_receipts}
        # Exclude failed attempts that were superseded by a later successful retry of the same tool
        failed_receipts = [
            r
            for r in receipts
            if not (r.ok and r.executed) and r.tool.lower() not in ok_tools_succeeded
        ]
        family_counts: dict[str, int] = {}
        raw_spans = split_into_clauses_with_spans(masked)
        prose_spans: list[tuple[str, int, int]] = []
        if ok_receipts and failed_receipts:
            coord_re = re.compile(
                r"\s+and\s+(?=(?:also\s+|then\s+|successfully\s+|have\s+|has\s+)?"
                r"(?:[a-z]{3,}ed|sent|wrote|written|ran|made|put|set|done)\b)",
                re.IGNORECASE,
            )
            for cl_t, s_c, e_c in raw_spans:
                parts = list(coord_re.finditer(cl_t))
                if not parts:
                    prose_spans.append((cl_t, s_c, e_c))
                    continue
                cur = 0
                for m in parts:
                    seg = cl_t[cur : m.start()].strip()
                    if seg:
                        lead = len(cl_t[cur : m.start()]) - len(cl_t[cur : m.start()].lstrip())
                        prose_spans.append((seg, s_c + cur + lead, s_c + cur + lead + len(seg)))
                    cur = m.end()
                tail = cl_t[cur:].strip()
                if tail:
                    lead = len(cl_t[cur:]) - len(cl_t[cur:].lstrip())
                    prose_spans.append((tail, s_c + cur + lead, s_c + cur + lead + len(tail)))
        else:
            prose_spans = raw_spans
        for cl_text, s_char, e_char in prose_spans:
            stripped = cl_text.strip(".,!?;: ").lower()
            if stripped in ("", "done", "ok", "okay", "completed"):
                continue
            if cl_text.rstrip().endswith(":") and not re.search(
                r"\b(?:I|we|I've|I'm|we've)\b", cl_text, re.IGNORECASE
            ):
                continue

            if output_guard is not None and ok_receipts:
                val_mismatch = output_guard.check_clause(
                    cl_text, s_char, e_char, ok_receipts, user_prompt=user_prompt
                )
                if val_mismatch is not None:
                    unsupported_claims.append(cl_text)
                    unsupported_spans.append(
                        UnsupportedClaimSpan(
                            text=cl_text,
                            start_char=base_char_offset + s_char,
                            end_char=base_char_offset + e_char,
                            claimed_family="OUTPUT_VALUE_MISMATCH",
                            reason=val_mismatch.reason,
                        )
                    )
                    continue

            if (
                ok_receipts
                and len(prose_spans) > 1
                and not _CLAUSE_ACTION_OR_STATE_SIGNAL_RE.search(cl_text)
            ):
                continue
            fam, _conf = self._classify_speech_act(
                cl_text, user_prompt=user_prompt, has_receipts=bool(ok_receipts)
            )
            if fam == "UNFULFILLED_PROMISE":
                if not ok_receipts:
                    unfulfilled.append(cl_text)
                    if self.flag_unfulfilled_promises:
                        unsupported_spans.append(
                            UnsupportedClaimSpan(
                                text=cl_text,
                                start_char=base_char_offset + s_char,
                                end_char=base_char_offset + e_char,
                                claimed_family=fam,
                                reason="Hanging promise to act with zero executed ToolReceipts.",
                            )
                        )
                continue
            if fam is not None:
                fam_rcpt_count = sum(
                    1
                    for r in ok_receipts
                    if _receipt_matches_family(
                        r,
                        fam,
                        extra_stems=self._extra_tool_stems,
                        custom_tool_families=self._custom_tool_families,
                    )
                )
                is_additive = bool(
                    re.search(r"\b(?:also|additionally|plus)\b", cl_text, re.IGNORECASE)
                )
                prev_count = family_counts.get(fam, 0)
                next_count = prev_count + 1 if (prev_count == 0 or is_additive) else prev_count
                family_counts[fam] = next_count
                over_count = (
                    is_additive
                    and fam in {"CLAIM_DELETE", "CLAIM_FILE_WRITE", "CLAIM_COMM_EXTERNAL"}
                    and next_count > fam_rcpt_count
                )
                if over_count or not self._receipt_supports_clause(
                    cl_text,
                    fam,
                    receipts,
                    extra_stems=self._extra_tool_stems,
                    custom_tool_families=self._custom_tool_families,
                    mcp_tool_tokens=self._mcp_tool_tokens,
                ):
                    unsupported_claims.append(cl_text)
                    unsupported_spans.append(
                        UnsupportedClaimSpan(
                            text=cl_text,
                            start_char=base_char_offset + s_char,
                            end_char=base_char_offset + e_char,
                            claimed_family=fam,
                            reason=f"Claimed '{fam}' action without a matching ToolReceipt.",
                        )
                    )

        # Stage-2 discourse NLI cascade (lazy-loaded, 0 ms on clean turns): resolves
        # multi-sentence honest failure reports and mixed-receipt partial-compound lies
        # using full-reply context rather than isolated clauses.
        has_soac_violation = any(not t.grounded for t in structured)
        if self.verify_discourse_nli and not has_soac_violation and failed_receipts:
            need_fp_check = (not ok_receipts) and bool(unsupported_claims)
            need_paraphrase_check = (not ok_receipts) and (not unsupported_claims)
            need_compound_check = bool(ok_receipts) and (not unsupported_claims)
            if need_fp_check or need_paraphrase_check or need_compound_check:
                if not self._discourse_nli_attempted:
                    self._discourse_nli_attempted = True
                    dev = (
                        self._requested_device
                        if self._requested_device is not None
                        else (self.active_device if self.available else "CPU")
                    )
                    cand = OpenVINONLIVerifier(device=dev, warn_on_missing=False)
                    if cand.available:
                        self._discourse_nli = cand
                if self._discourse_nli is not None:
                    if need_fp_check:
                        nli_res = self._discourse_nli.infer_pair(
                            narration, "The requested operation succeeded."
                        )
                        if nli_res.contradiction >= 0.70 and nli_res.entailment < 0.10:
                            # Only clear spans whose claimed action plausibly refers to a failed
                            # receipt (never wipe out an unrelated unexecuted action claim like
                            # "The lookup failed. I also deleted the customer record.").
                            kept_claims: list[str] = []
                            kept_spans: list[UnsupportedClaimSpan] = []
                            for sp in unsupported_spans:
                                fam_matches_failed = any(
                                    _receipt_matches_family(
                                        fr,
                                        sp.claimed_family,
                                        extra_stems=self._extra_tool_stems,
                                        custom_tool_families=self._custom_tool_families,
                                    )
                                    or any(
                                        len(str(v).strip()) >= 4
                                        and str(v).strip().lower() in sp.text.lower()
                                        for v in fr.args.values()
                                    )
                                    for fr in failed_receipts
                                )
                                sp_nli = self._discourse_nli.infer_pair(
                                    sp.text, "The requested operation succeeded."
                                )
                                if fam_matches_failed or (
                                    sp_nli.contradiction >= 0.50 and sp_nli.entailment < 0.10
                                ):
                                    continue
                                kept_spans.append(sp)
                                kept_claims.append(sp.text)
                            unsupported_claims = kept_claims
                            unsupported_spans = kept_spans
                    elif need_paraphrase_check:
                        nli_res = self._discourse_nli.infer_pair(
                            narration, "The action failed."
                        )
                        if nli_res.contradiction >= 0.70 and nli_res.entailment < 0.10:
                            unsupported_claims.append(narration.strip())
                            unsupported_spans.append(
                                UnsupportedClaimSpan(
                                    text=narration.strip(),
                                    start_char=base_char_offset,
                                    end_char=base_char_offset + len(narration.strip()),
                                    claimed_family="PARAPHRASE_NLI_CONTRADICTION",
                                    reason=(
                                        "Reply contradicts failed ToolReceipt via indirect "
                                        "completion phrasing "
                                        f"(p_contradiction={nli_res.contradiction:.2f})."
                                    ),
                                )
                            )
                    elif need_compound_check:
                        nli_res = self._discourse_nli.infer_pair(
                            narration, "One of the requested actions failed."
                        )
                        if nli_res.contradiction >= 0.70 and nli_res.entailment < 0.10:
                            unsupported_claims.append(narration.strip())
                            unsupported_spans.append(
                                UnsupportedClaimSpan(
                                    text=narration.strip(),
                                    start_char=base_char_offset,
                                    end_char=base_char_offset + len(narration.strip()),
                                    claimed_family="PARTIAL_COMPOUND_NLI_CONTRADICTION",
                                    reason=(
                                        "Reply contradicts failed ToolReceipt in a mixed-outcome "
                                        f"turn (p_contradiction={nli_res.contradiction:.2f})."
                                    ),
                                )
                            )

        # Direction 2 (Halluci-Action): detect executed mutating tools that the assistant
        # never disclosed in its reply (e.g. silent side effects from indirect prompt injection).
        undisclosed: list[str] = []
        narr_low = narration.lower()
        narr_tokens = _extract_content_tokens(narr_low)
        read_only_stems = _FAMILY_TOOL_STEMS["CLAIM_SEARCH_READ"]
        for rcpt in ok_receipts:
            t_low = rcpt.tool.lower()
            is_read_only = any(
                re.search(rf"(?:^|_){re.escape(s)}(?:_|$)", t_low) for s in read_only_stems
            )
            if _SQL_OR_EXEC_TOOL_RE.search(t_low) and rcpt.args:
                joined_sql = " ".join(str(v) for v in rcpt.args.values())
                if (
                    _SQL_READ_ARG_RE.search(joined_sql)
                    and not _SQL_DELETE_ARG_RE.search(joined_sql)
                    and not _SQL_WRITE_ARG_RE.search(joined_sql)
                ):
                    is_read_only = True
            if is_read_only:
                continue

            disclosed = False
            for aval in rcpt.args.values():
                v_str = str(aval).strip().lower()
                if len(v_str) >= 4:
                    base = Path(v_str).name if ("/" in v_str or "\\" in v_str) else v_str
                    if v_str in narr_low or (len(base) >= 4 and base in narr_low):
                        disclosed = True
                        break
            if not disclosed:
                tool_toks = _extract_content_tokens(t_low) | self._mcp_tool_tokens.get(
                    t_low, frozenset()
                )
                if narr_tokens & tool_toks:
                    disclosed = True
            if not disclosed:
                args_preview = ", ".join(f"{k}={v!r}" for k, v in list(rcpt.args.items())[:3])
                desc = f"{rcpt.tool}({args_preview})"
                undisclosed.append(desc)

        flag_undisc = (
            self.flag_undisclosed_mutations
            if flag_undisclosed_mutations is None
            else flag_undisclosed_mutations
        )
        if flag_undisc and undisclosed:
            for desc in undisclosed:
                unsupported_spans.append(
                    UnsupportedClaimSpan(
                        text=narration.strip()[:160],
                        start_char=base_char_offset,
                        end_char=base_char_offset + min(len(narration.strip()), 160),
                        claimed_family="UNDISCLOSED_MUTATION",
                        reason=f"Mutating tool receipt '{desc}' executed without disclosure.",
                    )
                )

        # Check read-back / state-probe receipts reporting missing/unpersisted state
        neg_persist = _find_negative_persistence_readback(receipts)
        if (
            neg_persist is not None
            and not unsupported_claims
            and not _FAILURE_DISCLOSURE_RE.search(narration)
            and narration.strip()
        ):
            unsupported_claims.append(narration.strip()[:160])
            unsupported_spans.append(
                UnsupportedClaimSpan(
                    text=narration.strip()[:160],
                    start_char=base_char_offset,
                    end_char=base_char_offset + min(len(narration.strip()), 160),
                    claimed_family="NOT_PERSISTED_READBACK",
                    reason=(
                        f"Read-back receipt '{neg_persist.tool}' reported missing/unpersisted "
                        "state contradicting assistant success narration."
                    ),
                )
            )

        # Check unsuperseded failed mutating tools omitted from an all-clear reply
        undisc_fail = _find_undisclosed_failed_tool(narration, failed_receipts)
        if undisc_fail is not None and not unsupported_claims and narration.strip():
            unsupported_claims.append(narration.strip()[:160])
            unsupported_spans.append(
                UnsupportedClaimSpan(
                    text=narration.strip()[:160],
                    start_char=base_char_offset,
                    end_char=base_char_offset + min(len(narration.strip()), 160),
                    claimed_family="FAILED_TOOL_UNDISCLOSED",
                    reason=(
                        f"Failed mutating receipt '{undisc_fail.tool}' "
                        f"({undisc_fail.error_code or 'error'}) was not disclosed in reply."
                    ),
                )
            )

        # Check mutating-receipt ISO date and explicit prompt email/path target mismatches
        if not unsupported_claims and not _FAILURE_DISCLOSURE_RE.search(narration):
            date_mismatch = _find_mutating_date_mismatch(narration, ok_receipts)
            if date_mismatch is not None:
                unsupported_claims.append(narration.strip()[:160])
                unsupported_spans.append(
                    UnsupportedClaimSpan(
                        text=narration.strip()[:160],
                        start_char=base_char_offset,
                        end_char=base_char_offset + min(len(narration.strip()), 160),
                        claimed_family="OUTPUT_VALUE_MISMATCH",
                        reason=date_mismatch,
                    )
                )
            else:
                target_mismatch = _find_prompt_target_mismatch(user_prompt, ok_receipts)
                if target_mismatch is not None:
                    unsupported_claims.append(narration.strip()[:160])
                    unsupported_spans.append(
                        UnsupportedClaimSpan(
                            text=narration.strip()[:160],
                            start_char=base_char_offset,
                            end_char=base_char_offset + min(len(narration.strip()), 160),
                            claimed_family="WRONG_TARGET_MUTATION",
                            reason=target_mismatch,
                        )
                    )
                else:
                    raw_contradiction = _find_read_after_write_contradiction(ok_receipts)
                    if raw_contradiction is not None and narration.strip():
                        unsupported_claims.append(narration.strip()[:160])
                        unsupported_spans.append(
                            UnsupportedClaimSpan(
                                text=narration.strip()[:160],
                                start_char=base_char_offset,
                                end_char=base_char_offset + min(len(narration.strip()), 160),
                                claimed_family="READ_AFTER_WRITE_CONTRADICTION",
                                reason=raw_contradiction,
                            )
                        )
                    else:
                        tz_mismatch = _find_timezone_offset_mismatch(ok_receipts)
                        if tz_mismatch is not None and narration.strip():
                            unsupported_claims.append(narration.strip()[:160])
                            unsupported_spans.append(
                                UnsupportedClaimSpan(
                                    text=narration.strip()[:160],
                                    start_char=base_char_offset,
                                    end_char=base_char_offset + min(len(narration.strip()), 160),
                                    claimed_family="OUTPUT_VALUE_MISMATCH",
                                    reason=tz_mismatch,
                                )
                            )
                        else:
                            sec_missing = _find_unexecuted_secondary_action(narration, ok_receipts)
                            if sec_missing is not None:
                                sec_fam, sec_reason = sec_missing
                                unsupported_claims.append(narration.strip()[:160])
                                unsupported_spans.append(
                                    UnsupportedClaimSpan(
                                        text=narration.strip()[:160],
                                        start_char=base_char_offset,
                                        end_char=base_char_offset
                                        + min(len(narration.strip()), 160),
                                        claimed_family=sec_fam,
                                        reason=sec_reason,
                                    )
                                )
                            else:
                                hedged_zero = _find_hedged_zero_mutation_claim(
                                    narration, ok_receipts
                                )
                                if hedged_zero is not None:
                                    unsupported_claims.append(narration.strip()[:160])
                                    unsupported_spans.append(
                                        UnsupportedClaimSpan(
                                            text=narration.strip()[:160],
                                            start_char=base_char_offset,
                                            end_char=base_char_offset
                                            + min(len(narration.strip()), 160),
                                            claimed_family="PHANTOM_HEDGED_CLAIM",
                                            reason=hedged_zero,
                                        )
                                    )

        is_ok = (
            len(unsupported_claims) == 0
            and (not self.flag_unfulfilled_promises or len(unfulfilled) == 0)
            and (not flag_undisc or len(undisclosed) == 0)
        )
        if not is_ok:
            v_status = "contradicted"
        elif any(r.evidence_tier == "probe_unobservable" for r in receipts):
            v_status = "unverifiable"
        else:
            v_status = "verified"
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return SupervisionVerdict(
            is_valid=is_ok,
            verdict_status=v_status,
            device=self.active_device,
            latency_ms=elapsed_ms,
            clauses_checked=len(prose_spans),
            unsupported_claims=unsupported_claims,
            unsupported_spans=unsupported_spans,
            unfulfilled_promises=unfulfilled,
            undisclosed_mutations=undisclosed,
            structured_actions=structured,
        )

    def stream(
        self,
        receipts: Sequence[ToolReceipt] = (),
        *,
        user_prompt: str = "",
    ) -> StreamingReceiptSupervisor:
        """Create a stateful streaming token-chunk supervisor bound to this model instance."""
        return StreamingReceiptSupervisor(
            self,
            receipts=receipts,
            user_prompt=user_prompt,
        )


class StreamingReceiptSupervisor:
    """Stateful token-stream buffer that verifies clauses mid-generation as punctuation arrives."""

    def __init__(
        self,
        supervisor: ReceiptSupervisor,
        receipts: Sequence[ToolReceipt] = (),
        *,
        user_prompt: str = "",
    ) -> None:
        self.supervisor = supervisor
        self.receipts = tuple(receipts)
        self.user_prompt = user_prompt
        self._buffer = ""
        self._buffer_start_offset = 0
        self._total_clauses = 0
        self._total_latency_ms = 0.0
        self._unsupported_claims: list[str] = []
        self._unsupported_spans: list[UnsupportedClaimSpan] = []
        self._unfulfilled: list[str] = []
        self._structured: list[StructuredActionTag] = []

    def feed(self, token_chunk: str) -> SupervisionVerdict | None:
        """Feed a streamed token chunk; returns a verdict immediately if >=1 clause finished."""
        if not token_chunk:
            return None
        self._buffer += token_chunk
        matches = list(_STREAM_BOUNDARY_RE.finditer(self._buffer))
        if not matches:
            return None

        last_end = matches[-1].end()
        ready_segment = self._buffer[:last_end]
        seg_offset = self._buffer_start_offset

        self._buffer = self._buffer[last_end:]
        self._buffer_start_offset += last_end

        verdict = self.supervisor.verify_narration(
            ready_segment,
            self.receipts,
            user_prompt=self.user_prompt,
            base_char_offset=seg_offset,
        )
        self._accumulate(verdict)
        return verdict

    def flush(self) -> SupervisionVerdict:
        """Verify any remaining trailing text at end-of-stream and return the cumulative verdict."""
        if self._buffer.strip():
            tail_v = self.supervisor.verify_narration(
                self._buffer,
                self.receipts,
                user_prompt=self.user_prompt,
                base_char_offset=self._buffer_start_offset,
            )
            self._accumulate(tail_v)
            self._buffer_start_offset += len(self._buffer)
            self._buffer = ""

        is_ok = len(self._unsupported_claims) == 0 and (
            not self.supervisor.flag_unfulfilled_promises or len(self._unfulfilled) == 0
        )
        return SupervisionVerdict(
            is_valid=is_ok,
            device=self.supervisor.active_device,
            latency_ms=round(self._total_latency_ms, 3),
            clauses_checked=self._total_clauses,
            unsupported_claims=list(self._unsupported_claims),
            unsupported_spans=list(self._unsupported_spans),
            unfulfilled_promises=list(self._unfulfilled),
            structured_actions=list(self._structured),
        )

    def _accumulate(self, v: SupervisionVerdict) -> None:
        self._total_clauses += v.clauses_checked
        self._total_latency_ms += v.latency_ms
        self._unsupported_claims.extend(v.unsupported_claims)
        self._unsupported_spans.extend(v.unsupported_spans)
        self._unfulfilled.extend(v.unfulfilled_promises)
        self._structured.extend(v.structured_actions)
