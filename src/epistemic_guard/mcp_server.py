"""Standalone stdio MCP JSON-RPC 2.0 server exposing `epistemic-guard` verification tools."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping, Sequence
from typing import Any, BinaryIO, Final

from epistemic_guard.corroboration import corroborate_sources
from epistemic_guard.exceptions import EpistemicGuardError, ProvenanceError
from epistemic_guard.mcp_proxy import read_public_key_file, verify_receipt_log_file
from epistemic_guard.provenance import ProvenanceGate
from epistemic_guard.schemas import Provenance, ToolReceipt
from epistemic_guard.store import MerkleReceiptLog
from epistemic_guard.supervision import ReceiptSupervisor

_DEFAULT_PROTOCOL_VERSION: Final[str] = "2024-11-05"
_SUPPORTED_PROTOCOLS: Final[frozenset[str]] = frozenset(
    {"2024-11-05", "2025-03-26", "2025-06-18"}
)
_SERVER_VERSION: Final[str] = "0.5.2"

MCP_TOOLS: Final[tuple[dict[str, object], ...]] = (
    {
        "name": "verify_narration",
        "description": (
            "Audit an assistant draft reply clause-by-clause against in-memory turn receipts "
            "to flag fabricated tool-action claims, unlike verify_receipt_log and inspect_receipt "
            "which read on-disk JSONL logs. Call before sending a tool-using reply (pass "
            "receipts=[] when zero tools ran, and pass expected_pubkey_hex or expected_session_id "
            "to reject unsigned or replayed receipts); do not use for RAG citation checking (use "
            "corroborate_sources) or log-file cryptography. Runs offline with zero network calls: "
            "ungrounded claims return is_valid=false with character offsets and retry_prompt "
            "rather than raising an error, whereas malformed, forged, or expired receipts return "
            "isError=true."
        ),
        "annotations": {
            "title": "Verify Assistant Narration Against Tool Receipts",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "inputSchema": {
            "type": "object",
            "properties": {
                "narration": {
                    "type": "string",
                    "description": "Assistant reply text to audit for ungrounded tool claims.",
                },
                "receipts": {
                    "type": "array",
                    "description": (
                        "Tool execution receipts for the current turn (ToolReceipt objects or "
                        "AER-1 dicts; pass [] if zero tools ran)."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "call_id": {
                                "type": "string",
                                "description": "Unique tool call identifier.",
                            },
                            "tool": {
                                "type": "string",
                                "description": "Canonical name of the invoked tool.",
                            },
                            "ok": {
                                "type": "boolean",
                                "description": "True if the tool succeeded without error.",
                            },
                            "executed": {
                                "type": "boolean",
                                "description": (
                                    "False if rejected before execution (e.g. schema error)."
                                ),
                            },
                            "error_code": {
                                "type": "string",
                                "description": "Optional machine-readable error code on failure.",
                            },
                            "args": {
                                "type": "object",
                                "description": "Scalar key-value arguments passed to the tool.",
                            },
                            "output": {
                                "type": "string",
                                "description": "Short summary or return payload from the tool.",
                            },
                            "output_sha256": {
                                "type": "string",
                                "description": (
                                    "Optional 64-char hex SHA-256 digest of output (auto-computed "
                                    "if omitted)."
                                ),
                            },
                            "timestamp_utc": {
                                "type": "string",
                                "description": (
                                    "Optional ISO-8601 UTC mint timestamp checked by "
                                    "max_receipt_age_seconds."
                                ),
                            },
                            "session_id": {
                                "type": "string",
                                "description": (
                                    "Optional proxy session identifier checked by "
                                    "expected_session_id."
                                ),
                            },
                            "signature": {
                                "type": "string",
                                "description": (
                                    "Optional 128-char hex Ed25519 signature checked when "
                                    "expected_pubkey_hex or pubkey_file is supplied."
                                ),
                            },
                        },
                        "required": ["tool"],
                        "additionalProperties": True,
                    },
                },
                "user_prompt": {
                    "type": "string",
                    "description": "Optional user prompt for conversational context.",
                },
                "expected_pubkey_hex": {
                    "type": "string",
                    "description": (
                        "Optional pinned 64-char hex Ed25519 public key; when set, every receipt "
                        "must carry a valid Ed25519 signature."
                    ),
                },
                "pubkey_file": {
                    "type": "string",
                    "description": "Optional path to a pinned Ed25519 public key JSON or hex file.",
                },
                "expected_session_id": {
                    "type": "string",
                    "description": (
                        "Optional proxy session identifier; rejects receipts from other sessions."
                    ),
                },
                "max_receipt_age_seconds": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": (
                        "Optional freshness window in seconds; rejects stale or replayed receipts."
                    ),
                },
            },
            "required": ["narration", "receipts"],
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "is_valid": {
                    "type": "boolean",
                    "description": "True when every action claim is backed by a receipt.",
                },
                "signature_verified": {
                    "type": "boolean",
                    "description": "True when receipts were verified against a pinned Ed25519 key.",
                },
                "device": {
                    "type": "string",
                    "description": "Inference device used for supervision (e.g. CPU).",
                },
                "latency_ms": {
                    "type": "number",
                    "description": "Wall-clock supervision latency in milliseconds.",
                },
                "unsupported_claims": {
                    "type": "array",
                    "description": "Verbatim assistant clauses lacking receipt support.",
                    "items": {"type": "string"},
                },
                "unsupported_spans": {
                    "type": "array",
                    "description": "Character-level spans for each ungrounded action claim.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Matched ungrounded clause text.",
                            },
                            "start_char": {
                                "type": "integer",
                                "description": "Zero-based start character offset in narration.",
                            },
                            "end_char": {
                                "type": "integer",
                                "description": "Zero-based exclusive end character offset.",
                            },
                            "claimed_family": {
                                "type": "string",
                                "description": "Inferred action category of the claim.",
                            },
                        },
                        "required": ["text", "start_char", "end_char", "claimed_family"],
                    },
                },
                "retry_prompt": {
                    "type": "string",
                    "description": "Corrective repair instruction when is_valid is false, else ''.",
                },
            },
            "required": [
                "is_valid",
                "signature_verified",
                "device",
                "latency_ms",
                "unsupported_claims",
                "unsupported_spans",
                "retry_prompt",
            ],
            "additionalProperties": False,
        },
    },
    {
        "name": "verify_receipt_log",
        "description": (
            "Validate whole-file cryptographic integrity of an on-disk JSONL receipt log written "
            "by epistemic-guard mcp-proxy, returning aggregate chain, session, and Merkle root "
            "status rather than individual entries (use inspect_receipt to fetch entries). Call "
            "in CI gates or post-session audits; do not use to check assistant prose (use "
            "verify_narration) or mint receipts (signing is confined to the isolated mcp-proxy "
            "parent process). Passing both pubkey_file and expected_pubkey_hex requires identical "
            "32-byte keys (prev_log_path verifies key-rotation chain continuity), and any missing "
            "file, key/session mismatch, sequence gap, or tampered digest returns isError=true."
        ),
        "annotations": {
            "title": "Verify Signed MCP Merkle Receipt Log",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "inputSchema": {
            "type": "object",
            "properties": {
                "log_path": {
                    "type": "string",
                    "description": "Filesystem path to the JSONL receipt log file.",
                },
                "expected_pubkey_hex": {
                    "type": "string",
                    "description": "Optional pinned 64-character hex Ed25519 public key.",
                },
                "pubkey_file": {
                    "type": "string",
                    "description": "Optional path to a pinned public key JSON or hex file.",
                },
                "expected_session_id": {
                    "type": "string",
                    "description": "Optional expected session_id bound into each signed receipt.",
                },
                "prev_log_path": {
                    "type": "string",
                    "description": (
                        "Optional predecessor JSONL log path to verify key-rotation hash-chain "
                        "continuity."
                    ),
                },
            },
            "required": ["log_path"],
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "log_valid": {
                    "type": "boolean",
                    "description": "True when all signatures, chain links, and Merkle roots hold.",
                },
                "entry_count": {
                    "type": "integer",
                    "description": "Total verified receipt records in the JSONL file.",
                },
                "merkle_root": {
                    "type": "string",
                    "description": "Hex-encoded 64-char RFC 6962 SHA-256 Merkle root.",
                },
                "head_chain_hash": {
                    "type": "string",
                    "description": "Hex-encoded cumulative sequential SHA-256 chain tip.",
                },
                "prev_chain_sha256": {
                    "type": "string",
                    "description": (
                        "Initial 64-char chain anchor (0*64 or rotated predecessor tip)."
                    ),
                },
                "session_id": {
                    "type": "string",
                    "description": "Session identifier bound inside the verified receipts.",
                },
                "public_key_hex": {
                    "type": "string",
                    "description": (
                        "Hex-encoded 32-byte Ed25519 public key verified across entries."
                    ),
                },
                "key_pinned": {
                    "type": "boolean",
                    "description": "True when verified against an explicit caller-pinned key.",
                },
            },
            "required": [
                "log_valid",
                "entry_count",
                "merkle_root",
                "head_chain_hash",
                "prev_chain_sha256",
                "session_id",
                "public_key_hex",
                "key_pinned",
            ],
            "additionalProperties": False,
        },
    },
    {
        "name": "inspect_receipt",
        "description": (
            "Retrieve individual signed receipt records, standalone RFC 6962 Merkle inclusion "
            "proofs, and optional AER-1 envelopes from an on-disk JSONL receipt log, whereas "
            "verify_receipt_log returns only a whole-log summary. Call when auditing specific "
            "tool invocations, slicing current-turn receipts via since_seq, or exporting portable "
            "proofs; do not use for a fast whole-file summary (use verify_receipt_log) or prose "
            "checking (use verify_narration). Cryptographically verifies the entire log before "
            "returning data: filters seq, since_seq, call_id, and tool combine via logical AND, "
            "while a missing/tampered file, key/session mismatch, or out-of-range seq returns "
            "isError=true."
        ),
        "annotations": {
            "title": "Inspect Signed Receipt and Merkle Inclusion Proof",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "inputSchema": {
            "type": "object",
            "properties": {
                "log_path": {
                    "type": "string",
                    "description": "Filesystem path to the JSONL receipt log file.",
                },
                "seq": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Optional exact zero-based log sequence index to retrieve.",
                },
                "since_seq": {
                    "type": "integer",
                    "minimum": 0,
                    "description": (
                        "Optional minimum zero-based sequence index (inclusive) for turn slicing."
                    ),
                },
                "call_id": {
                    "type": "string",
                    "description": "Optional exact tool call identifier to filter by.",
                },
                "tool": {
                    "type": "string",
                    "description": "Optional canonical tool name to filter receipts by.",
                },
                "expected_pubkey_hex": {
                    "type": "string",
                    "description": "Optional pinned 64-character hex Ed25519 public key.",
                },
                "pubkey_file": {
                    "type": "string",
                    "description": "Optional path to a pinned public key JSON or hex file.",
                },
                "expected_session_id": {
                    "type": "string",
                    "description": "Optional expected session_id bound inside each signed receipt.",
                },
                "prev_log_path": {
                    "type": "string",
                    "description": (
                        "Optional predecessor JSONL log path to verify key-rotation continuity."
                    ),
                },
                "include_aer1": {
                    "type": "boolean",
                    "description": (
                        "When true, attach the canonical AER-1 interoperable receipt envelope "
                        "to each matched entry (default: false)."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Maximum number of matched entries to return (default: 20).",
                },
            },
            "required": ["log_path"],
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "log_valid": {
                    "type": "boolean",
                    "description": "True after full-file signature and Merkle verification.",
                },
                "entry_count": {
                    "type": "integer",
                    "description": "Total entries present in the verified JSONL log.",
                },
                "merkle_root": {
                    "type": "string",
                    "description": "Hex-encoded 64-char RFC 6962 SHA-256 Merkle root of the log.",
                },
                "public_key_hex": {
                    "type": "string",
                    "description": "Hex-encoded Ed25519 public key that signed the log.",
                },
                "key_pinned": {
                    "type": "boolean",
                    "description": "True when verified against expected_pubkey_hex or pubkey_file.",
                },
                "match_count": {
                    "type": "integer",
                    "description": "Number of filtered receipt entries returned in entries.",
                },
                "entries": {
                    "type": "array",
                    "description": "Matched receipt records with standalone Merkle proofs.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "seq": {
                                "type": "integer",
                                "description": "Zero-based sequence index in the JSONL log.",
                            },
                            "receipt": {
                                "type": "object",
                                "description": "Signed ToolReceipt fields (tool, args, ok, etc.).",
                            },
                            "merkle_inclusion_proof": {
                                "type": "object",
                                "description": "Standalone O(log N) RFC 6962 inclusion proof.",
                                "properties": {
                                    "leaf_index": {
                                        "type": "integer",
                                        "description": "Zero-based leaf position in Merkle tree.",
                                    },
                                    "tree_size": {
                                        "type": "integer",
                                        "description": "Total leaf count in the Merkle tree.",
                                    },
                                    "leaf_sha256": {
                                        "type": "string",
                                        "description": "Domain-separated 0x00 leaf SHA-256 hex.",
                                    },
                                    "siblings": {
                                        "type": "array",
                                        "description": "Ordered sibling hashes from leaf to root.",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "side": {
                                                    "type": "string",
                                                    "enum": ["L", "R"],
                                                    "description": "Sibling position (L or R).",
                                                },
                                                "sha256": {
                                                    "type": "string",
                                                    "description": "64-char hex sibling digest.",
                                                },
                                            },
                                            "required": ["side", "sha256"],
                                        },
                                    },
                                    "root_sha256": {
                                        "type": "string",
                                        "description": "Expected 64-char hex RFC 6962 Merkle root.",
                                    },
                                },
                                "required": [
                                    "leaf_index",
                                    "tree_size",
                                    "leaf_sha256",
                                    "siblings",
                                    "root_sha256",
                                ],
                            },
                            "aer1": {
                                "type": "object",
                                "description": "Canonical AER-1 envelope when include_aer1=true.",
                            },
                        },
                        "required": ["seq", "receipt", "merkle_inclusion_proof"],
                    },
                },
            },
            "required": [
                "log_valid",
                "entry_count",
                "merkle_root",
                "public_key_hex",
                "key_pinned",
                "match_count",
                "entries",
            ],
            "additionalProperties": False,
        },
    },
    {
        "name": "corroborate_sources",
        "description": (
            "Evaluate independent evidentiary support across retrieved RAG or web snippets by "
            "collapsing syndicated wire-copy duplicates and checking cross-domain contradictions. "
            "Call before answering factual queries from retrieved passages; do not use to fetch "
            "web URLs (URLs are parsed offline solely for domain grouping and never contacted "
            "over the network) or to audit tool-execution prose (use verify_narration). Supply "
            "at least two sources from distinct domains to reach multi-source proof standards "
            "(sources=[] yields speculation with meets_casual_burden=false); contradictions set "
            "disputed=true without failing, whereas an empty query or malformed sources item "
            "returns isError=true."
        ),
        "annotations": {
            "title": "Corroborate and Deduplicate RAG Sources",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "User question or claim being answered from retrieved sources.",
                },
                "sources": {
                    "type": "array",
                    "description": "Retrieved web or RAG snippets with source URLs.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "Source URL or hostname parsed offline for domain.",
                            },
                            "snippet": {
                                "type": "string",
                                "description": "Retrieved text passage from this source.",
                            },
                        },
                        "required": ["url", "snippet"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["query", "sources"],
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "hazard": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Classified risk tier of query scaling required corroboration.",
                },
                "achieved_standard": {
                    "type": "string",
                    "enum": [
                        "speculation",
                        "balance of probabilities",
                        "clear and convincing",
                        "beyond reasonable doubt",
                    ],
                    "description": "Highest evidentiary standard met after deduplication.",
                },
                "raw_sources_count": {
                    "type": "integer",
                    "description": "Total input snippet items supplied in sources.",
                },
                "independent_sources_count": {
                    "type": "integer",
                    "description": "Remaining independent source clusters after deduplication.",
                },
                "deduplicated_duplicates_collapsed": {
                    "type": "integer",
                    "description": "Count of mirrored or syndicated passages collapsed.",
                },
                "disputed": {
                    "type": "boolean",
                    "description": "True if cross-domain NLI detected conflicting claims.",
                },
                "dispute_sources": {
                    "type": "array",
                    "description": "Domain or URL pairs exhibiting NLI contradiction.",
                    "items": {"type": "string"},
                },
                "required_casual_standard": {
                    "type": "string",
                    "enum": [
                        "speculation",
                        "balance of probabilities",
                        "clear and convincing",
                        "beyond reasonable doubt",
                    ],
                    "description": "Minimum proof standard required for hazard tier.",
                },
                "meets_casual_burden": {
                    "type": "boolean",
                    "description": "True if achieved_standard satisfies required_casual_standard.",
                },
            },
            "required": [
                "hazard",
                "achieved_standard",
                "raw_sources_count",
                "independent_sources_count",
                "deduplicated_duplicates_collapsed",
                "disputed",
                "dispute_sources",
                "required_casual_standard",
                "meets_casual_burden",
            ],
            "additionalProperties": False,
        },
    },
)


def _coerce_receipt_args(raw_args: object) -> dict[str, str | int | float | bool]:
    if not isinstance(raw_args, Mapping):
        return {}
    clean: dict[str, str | int | float | bool] = {}
    for k, v in raw_args.items():
        key = str(k)
        if isinstance(v, bool) or isinstance(v, (str, int)):
            clean[key] = v
        elif isinstance(v, float):
            clean[key] = v if math.isfinite(v) else str(v)
        else:
            try:
                clean[key] = json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False)
            except (TypeError, ValueError):
                clean[key] = str(v)
    return clean


def _parse_receipts(raw_items: object) -> list[ToolReceipt]:
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        raise ValueError("'receipts' must be an array of receipt objects.")
    parsed: list[ToolReceipt] = []
    for idx, item in enumerate(raw_items):
        if not isinstance(item, Mapping):
            raise ValueError(f"receipts[{idx}] must be an object.")
        if (
            "receipt_schema_version" in item
            or "canonical_bytes" in item
            or item.get("aer1_version")
            or item.get("spec_version")
        ):
            parsed.append(ToolReceipt.from_aer1(dict(item)))
            continue
        tool_name = item.get("tool")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError(f"receipts[{idx}].tool must be a non-empty string.")
        call_id_raw = item.get("call_id")
        call_id = call_id_raw if isinstance(call_id_raw, str) and call_id_raw else f"call-{idx}"
        ok_raw = item.get("ok", True)
        exec_raw = item.get("executed", True)
        err_raw = item.get("error_code", "")
        out_raw = item.get("output", "")
        out_sha_raw = item.get("output_sha256", "")
        sig_raw = item.get("signature", "")
        sess_raw = item.get("session_id", "")
        ts_raw = item.get("timestamp_utc")
        rcpt_kwargs: dict[str, Any] = {
            "call_id": call_id,
            "tool": tool_name,
            "ok": bool(ok_raw),
            "executed": bool(exec_raw),
            "error_code": str(err_raw) if err_raw is not None else "",
            "args": _coerce_receipt_args(item.get("args")),
            "output": str(out_raw) if out_raw is not None else "",
            "output_sha256": str(out_sha_raw) if out_sha_raw is not None else "",
            "session_id": str(sess_raw) if sess_raw is not None else "",
            "signature": str(sig_raw) if sig_raw is not None else "",
        }
        if isinstance(ts_raw, str) and ts_raw.strip():
            rcpt_kwargs["timestamp_utc"] = ts_raw
        parsed.append(ToolReceipt(**rcpt_kwargs))
    return parsed


def _resolve_pinned_gate(args: Mapping[str, object]) -> ProvenanceGate | None:
    pubkey_raw = args.get("expected_pubkey_hex")
    pubkey_hex = pubkey_raw.strip().lower() if isinstance(pubkey_raw, str) and pubkey_raw else None
    pubkey_file_raw = args.get("pubkey_file")
    file_hex = (
        read_public_key_file(pubkey_file_raw)
        if isinstance(pubkey_file_raw, str) and pubkey_file_raw.strip()
        else None
    )
    if pubkey_hex is not None:
        raw_bytes = bytes.fromhex(pubkey_hex)
        if len(raw_bytes) != 32:
            raise ProvenanceError("expected_pubkey_hex must decode to 32 bytes.")
        if file_hex is not None and file_hex != pubkey_hex:
            raise ProvenanceError("Conflicting expected_pubkey_hex and pubkey_file values.")
        return ProvenanceGate({Provenance.TOOL_VERIFIED: raw_bytes})
    if file_hex is not None:
        return ProvenanceGate({Provenance.TOOL_VERIFIED: bytes.fromhex(file_hex)})
    return None


def _tool_result(payload: Mapping[str, object], *, is_error: bool = False) -> dict[str, object]:
    body = dict(payload)
    res: dict[str, object] = {
        "content": [
            {
                "type": "text",
                "text": json.dumps(body, sort_keys=True, allow_nan=False),
            }
        ],
        "isError": is_error,
    }
    if not is_error:
        res["structuredContent"] = body
    return res


class EpistemicGuardMcpServer:
    """Stateless JSON-RPC 2.0 MCP server handler for `epistemic-guard`."""

    def __init__(self, *, device: str = "CPU") -> None:
        self._device = device
        self._supervisor: ReceiptSupervisor | None = None

    def _get_supervisor(self) -> ReceiptSupervisor:
        if self._supervisor is None:
            self._supervisor = ReceiptSupervisor(device=self._device, warn_on_missing=False)
        return self._supervisor

    def handle_message(self, msg: object) -> dict[str, object] | list[dict[str, object]] | None:
        """Process a decoded JSON-RPC message or batch and return the response object."""
        if isinstance(msg, list):
            if not msg:
                return {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "Invalid empty JSON-RPC batch."},
                }
            responses: list[dict[str, object]] = []
            for entry in msg:
                res = self._handle_single(entry)
                if res is not None:
                    responses.append(res)
            return responses or None
        return self._handle_single(msg)

    def _handle_single(self, msg: object) -> dict[str, object] | None:
        if not isinstance(msg, Mapping):
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Invalid JSON-RPC request object."},
            }
        has_id = "id" in msg
        req_id = msg.get("id")
        method = msg.get("method")
        if not isinstance(method, str):
            if not has_id:
                return None
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32600, "message": "Missing string 'method' field."},
            }

        if not has_id:
            return None

        params_raw = msg.get("params")
        params: Mapping[str, object] = params_raw if isinstance(params_raw, Mapping) else {}

        if method == "initialize":
            requested_ver = params.get("protocolVersion")
            proto = (
                requested_ver
                if isinstance(requested_ver, str) and requested_ver in _SUPPORTED_PROTOCOLS
                else _DEFAULT_PROTOCOL_VERSION
            )
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": proto,
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"listChanged": False},
                        "prompts": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": "epistemic-guard",
                        "version": _SERVER_VERSION,
                    },
                },
            }

        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": list(MCP_TOOLS)}}

        if method == "resources/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"resources": []}}

        if method == "resources/templates/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"resourceTemplates": []}}

        if method == "prompts/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"prompts": []}}

        if method == "tools/call":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": self._dispatch_tool(params),
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    def _dispatch_tool(self, params: Mapping[str, object]) -> dict[str, object]:
        name = params.get("name")
        raw_args = params.get("arguments")
        args: Mapping[str, object] = raw_args if isinstance(raw_args, Mapping) else {}

        try:
            if name == "verify_narration":
                return self._call_verify_narration(args)
            if name == "verify_receipt_log":
                return self._call_verify_receipt_log(args)
            if name == "inspect_receipt":
                return self._call_inspect_receipt(args)
            if name == "corroborate_sources":
                return self._call_corroborate_sources(args)
        except (EpistemicGuardError, OSError, ValueError) as exc:
            return _tool_result({"error": str(exc)}, is_error=True)

        return _tool_result({"error": f"Unknown tool: {name!r}"}, is_error=True)

    def _call_verify_narration(self, args: Mapping[str, object]) -> dict[str, object]:
        narration = args.get("narration")
        if not isinstance(narration, str):
            raise ValueError("Missing required string argument 'narration'.")
        receipts = _parse_receipts(args.get("receipts", []))
        prompt_raw = args.get("user_prompt", "")
        user_prompt = prompt_raw if isinstance(prompt_raw, str) else ""
        pinned_gate = _resolve_pinned_gate(args)
        sess_raw = args.get("expected_session_id")
        expected_sess = sess_raw if isinstance(sess_raw, str) and sess_raw.strip() else None
        age_raw = args.get("max_receipt_age_seconds")
        max_age: float | None = None
        if age_raw is not None:
            if isinstance(age_raw, bool) or not isinstance(age_raw, (int, float)) or age_raw <= 0:
                raise ValueError("'max_receipt_age_seconds' must be a positive number.")
            max_age = float(age_raw)

        verdict = self._get_supervisor().verify_narration(
            narration,
            receipts=receipts,
            user_prompt=user_prompt,
            gate=pinned_gate,
            expected_session_id=expected_sess,
            max_receipt_age_seconds=max_age,
        )
        return _tool_result(
            {
                "is_valid": verdict.is_valid,
                "signature_verified": pinned_gate is not None,
                "device": verdict.device,
                "latency_ms": round(verdict.latency_ms, 3),
                "unsupported_claims": list(verdict.unsupported_claims),
                "unsupported_spans": [
                    {
                        "text": sp.text,
                        "start_char": sp.start_char,
                        "end_char": sp.end_char,
                        "claimed_family": sp.claimed_family,
                    }
                    for sp in verdict.unsupported_spans
                ],
                "retry_prompt": "" if verdict.is_valid else verdict.to_retry_prompt(receipts),
            }
        )

    def _call_verify_receipt_log(self, args: Mapping[str, object]) -> dict[str, object]:
        log_path = args.get("log_path")
        if not isinstance(log_path, str) or not log_path.strip():
            raise ValueError("Missing required string argument 'log_path'.")
        pubkey_raw = args.get("expected_pubkey_hex")
        pubkey_hex = pubkey_raw if isinstance(pubkey_raw, str) and pubkey_raw else None
        pubkey_file_raw = args.get("pubkey_file")
        pubkey_file = (
            pubkey_file_raw if isinstance(pubkey_file_raw, str) and pubkey_file_raw else None
        )
        sess_raw = args.get("expected_session_id")
        expected_sess = sess_raw if isinstance(sess_raw, str) and sess_raw.strip() else None
        prev_raw = args.get("prev_log_path")
        prev_log = prev_raw if isinstance(prev_raw, str) and prev_raw.strip() else None

        verified = verify_receipt_log_file(
            log_path,
            expected_pubkey_hex=pubkey_hex,
            pubkey_file=pubkey_file,
            expected_session_id=expected_sess,
            prev_log_path=prev_log,
        )
        return _tool_result(
            {
                "log_valid": True,
                "entry_count": verified.entry_count,
                "merkle_root": verified.merkle_root,
                "head_chain_hash": verified.head_chain_hash,
                "prev_chain_sha256": verified.prev_chain_sha256,
                "session_id": verified.session_id,
                "public_key_hex": verified.public_key_hex,
                "key_pinned": verified.key_pinned,
            }
        )

    def _call_inspect_receipt(self, args: Mapping[str, object]) -> dict[str, object]:
        log_path = args.get("log_path")
        if not isinstance(log_path, str) or not log_path.strip():
            raise ValueError("Missing required string argument 'log_path'.")

        seq_raw = args.get("seq")
        seq_filter: int | None = None
        if seq_raw is not None:
            if isinstance(seq_raw, bool) or not isinstance(seq_raw, int) or seq_raw < 0:
                raise ValueError("'seq' must be a non-negative integer.")
            seq_filter = seq_raw

        since_raw = args.get("since_seq")
        since_filter = 0
        if since_raw is not None:
            if isinstance(since_raw, bool) or not isinstance(since_raw, int) or since_raw < 0:
                raise ValueError("'since_seq' must be a non-negative integer.")
            since_filter = since_raw

        limit_raw = args.get("limit", 20)
        if (
            isinstance(limit_raw, bool)
            or not isinstance(limit_raw, int)
            or not (1 <= limit_raw <= 100)
        ):
            raise ValueError("'limit' must be an integer between 1 and 100.")
        limit = limit_raw

        call_id_raw = args.get("call_id")
        call_id_filter = (
            call_id_raw if isinstance(call_id_raw, str) and call_id_raw.strip() else None
        )
        tool_raw = args.get("tool")
        tool_filter = tool_raw if isinstance(tool_raw, str) and tool_raw.strip() else None
        include_aer1 = bool(args.get("include_aer1", False))

        pubkey_raw = args.get("expected_pubkey_hex")
        pubkey_hex = pubkey_raw if isinstance(pubkey_raw, str) and pubkey_raw else None
        pubkey_file_raw = args.get("pubkey_file")
        pubkey_file = (
            pubkey_file_raw if isinstance(pubkey_file_raw, str) and pubkey_file_raw else None
        )
        sess_raw = args.get("expected_session_id")
        expected_sess = sess_raw if isinstance(sess_raw, str) and sess_raw.strip() else None
        prev_raw = args.get("prev_log_path")
        prev_log = prev_raw if isinstance(prev_raw, str) and prev_raw.strip() else None

        verified = verify_receipt_log_file(
            log_path,
            expected_pubkey_hex=pubkey_hex,
            pubkey_file=pubkey_file,
            expected_session_id=expected_sess,
            prev_log_path=prev_log,
        )
        if seq_filter is not None and seq_filter >= verified.entry_count:
            raise ValueError(
                f"Receipt seq={seq_filter} out of range for log with "
                f"{verified.entry_count} entries."
            )

        merkle = MerkleReceiptLog(initial_chain_hash=verified.prev_chain_sha256)
        for rcpt in verified.receipts:
            merkle.append(rcpt)

        candidate_indices = (
            [seq_filter]
            if seq_filter is not None
            else list(range(since_filter, verified.entry_count))
        )
        entries_out: list[dict[str, object]] = []
        for idx in candidate_indices:
            if idx < since_filter:
                continue
            rcpt = verified.receipts[idx]
            if call_id_filter is not None and rcpt.call_id != call_id_filter:
                continue
            if tool_filter is not None and rcpt.tool != tool_filter:
                continue
            proof = merkle.inclusion_proof(idx)
            item: dict[str, object] = {
                "seq": idx,
                "receipt": rcpt.model_dump(),
                "merkle_inclusion_proof": {
                    "leaf_index": proof.leaf_index,
                    "tree_size": proof.tree_size,
                    "leaf_sha256": proof.leaf_sha256,
                    "siblings": [
                        {"side": side, "sha256": digest} for side, digest in proof.siblings
                    ],
                    "root_sha256": proof.root_sha256,
                },
            }
            if include_aer1:
                item["aer1"] = rcpt.to_aer1()
            entries_out.append(item)
            if len(entries_out) >= limit:
                break

        return _tool_result(
            {
                "log_valid": True,
                "entry_count": verified.entry_count,
                "merkle_root": verified.merkle_root,
                "public_key_hex": verified.public_key_hex,
                "key_pinned": verified.key_pinned,
                "match_count": len(entries_out),
                "entries": entries_out,
            }
        )

    def _call_corroborate_sources(self, args: Mapping[str, object]) -> dict[str, object]:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Missing required non-empty string argument 'query'.")
        raw_sources = args.get("sources")
        if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
            raise ValueError("Missing required array argument 'sources'.")
        clean_sources: list[dict[str, str]] = []
        for idx, item in enumerate(raw_sources):
            if not isinstance(item, Mapping):
                raise ValueError(f"sources[{idx}] must be an object with 'url' and 'snippet'.")
            url = item.get("url")
            snippet = item.get("snippet")
            if not isinstance(url, str) or not isinstance(snippet, str):
                raise ValueError(f"sources[{idx}] requires string 'url' and 'snippet' fields.")
            clean_sources.append({"url": url, "snippet": snippet})

        burden = corroborate_sources(
            query,
            clean_sources,
            device=self._device,
            warn_on_missing=False,
        )
        return _tool_result(
            {
                "hazard": burden.hazard.value,
                "achieved_standard": burden.achieved_standard.value,
                "raw_sources_count": burden.raw_sources_count,
                "independent_sources_count": burden.independent_sources_count,
                "deduplicated_duplicates_collapsed": burden.deduplicated_duplicates_collapsed,
                "disputed": burden.disputed,
                "dispute_sources": list(burden.dispute_sources),
                "required_casual_standard": burden.required_casual_standard.value,
                "meets_casual_burden": burden.meets_casual_burden,
            }
        )


def run_mcp_stdio_server(
    *,
    device: str = "CPU",
    client_in: BinaryIO | None = None,
    client_out: BinaryIO | None = None,
) -> int:
    """Serve `epistemic-guard` verification tools over newline-delimited MCP stdio."""
    server = EpistemicGuardMcpServer(device=device)
    in_stream = client_in if client_in is not None else sys.stdin.buffer
    out_stream = client_out if client_out is not None else sys.stdout.buffer

    while True:
        raw_line = in_stream.readline()
        if not raw_line:
            return 0
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            decoded: object = json.loads(stripped.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
            out_stream.write((json.dumps(err_resp, separators=(",", ":")) + "\n").encode("utf-8"))
            out_stream.flush()
            continue

        response = server.handle_message(decoded)
        if response is not None:
            encoded = (json.dumps(response, separators=(",", ":")) + "\n").encode("utf-8")
            out_stream.write(encoded)
            out_stream.flush()
