"""Standalone stdio MCP JSON-RPC 2.0 server exposing `epistemic-guard` verification tools."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from typing import BinaryIO, Final

from epistemic_guard.corroboration import corroborate_sources
from epistemic_guard.exceptions import EpistemicGuardError
from epistemic_guard.mcp_proxy import verify_receipt_log_file
from epistemic_guard.schemas import ToolReceipt
from epistemic_guard.store import MerkleReceiptLog
from epistemic_guard.supervision import ReceiptSupervisor

_DEFAULT_PROTOCOL_VERSION: Final[str] = "2024-11-05"
_SUPPORTED_PROTOCOLS: Final[frozenset[str]] = frozenset(
    {"2024-11-05", "2025-03-26", "2025-06-18"}
)
_SERVER_VERSION: Final[str] = "0.4.2"

MCP_TOOLS: Final[tuple[dict[str, object], ...]] = (
    {
        "name": "verify_narration",
        "description": (
            "Audit an assistant reply clause-by-clause against structured in-memory tool "
            "execution receipts (or AER-1 envelopes) on local CPU encoders to detect fabricated "
            "action claims and generate a corrective retry prompt. Use this solely for semantic "
            "prose supervision; use verify_receipt_log for cryptographic JSONL file verification."
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
                        "Tool execution receipts for the current turn (ToolReceipt objects "
                        "with tool, ok, executed, args, output; pass [] if zero tools ran)."
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
                        },
                        "required": ["tool"],
                        "additionalProperties": True,
                    },
                },
                "user_prompt": {
                    "type": "string",
                    "description": "Optional user prompt for conversational context.",
                },
            },
            "required": ["narration", "receipts"],
            "additionalProperties": False,
        },
    },
    {
        "name": "verify_receipt_log",
        "description": (
            "Cryptographically verify an entire on-disk JSONL receipt log produced by "
            "`epistemic-guard mcp-proxy`: validates detached Ed25519 signatures, pinned public "
            "keys, sequential SHA-256 chain hashes, and the aggregate RFC 6962 Merkle root. "
            "Receipt signing occurs strictly inside the isolated mcp-proxy parent process so "
            "untrusted agents cannot mint receipts; use inspect_receipt to query single entries."
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
            },
            "required": ["log_path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "inspect_receipt",
        "description": (
            "Query a cryptographically verified JSONL receipt log by zero-based sequence index "
            "(`seq`), tool invocation ID (`call_id`), or tool name (`tool`), returning matching "
            "signed ToolReceipt records alongside standalone O(log N) RFC 6962 Merkle inclusion "
            "proofs and optional canonical AER-1 interoperable receipt envelopes."
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
    },
    {
        "name": "corroborate_sources",
        "description": (
            "Collapse syndicated wire-copy mirrors across retrieved RAG or web URLs via "
            "word-shingle MinHash/SimHash clustering, check cross-domain NLI contradictions, "
            "and evaluate whether independent sources meet hazard-scaled evidentiary burden."
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
                                "description": "Canonical URL or domain identifier of the passage.",
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
    },
)


def _coerce_receipt_args(raw_args: object) -> dict[str, str | int | float | bool]:
    if not isinstance(raw_args, Mapping):
        return {}
    clean: dict[str, str | int | float | bool] = {}
    for k, v in raw_args.items():
        key = str(k)
        if isinstance(v, (str, int, float, bool)):
            clean[key] = v
        else:
            clean[key] = json.dumps(v, sort_keys=True, separators=(",", ":"))
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
        parsed.append(
            ToolReceipt(
                call_id=call_id,
                tool=tool_name,
                ok=bool(ok_raw),
                executed=bool(exec_raw),
                error_code=str(err_raw) if err_raw is not None else "",
                args=_coerce_receipt_args(item.get("args")),
                output=str(out_raw) if out_raw is not None else "",
            )
        )
    return parsed


def _tool_result(payload: Mapping[str, object], *, is_error: bool = False) -> dict[str, object]:
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(dict(payload), sort_keys=True),
            }
        ],
        "isError": is_error,
    }


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

        verdict = self._get_supervisor().verify_narration(
            narration,
            receipts=receipts,
            user_prompt=user_prompt,
        )
        return _tool_result(
            {
                "is_valid": verdict.is_valid,
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

        verified = verify_receipt_log_file(
            log_path,
            expected_pubkey_hex=pubkey_hex,
            pubkey_file=pubkey_file,
        )
        return _tool_result(
            {
                "log_valid": True,
                "entry_count": verified.entry_count,
                "merkle_root": verified.merkle_root,
                "head_chain_hash": verified.head_chain_hash,
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

        verified = verify_receipt_log_file(
            log_path,
            expected_pubkey_hex=pubkey_hex,
            pubkey_file=pubkey_file,
        )
        if seq_filter is not None and seq_filter >= verified.entry_count:
            raise ValueError(
                f"Receipt seq={seq_filter} out of range for log with "
                f"{verified.entry_count} entries."
            )

        merkle = MerkleReceiptLog()
        for rcpt in verified.receipts:
            merkle.append(rcpt)

        candidate_indices = (
            [seq_filter] if seq_filter is not None else list(range(verified.entry_count))
        )
        entries_out: list[dict[str, object]] = []
        for idx in candidate_indices:
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
