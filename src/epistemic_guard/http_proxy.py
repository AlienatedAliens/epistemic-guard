"""Streamable HTTP / SSE MCP reverse proxy with Ed25519 Merkle receipts, `tools/list` rug-pull
pinning, and durable Lost-Response Contract (`A1`–`A5`) idempotent operation recovery.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.server
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final
from uuid import uuid4

from epistemic_guard.exceptions import ToolPinMismatchError
from epistemic_guard.mcp_proxy import (
    _PRE_EXEC_JSONRPC_CODES,
    ReceiptLogWriter,
    _classify_mcp_result,
    _coerce_scalar,
    _sanitize_tools_list_item,
    compute_mcp_tool_sha256,
    filter_mcp_result_object,
    load_or_create_mcp_signer,
)
from epistemic_guard.schemas import ToolReceipt

_META_OP_KEYS: Final[tuple[str, ...]] = (
    "operation_key",
    "operationKey",
    "idempotency_key",
    "idempotencyKey",
    "operation_id",
    "operationId",
)
_FORWARDED_HEADER_ALLOWLIST: Final[frozenset[str]] = frozenset(
    {
        "accept",
        "authorization",
        "content-type",
        "mcp-session-id",
        "mcp-protocol-version",
    }
)


def _canonical_operation_input_sha256(tool_name: str, args_map: Mapping[str, object]) -> str:
    payload = {"tool": tool_name, "arguments": dict(args_map)}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(blob).hexdigest()


def _extract_operation_key(
    item: Mapping[str, object], headers: Mapping[str, str]
) -> str | None:
    for hkey, hval in headers.items():
        if hkey.lower() in {"idempotency-key", "x-operation-key", "x-idempotency-key"}:
            stripped = hval.strip()
            if stripped:
                return stripped
    params = item.get("params")
    if isinstance(params, Mapping):
        meta = params.get("_meta")
        if isinstance(meta, Mapping):
            for mkey in _META_OP_KEYS:
                raw = meta.get(mkey)
                if isinstance(raw, str) and raw.strip():
                    return raw.strip()
        for mkey in _META_OP_KEYS:
            raw = params.get(mkey)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
    return None


def _parse_sse_jsonrpc_messages(raw_body: bytes) -> list[dict[str, object]]:
    """Extract JSON-RPC objects emitted across `text/event-stream` (`data: ...`) SSE frames."""
    text = raw_body.decode("utf-8", errors="replace")
    messages: list[dict[str, object]] = []
    data_lines: list[str] = []

    def _flush_event() -> None:
        if not data_lines:
            return
        joined = "\n".join(data_lines).strip()
        data_lines.clear()
        if not joined or joined[:1] not in ("{", "["):
            return
        with contextlib.suppress(Exception):
            parsed = json.loads(joined)
            if isinstance(parsed, Mapping):
                messages.append(dict(parsed))
            elif isinstance(parsed, list):
                for el in parsed:
                    if isinstance(el, Mapping):
                        messages.append(dict(el))

    for line in text.splitlines():
        if not line.strip():
            _flush_event()
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    _flush_event()
    return messages


class McpStreamableHttpProxy:
    """Mediates JSON-RPC 2.0 Streamable HTTP / SSE requests to an upstream MCP endpoint."""

    def __init__(
        self,
        upstream_url: str,
        writer: ReceiptLogWriter,
        *,
        pin_tools: bool = False,
        strip_untrusted_annotations: bool = False,
        raise_on_rugpull: bool = False,
        filter_web_results: bool = False,
        timeout_s: float = 30.0,
    ) -> None:
        parsed = urllib.parse.urlsplit(upstream_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                f"upstream_url must be a valid http:// or https:// URL; got {upstream_url!r}."
            )
        self.upstream_url = upstream_url.strip()
        self.writer = writer
        self.pin_tools = pin_tools
        self.strip_untrusted_annotations = strip_untrusted_annotations
        self.raise_on_rugpull = raise_on_rugpull
        self.filter_web_results = filter_web_results
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._call_seq = 0
        self._rugpull_seq = 0
        self._tool_pins: dict[str, str] = {}
        self._compromised_tools: set[str] = set()
        self._observed_tools: dict[str, dict[str, Any]] = {}

    @property
    def observed_tools(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(v) for v in self._observed_tools.values()]

    @property
    def tool_pins(self) -> dict[str, str]:
        with self._lock:
            return dict(self._tool_pins)

    def _next_call_id(self, prefix: str = "http-mcp") -> str:
        with self._lock:
            self._call_seq += 1
            return f"{prefix}-{self._call_seq}-{uuid4().hex[:8]}"

    def recover_operation(self, operation_key: str) -> tuple[int, dict[str, object]]:
        """Lost-Response Contract `A3` & `A5`: resolve committed digest by `K` or `receipt_id`."""
        entry = self.writer.lookup_receipt_or_operation(operation_key)
        if entry is None:
            return (
                404,
                {
                    "status": "not_found",
                    "operation_key": operation_key,
                    "error": f"No committed receipt found for operation_key={operation_key!r}.",
                },
            )
        aer1_doc = entry.aer1 if entry.aer1 is not None else entry.receipt.to_aer1()
        out_sha = entry.receipt.output_sha256
        return (
            200,
            {
                "status": "committed" if entry.receipt.ok else "failed",
                "operation_key": operation_key,
                "call_id": entry.receipt.call_id,
                "tool": entry.receipt.tool,
                "ok": entry.receipt.ok,
                "executed": entry.receipt.executed,
                "output": entry.receipt.output,
                "output_sha256": out_sha,
                "aer1_output_hash": str(aer1_doc["output_hash"]),
                "receipt": entry.receipt.model_dump(),
                "aer1": aer1_doc,
                "merkle_root": entry.merkle_root,
                "cached_rpc_result": entry.cached_rpc_result,
            },
        )

    def _inspect_tools_list(self, resp_obj: Mapping[str, object]) -> list[str]:
        res = resp_obj.get("result")
        if not isinstance(res, Mapping):
            return []
        tools = res.get("tools")
        if not isinstance(tools, list):
            return []
        mutated: list[tuple[str, str, str]] = []
        with self._lock:
            for tdef in tools:
                if not isinstance(tdef, Mapping):
                    continue
                tname = tdef.get("name")
                if not isinstance(tname, str) or not tname:
                    continue
                digest = compute_mcp_tool_sha256(tdef)
                prev = self._tool_pins.get(tname)
                if prev is None:
                    self._tool_pins[tname] = digest
                    self._observed_tools[tname] = dict(tdef)
                elif prev != digest:
                    mutated.append((tname, prev, digest))
                    if self.pin_tools:
                        self._compromised_tools.add(tname)
                    else:
                        self._tool_pins[tname] = digest
                        self._observed_tools[tname] = dict(tdef)

        for tname, old_h, new_h in mutated:
            with self._lock:
                self._rugpull_seq += 1
                rseq = self._rugpull_seq
            self.writer.append(
                ToolReceipt(
                    call_id=f"http-rugpull-{tname}-{rseq}-{uuid4().hex[:8]}",
                    tool=tname,
                    ok=False,
                    executed=False,
                    error_code="mcp_tool_rugpull",
                    args={"pinned_sha256": old_h, "observed_sha256": new_h},
                    output=(
                        f"HTTP MCP tools/list schema mutation detected for '{tname}': "
                        f"pinned={old_h[:16]}... != observed={new_h[:16]}..."
                    ),
                    tool_definition_sha256=old_h,
                    endpoint_id=self.upstream_url,
                )
            )
            if self.raise_on_rugpull:
                raise ToolPinMismatchError(
                    f"HTTP MCP tool '{tname}' mutated mid-session: {old_h} -> {new_h}"
                )
        return [m[0] for m in mutated]

    def _build_receipt(
        self,
        call_id: str,
        tool_name: str,
        args_map: dict[str, str | int | float | bool],
        response_obj: Mapping[str, object],
        *,
        tool_def_sha256: str = "",
    ) -> ToolReceipt:
        if "error" in response_obj:
            err = response_obj.get("error")
            code = -32000
            msg = "JSON-RPC error"
            if isinstance(err, Mapping):
                raw_code = err.get("code")
                if isinstance(raw_code, int) and not isinstance(raw_code, bool):
                    code = raw_code
                raw_msg = err.get("message")
                if isinstance(raw_msg, str):
                    msg = raw_msg
            executed = code not in _PRE_EXEC_JSONRPC_CODES
            return ToolReceipt(
                call_id=call_id,
                tool=tool_name,
                ok=False,
                executed=executed,
                error_code=f"jsonrpc_{code}",
                args=args_map,
                output=f"JSON-RPC {code}: {msg}",
                tool_definition_sha256=tool_def_sha256,
                endpoint_id=self.upstream_url,
            )

        status, output_text, is_validation = _classify_mcp_result(
            response_obj.get("result"), tool_name
        )
        if status == "indeterminate":
            return ToolReceipt(
                call_id=call_id,
                tool=tool_name,
                ok=False,
                executed=True,
                error_code="mcp_tool_result_indeterminate",
                args=args_map,
                output=output_text,
                tool_definition_sha256=tool_def_sha256,
                endpoint_id=self.upstream_url,
            )
        if status == "ok":
            return ToolReceipt(
                call_id=call_id,
                tool=tool_name,
                ok=True,
                executed=True,
                error_code="",
                args=args_map,
                output=output_text,
                tool_definition_sha256=tool_def_sha256,
                endpoint_id=self.upstream_url,
            )
        return ToolReceipt(
            call_id=call_id,
            tool=tool_name,
            ok=False,
            executed=not is_validation,
            error_code="invalid_tool_arguments" if is_validation else "mcp_tool_error",
            args=args_map,
            output=output_text,
            tool_definition_sha256=tool_def_sha256,
            endpoint_id=self.upstream_url,
        )

    def handle_post(
        self,
        raw_body: bytes,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        """Forward or idempotently resolve a JSON-RPC 2.0 `POST` payload."""
        hdrs = dict(headers or {})
        try:
            parsed_req = json.loads(raw_body.decode("utf-8"))
        except Exception as exc:
            err_doc = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            }
            return (
                400,
                {"Content-Type": "application/json"},
                json.dumps(err_doc, separators=(",", ":")).encode("utf-8"),
            )

        is_single_map = isinstance(parsed_req, Mapping)
        method = str(parsed_req.get("method") or "") if is_single_map else ""
        rpc_id = parsed_req.get("id") if is_single_map else None

        tool_name = "<unknown>"
        args_map: dict[str, str | int | float | bool] = {}
        op_key: str | None = None
        input_sha: str | None = None
        tool_def_sha = ""

        if is_single_map and method == "tools/call":
            params = parsed_req.get("params")
            if isinstance(params, Mapping):
                rname = params.get("name")
                tool_name = rname if isinstance(rname, str) and rname else "<missing_tool_name>"
                rargs = params.get("arguments")
                if isinstance(rargs, Mapping):
                    args_map = {str(k): _coerce_scalar(v) for k, v in rargs.items()}
                elif rargs is not None:
                    args_map = {"_raw_arguments": _coerce_scalar(rargs)}
            else:
                tool_name = "<invalid_params>"
                args_map = {"_raw_params": _coerce_scalar(params)}

            with self._lock:
                tool_def_sha = self._tool_pins.get(tool_name, "")
                is_compromised = self.pin_tools and tool_name in self._compromised_tools

            if is_compromised:
                refuse_cid = self._next_call_id("http-pin-refused")
                self.writer.append(
                    ToolReceipt(
                        call_id=refuse_cid,
                        tool=tool_name,
                        ok=False,
                        executed=False,
                        error_code="mcp_tool_pin_refused",
                        args=args_map,
                        output=(
                            f"Refused tools/call for '{tool_name}': tool definition changed "
                            "mid-session from approved pin."
                        ),
                        tool_definition_sha256=tool_def_sha,
                        endpoint_id=self.upstream_url,
                    )
                )
                refuse_rpc = {
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "error": {
                        "code": -32603,
                        "message": (
                            f"MCP tool '{tool_name}' invocation refused: definition digest "
                            "differs from approved session pin."
                        ),
                    },
                }
                return (
                    409,
                    {
                        "Content-Type": "application/json",
                        "X-Epistemic-Guard-Outcome": "refused_tool_pin_mismatch",
                    },
                    json.dumps(refuse_rpc, separators=(",", ":")).encode("utf-8"),
                )

            op_key = _extract_operation_key(parsed_req, hdrs)
            if op_key is not None:
                input_sha = _canonical_operation_input_sha256(tool_name, args_map)
                prior = self.writer.lookup_operation(op_key)
                if prior is not None:
                    # A2: Mismatched tool/parameter retry under existing operation key -> 409
                    if prior.operation_input_sha256 != input_sha:
                        conflict_cid = self._next_call_id("http-conflict")
                        self.writer.append(
                            ToolReceipt(
                                call_id=conflict_cid,
                                tool=tool_name,
                                ok=False,
                                executed=False,
                                error_code="idempotency_parameter_mismatch",
                                args=args_map,
                                output=(
                                    f"Operation key {op_key!r} already bound to a different "
                                    "tool/argument commitment."
                                ),
                                tool_definition_sha256=tool_def_sha,
                                endpoint_id=self.upstream_url,
                            )
                        )
                        conflict_rpc = {
                            "jsonrpc": "2.0",
                            "id": rpc_id,
                            "error": {
                                "code": -32602,
                                "message": (
                                    f"Idempotency binding mismatch for operation_key={op_key!r}: "
                                    "parameters differ from original committed request."
                                ),
                                "data": {
                                    "error_code": "idempotency_parameter_mismatch",
                                    "operation_key": op_key,
                                },
                            },
                        }
                        return (
                            409,
                            {
                                "Content-Type": "application/json",
                                "X-Epistemic-Guard-Outcome": "conflict_parameter_mismatch",
                            },
                            json.dumps(conflict_rpc, separators=(",", ":")).encode("utf-8"),
                        )

                    # A1 & A4: Identical retry -> replay committed outcome with 0 new executions
                    replayed_rpc: dict[str, object]
                    if prior.cached_rpc_result is not None:
                        replayed_rpc = dict(prior.cached_rpc_result)
                        replayed_rpc["id"] = rpc_id
                    else:
                        replayed_rpc = {
                            "jsonrpc": "2.0",
                            "id": rpc_id,
                            "result": {
                                "content": [{"type": "text", "text": prior.receipt.output}],
                                "isError": not prior.receipt.ok,
                            },
                        }
                    aer1_doc = (
                        prior.aer1 if prior.aer1 is not None else prior.receipt.to_aer1()
                    )
                    return (
                        200,
                        {
                            "Content-Type": "application/json",
                            "X-Epistemic-Guard-Replayed": "true",
                            "X-Epistemic-Guard-Receipt-Id": prior.receipt.call_id,
                            "X-Epistemic-Guard-Output-Sha256": prior.receipt.output_sha256,
                            "X-Epistemic-Guard-Aer1-Output-Hash": str(aer1_doc["output_hash"]),
                        },
                        json.dumps(replayed_rpc, separators=(",", ":")).encode("utf-8"),
                    )

        fwd_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        for hk, hv in hdrs.items():
            if hk.lower() in _FORWARDED_HEADER_ALLOWLIST:
                fwd_headers[hk] = hv

        req = urllib.request.Request(
            self.upstream_url,
            data=raw_body,
            headers=fwd_headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                status_code = int(resp.status)
                content_type = str(resp.headers.get("Content-Type") or "application/json")
                mcp_sess_hdr = resp.headers.get("Mcp-Session-Id")
                resp_body = resp.read()
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")[:512]
            if is_single_map and method == "tools/call":
                cid = self._next_call_id("http-upstream-err")
                self.writer.append(
                    ToolReceipt(
                        call_id=cid,
                        tool=tool_name,
                        ok=False,
                        executed=int(exc.code) >= 500,
                        error_code=f"mcp_http_{exc.code}",
                        args=args_map,
                        output=f"Upstream HTTP {exc.code}: {err_body}",
                        tool_definition_sha256=tool_def_sha,
                        endpoint_id=self.upstream_url,
                    )
                )
            rpc_err = {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {
                    "code": -32000,
                    "message": f"Upstream MCP HTTP {exc.code}",
                    "data": {"error_code": f"mcp_http_{exc.code}", "detail": err_body},
                },
            }
            return (
                502 if int(exc.code) >= 500 else int(exc.code),
                {"Content-Type": "application/json"},
                json.dumps(rpc_err, separators=(",", ":")).encode("utf-8"),
            )
        except Exception as exc:
            # Persistence-gap / dropped upstream transport -> explicit unresolved error (A4)
            if is_single_map and method == "tools/call":
                cid = self._next_call_id("http-drop")
                self.writer.append(
                    ToolReceipt(
                        call_id=cid,
                        tool=tool_name,
                        ok=False,
                        executed=True,
                        error_code="mcp_http_upstream_indeterminate",
                        args=args_map,
                        output=f"Upstream connection failed before response: {exc}",
                        tool_definition_sha256=tool_def_sha,
                        endpoint_id=self.upstream_url,
                    )
                )
            rpc_drop = {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {
                    "code": -32000,
                    "message": f"Upstream MCP transport error (outcome indeterminate): {exc}",
                    "data": {"error_code": "mcp_http_upstream_indeterminate"},
                },
            }
            return (
                502,
                {"Content-Type": "application/json"},
                json.dumps(rpc_drop, separators=(",", ":")).encode("utf-8"),
            )

        out_headers: dict[str, str] = {"Content-Type": content_type}
        if mcp_sess_hdr:
            out_headers["Mcp-Session-Id"] = mcp_sess_hdr

        is_sse = "text/event-stream" in content_type.lower()
        rpc_messages: list[dict[str, object]] = []
        if is_sse:
            rpc_messages = _parse_sse_jsonrpc_messages(resp_body)
        else:
            try:
                decoded = json.loads(resp_body.decode("utf-8"))
                if isinstance(decoded, Mapping):
                    rpc_messages = [dict(decoded)]
                elif isinstance(decoded, list):
                    rpc_messages = [dict(x) for x in decoded if isinstance(x, Mapping)]
            except Exception:
                rpc_messages = []

        if is_single_map and method == "tools/list":
            for msg_obj in rpc_messages:
                if "result" in msg_obj:
                    mutated_names = self._inspect_tools_list(msg_obj)
                    if mutated_names and self.pin_tools:
                        rug_err = {
                            "jsonrpc": "2.0",
                            "id": rpc_id,
                            "error": {
                                "code": -32603,
                                "message": (
                                    "MCP tools/list pin mismatch (rug-pull blocked): "
                                    + ", ".join(mutated_names)
                                ),
                            },
                        }
                        return (
                            409,
                            {"Content-Type": "application/json"},
                            json.dumps(rug_err, separators=(",", ":")).encode("utf-8"),
                        )
                    if self.strip_untrusted_annotations and not is_sse:
                        san_obj, san_changed = _sanitize_tools_list_item(msg_obj)
                        if san_changed:
                            resp_body = json.dumps(san_obj, separators=(",", ":")).encode("utf-8")
                            out_headers["Content-Length"] = str(len(resp_body))

        if is_single_map and method == "tools/call":
            matched_resp: dict[str, object] | None = None
            for msg_obj in rpc_messages:
                if "method" not in msg_obj and msg_obj.get("id") == rpc_id:
                    matched_resp = msg_obj
                    break
            if matched_resp is None and rpc_messages:
                matched_resp = rpc_messages[-1]

            if matched_resp is not None:
                active_resp = matched_resp
                if self.filter_web_results and not is_sse:
                    filtered_resp, changes = filter_mcp_result_object(tool_name, matched_resp)
                    if changes > 0:
                        active_resp = filtered_resp
                        resp_body = json.dumps(active_resp, separators=(",", ":")).encode("utf-8")

                cid = self._next_call_id("http-mcp")
                rcpt = self._build_receipt(
                    cid,
                    tool_name,
                    args_map,
                    active_resp,
                    tool_def_sha256=tool_def_sha,
                )
                entry = self.writer.append(
                    rcpt,
                    operation_key=op_key,
                    operation_input_sha256=input_sha,
                    cached_rpc_result=active_resp,
                )
                out_headers["X-Epistemic-Guard-Replayed"] = "false"
                out_headers["X-Epistemic-Guard-Receipt-Id"] = entry.receipt.call_id
                out_headers["X-Epistemic-Guard-Output-Sha256"] = entry.receipt.output_sha256
                if entry.receipt.tool_definition_sha256:
                    out_headers["X-Epistemic-Guard-Tool-Def-Sha256"] = (
                        entry.receipt.tool_definition_sha256
                    )

        return (status_code, out_headers, resp_body)


class McpHttpProxyServer(http.server.ThreadingHTTPServer):
    """Threaded HTTP server binding `McpStreamableHttpProxy` (`POST` + `GET /receipts/<key>`)."""

    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        proxy: McpStreamableHttpProxy,
    ) -> None:
        self.proxy = proxy
        super().__init__(server_address, _McpProxyRequestHandler)


class _McpProxyRequestHandler(http.server.BaseHTTPRequestHandler):
    server: McpHttpProxyServer

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path == "/health":
            sess = self.server.proxy.writer.session_id
            body = json.dumps({"ok": True, "session_id": sess}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path.startswith("/receipts/"):
            op_key = urllib.parse.unquote(parsed.path[len("/receipts/") :]).strip("/")
            code, doc = self.server.proxy.recover_operation(op_key)
            body = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        not_found = b'{"error":"Not Found"}'
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(not_found)))
        self.end_headers()
        self.wfile.write(not_found)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw_body = self.rfile.read(length) if length > 0 else b""
        hdr_map = {str(k): str(v) for k, v in self.headers.items()}
        code, resp_hdrs, resp_body = self.server.proxy.handle_post(raw_body, headers=hdr_map)
        self.send_response(code)
        for hk, hv in resp_hdrs.items():
            self.send_header(hk, hv)
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        if resp_body:
            try:
                self.wfile.write(resp_body)
            except BrokenPipeError:
                pass


def run_mcp_http_proxy(
    upstream_url: str,
    *,
    log_path: Path | str,
    host: str = "127.0.0.1",
    port: int = 8765,
    key_file: Path | str | None = None,
    pubkey_out: Path | str | None = None,
    truncate_log: bool = False,
    include_aer1: bool = False,
    pin_tools: bool = False,
    strip_untrusted_annotations: bool = False,
    filter_web_results: bool = False,
    session_id: str | None = None,
    prev_log_path: Path | str | None = None,
    prev_pubkey_file: Path | str | None = None,
    timeout_s: float = 30.0,
) -> int:
    """Run a blocking Streamable HTTP MCP receipt proxy until `KeyboardInterrupt`."""
    signer = load_or_create_mcp_signer(key_file=key_file, pubkey_out=pubkey_out)
    writer = ReceiptLogWriter(
        log_path,
        signer,
        truncate=truncate_log,
        include_aer1=include_aer1,
        session_id=session_id,
        prev_log_path=prev_log_path,
        prev_pubkey_file=prev_pubkey_file,
    )
    proxy = McpStreamableHttpProxy(
        upstream_url,
        writer,
        pin_tools=pin_tools,
        strip_untrusted_annotations=strip_untrusted_annotations,
        filter_web_results=filter_web_results,
        timeout_s=timeout_s,
    )
    httpd = McpHttpProxyServer((host, port), proxy)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0
