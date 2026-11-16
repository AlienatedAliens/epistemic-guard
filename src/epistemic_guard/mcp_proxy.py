"""Out-of-process stdio MCP JSON-RPC 2.0 receipt proxy and Merkle log verifier."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Final

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import BaseModel, Field

from epistemic_guard.exceptions import MerkleTamperError, ProvenanceError, SignatureTamperError
from epistemic_guard.provenance import DomainSigner, ProvenanceGate
from epistemic_guard.schemas import Provenance, ToolReceipt
from epistemic_guard.store import MerkleReceiptLog

_PRE_EXEC_JSONRPC_CODES: Final[frozenset[int]] = frozenset({-32700, -32600, -32601, -32602})
_MCP_VALIDATION_MARKERS: Final[tuple[str, ...]] = (
    "input validation error",
    "invalid arguments",
    "invalid_arguments",
    "invalid tool arguments",
    "validation error for tool",
    "missing required argument",
    "missing required property",
)
_ENV_SIGNING_KEY_HEX: Final[str] = "EPISTEMIC_GUARD_MCP_KEY_HEX"


class ReceiptLogEntry(BaseModel):
    """Single JSONL record written by `ReceiptLogWriter`."""

    seq: int = Field(ge=0)
    receipt: ToolReceipt
    public_key_hex: str
    leaf_sha256: str
    chain_sha256: str
    merkle_root: str
    aer1: dict[str, object] | None = None


@dataclass(frozen=True)
class ReceiptLogVerification:
    """Result of verifying an append-only `mcp-proxy` JSONL receipt log."""

    entry_count: int
    merkle_root: str
    head_chain_hash: str
    public_key_hex: str
    key_pinned: bool
    receipts: tuple[ToolReceipt, ...]


@dataclass(frozen=True)
class _PendingToolCall:
    call_id: str
    tool: str
    args: dict[str, str | int | float | bool]


def _coerce_scalar(val: object) -> str | int | float | bool:
    if isinstance(val, (str, int, float, bool)):
        return val
    try:
        return json.dumps(val, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return str(val)


def _rpc_id_key(raw_id: object) -> tuple[str, str] | None:
    if raw_id is None or isinstance(raw_id, bool):
        return None
    if isinstance(raw_id, int):
        return ("int", str(raw_id))
    if isinstance(raw_id, str):
        return ("str", raw_id)
    if isinstance(raw_id, float):
        return ("float", repr(raw_id))
    return None


def _decode_seed_bytes(raw: bytes, *, source: str) -> bytes:
    stripped = raw.strip()
    if len(stripped) == 64:
        try:
            return bytes.fromhex(stripped.decode("ascii"))
        except ValueError:
            pass
    if len(raw) == 32:
        return raw
    raise ProvenanceError(
        f"{source} must contain a 32-byte raw seed or 64-character hex string."
    )


def load_or_create_mcp_signer(
    *,
    key_file: Path | str | None = None,
    pubkey_out: Path | str | None = None,
) -> DomainSigner:
    """Load or create an isolated `TOOL_VERIFIED` Ed25519 signer for `mcp-proxy`."""
    if key_file is not None:
        kpath = Path(key_file)
        if kpath.exists():
            seed = _decode_seed_bytes(kpath.read_bytes(), source=str(kpath))
        else:
            seed = os.urandom(32)
            kpath.parent.mkdir(parents=True, exist_ok=True)
            kpath.write_text(seed.hex() + "\n", encoding="ascii")
            with contextlib.suppress(OSError):
                os.chmod(kpath, 0o600)
    elif os.environ.get(_ENV_SIGNING_KEY_HEX):
        env_val = os.environ[_ENV_SIGNING_KEY_HEX].encode("ascii")
        seed = _decode_seed_bytes(env_val, source=_ENV_SIGNING_KEY_HEX)
    else:
        seed = os.urandom(32)

    priv = Ed25519PrivateKey.from_private_bytes(seed)
    signer = DomainSigner(Provenance.TOOL_VERIFIED, priv)

    if pubkey_out is not None:
        ppath = Path(pubkey_out)
        ppath.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "provenance": Provenance.TOOL_VERIFIED.value,
            "public_key_hex": signer.public_key_bytes.hex(),
        }
        ppath.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    return signer


def read_public_key_file(pubkey_file: Path | str) -> str:
    """Read a 64-char hex Ed25519 public key from JSON or raw hex file."""
    raw = Path(pubkey_file).read_text(encoding="utf-8").strip()
    if raw.startswith("{"):
        parsed = json.loads(raw)
        if isinstance(parsed, Mapping) and isinstance(parsed.get("public_key_hex"), str):
            hex_str = parsed["public_key_hex"].strip()
        else:
            raise ProvenanceError(f"Missing 'public_key_hex' in {pubkey_file}.")
    else:
        hex_str = raw
    try:
        key_bytes = bytes.fromhex(hex_str)
    except ValueError as exc:
        raise ProvenanceError(f"Invalid hex public key in {pubkey_file}.") from exc
    if len(key_bytes) != 32:
        raise ProvenanceError(f"Ed25519 public key in {pubkey_file} must be 32 bytes.")
    return key_bytes.hex()


class ReceiptLogWriter:
    """Append-only JSONL writer maintaining an Ed25519 + RFC 6962 Merkle chain on disk."""

    def __init__(
        self,
        log_path: Path | str,
        signer: DomainSigner,
        *,
        truncate: bool = False,
        include_aer1: bool = False,
    ) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._signer = signer
        self._pubkey_hex = signer.public_key_bytes.hex()
        self._include_aer1 = include_aer1
        self._lock = threading.Lock()
        self._gate = ProvenanceGate({Provenance.TOOL_VERIFIED: signer.public_key_bytes})
        self._merkle = MerkleReceiptLog(gate=self._gate)

        if truncate and self.log_path.exists():
            self.log_path.unlink()
        elif self.log_path.exists() and self.log_path.stat().st_size > 0:
            self._replay_existing()

    @property
    def merkle_log(self) -> MerkleReceiptLog:
        return self._merkle

    def _replay_existing(self) -> None:
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines):
            if not line.strip():
                continue
            entry = ReceiptLogEntry.model_validate_json(line)
            if entry.seq != idx:
                raise MerkleTamperError(
                    f"Existing log sequence mismatch at line {idx}: "
                    f"expected {idx}, got {entry.seq}."
                )
            if entry.public_key_hex != self._pubkey_hex:
                raise ProvenanceError(
                    "Existing receipt log was signed by a different Ed25519 key. "
                    "Pass matching --key-file or --truncate."
                )
            chain_sha = self._merkle.append(entry.receipt)
            proof = self._merkle.inclusion_proof(idx)
            if chain_sha != entry.chain_sha256 or proof.root_sha256 != entry.merkle_root:
                raise MerkleTamperError(f"Existing receipt log failed Merkle check at seq={idx}.")

    def append(self, receipt: ToolReceipt) -> ReceiptLogEntry:
        """Sign `receipt`, update the Merkle log, and flush one JSONL line to disk."""
        with self._lock:
            signed = (
                receipt
                if receipt.signature.startswith("ed25519:")
                else self._signer.sign_receipt(receipt)
            )
            seq = len(self._merkle)
            chain_sha = self._merkle.append(signed)
            proof = self._merkle.inclusion_proof(seq)
            entry = ReceiptLogEntry(
                seq=seq,
                receipt=signed,
                public_key_hex=self._pubkey_hex,
                leaf_sha256=proof.leaf_sha256,
                chain_sha256=chain_sha,
                merkle_root=proof.root_sha256,
                aer1=signed.to_aer1() if self._include_aer1 else None,
            )
            serialized = json.dumps(
                entry.model_dump(exclude_none=True),
                sort_keys=True,
                separators=(",", ":"),
            )
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(serialized + "\n")
                handle.flush()
            return entry


def verify_receipt_log_file(
    log_path: Path | str,
    *,
    expected_pubkey_hex: str | None = None,
    pubkey_file: Path | str | None = None,
) -> ReceiptLogVerification:
    """Verify Ed25519 signatures, sequential hash links, and Merkle roots in a JSONL log."""
    pinned_hex: str | None = None
    if pubkey_file is not None:
        pinned_hex = read_public_key_file(pubkey_file)
    if expected_pubkey_hex is not None:
        norm = expected_pubkey_hex.strip().lower()
        if len(bytes.fromhex(norm)) != 32:
            raise ProvenanceError("expected_pubkey_hex must decode to 32 bytes.")
        if pinned_hex is not None and pinned_hex != norm:
            raise ProvenanceError("Conflicting expected_pubkey_hex and pubkey_file values.")
        pinned_hex = norm

    path = Path(log_path)
    raw_lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not raw_lines:
        empty_log = MerkleReceiptLog()
        return ReceiptLogVerification(
            entry_count=0,
            merkle_root=empty_log.merkle_root(),
            head_chain_hash=empty_log.head_chain_hash,
            public_key_hex=pinned_hex or "",
            key_pinned=pinned_hex is not None,
            receipts=(),
        )

    entries: list[ReceiptLogEntry] = []
    for idx, raw in enumerate(raw_lines):
        try:
            entries.append(ReceiptLogEntry.model_validate_json(raw))
        except Exception as exc:
            raise MerkleTamperError(f"Malformed receipt log entry at line {idx}: {exc}") from exc

    active_pub_hex = pinned_hex or entries[0].public_key_hex.strip().lower()
    try:
        pub_bytes = bytes.fromhex(active_pub_hex)
    except ValueError as exc:
        raise SignatureTamperError("Invalid public_key_hex in receipt log.") from exc
    if len(pub_bytes) != 32:
        raise SignatureTamperError("Ed25519 public key must be 32 bytes.")

    gate = ProvenanceGate({Provenance.TOOL_VERIFIED: pub_bytes})
    merkle = MerkleReceiptLog(gate=gate)

    for idx, entry in enumerate(entries):
        if entry.seq != idx:
            raise MerkleTamperError(
                f"Receipt sequence gap or reordering at index {idx}: found seq={entry.seq}."
            )
        if entry.public_key_hex.strip().lower() != active_pub_hex:
            raise SignatureTamperError(
                f"Public key mismatch at seq={idx}: expected {active_pub_hex}, "
                f"got {entry.public_key_hex}."
            )
        chain_sha = merkle.append(entry.receipt)
        proof = merkle.inclusion_proof(idx)
        if proof.leaf_sha256 != entry.leaf_sha256:
            raise MerkleTamperError(f"Leaf SHA-256 mismatch at seq={idx}.")
        if chain_sha != entry.chain_sha256:
            raise MerkleTamperError(f"Chain SHA-256 mismatch at seq={idx}.")
        if proof.root_sha256 != entry.merkle_root:
            raise MerkleTamperError(f"Merkle root mismatch at seq={idx}.")

    return ReceiptLogVerification(
        entry_count=len(entries),
        merkle_root=merkle.merkle_root(),
        head_chain_hash=merkle.head_chain_hash,
        public_key_hex=active_pub_hex,
        key_pinned=pinned_hex is not None,
        receipts=merkle.receipts,
    )


def _extract_mcp_result_text(result: object) -> tuple[bool, str, bool]:
    """Return `(ok, output_text, is_validation_rejection)` for an MCP `tools/call` result."""
    if not isinstance(result, Mapping):
        return (True, json.dumps(result, sort_keys=True, separators=(",", ":")), False)

    is_err = bool(result.get("isError", False))
    content = result.get("content")
    text_parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "text":
                txt = block.get("text")
                if isinstance(txt, str):
                    text_parts.append(txt)

    if text_parts:
        output_text = "\n".join(text_parts)
    else:
        output_text = json.dumps(result, sort_keys=True, separators=(",", ":"))

    if not is_err:
        return (True, output_text, False)

    lowered = output_text.lower()
    is_validation = any(marker in lowered for marker in _MCP_VALIDATION_MARKERS)
    return (False, output_text, is_validation)


class McpStdioProxy:
    """Out-of-process stdio JSON-RPC 2.0 proxy recording signed `ToolReceipt`s."""

    def __init__(self, writer: ReceiptLogWriter) -> None:
        self._writer = writer
        self._pending: dict[tuple[str, str], _PendingToolCall] = {}
        self._lock = threading.Lock()

    def observe_client_line(self, raw_line: bytes) -> None:
        """Inspect a client->server JSON-RPC line and register `tools/call` requests."""
        stripped = raw_line.strip()
        if not stripped or stripped[:1] not in (b"{", b"["):
            return
        try:
            payload = json.loads(stripped.decode("utf-8"))
        except Exception:
            return
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, Mapping) or item.get("method") != "tools/call":
                continue
            id_key = _rpc_id_key(item.get("id"))
            if id_key is None:
                continue
            params = item.get("params")
            if isinstance(params, Mapping):
                raw_name = params.get("name")
                tool_name = (
                    raw_name
                    if isinstance(raw_name, str) and raw_name
                    else "<missing_tool_name>"
                )
                raw_args = params.get("arguments")
                if isinstance(raw_args, Mapping):
                    args_map = {str(k): _coerce_scalar(v) for k, v in raw_args.items()}
                elif raw_args is None:
                    args_map = {}
                else:
                    args_map = {"_raw_arguments": _coerce_scalar(raw_args)}
            else:
                tool_name = "<invalid_params>"
                args_map = {"_raw_params": _coerce_scalar(params)}

            with self._lock:
                self._pending[id_key] = _PendingToolCall(
                    call_id=f"mcp-{id_key[1]}",
                    tool=tool_name,
                    args=args_map,
                )

    def observe_server_line(self, raw_line: bytes) -> None:
        """Inspect a server->client JSON-RPC line and commit receipts for completed calls."""
        stripped = raw_line.strip()
        if not stripped or stripped[:1] not in (b"{", b"["):
            return
        try:
            payload = json.loads(stripped.decode("utf-8"))
        except Exception:
            return
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, Mapping) or "method" in item:
                continue
            id_key = _rpc_id_key(item.get("id"))
            if id_key is None:
                continue
            with self._lock:
                pending = self._pending.pop(id_key, None)
            if pending is None:
                continue
            receipt = self._build_receipt_from_response(pending, item)
            self._writer.append(receipt)

    def flush_abandoned_calls(self, exit_code: int) -> None:
        """Record `ok=False` receipts for in-flight calls if the MCP server exits early."""
        with self._lock:
            remaining = list(self._pending.values())
            self._pending.clear()
        for pending in remaining:
            self._writer.append(
                ToolReceipt(
                    call_id=pending.call_id,
                    tool=pending.tool,
                    ok=False,
                    executed=True,
                    error_code="mcp_server_eof",
                    args=pending.args,
                    output=f"MCP server exited (code {exit_code}) before responding.",
                )
            )

    @staticmethod
    def _build_receipt_from_response(
        pending: _PendingToolCall,
        response: Mapping[str, object],
    ) -> ToolReceipt:
        if "error" in response:
            err = response.get("error")
            code = -32000
            msg = "JSON-RPC error"
            if isinstance(err, Mapping):
                raw_code = err.get("code")
                if isinstance(raw_code, int) and not isinstance(raw_code, bool):
                    code = raw_code
                raw_msg = err.get("message")
                if isinstance(raw_msg, str):
                    msg = raw_msg
                if "data" in err and err["data"] is not None:
                    msg = f"{msg} ({_coerce_scalar(err['data'])})"
            executed = code not in _PRE_EXEC_JSONRPC_CODES
            return ToolReceipt(
                call_id=pending.call_id,
                tool=pending.tool,
                ok=False,
                executed=executed,
                error_code=f"jsonrpc_{code}",
                args=pending.args,
                output=f"JSON-RPC {code}: {msg}",
            )

        ok, output_text, is_validation = _extract_mcp_result_text(response.get("result"))
        if ok:
            return ToolReceipt(
                call_id=pending.call_id,
                tool=pending.tool,
                ok=True,
                executed=True,
                error_code="",
                args=pending.args,
                output=output_text,
            )
        return ToolReceipt(
            call_id=pending.call_id,
            tool=pending.tool,
            ok=False,
            executed=not is_validation,
            error_code="invalid_tool_arguments" if is_validation else "mcp_tool_error",
            args=pending.args,
            output=output_text,
        )


def run_mcp_stdio_proxy(
    server_cmd: Sequence[str],
    *,
    log_path: Path | str,
    key_file: Path | str | None = None,
    pubkey_out: Path | str | None = None,
    truncate_log: bool = False,
    include_aer1: bool = False,
    client_in: BinaryIO | None = None,
    client_out: BinaryIO | None = None,
) -> int:
    """Spawn `server_cmd` and proxy MCP stdio while logging Ed25519 `ToolReceipt`s."""
    if not server_cmd:
        raise ValueError("server_cmd must include at least the target executable.")

    signer = load_or_create_mcp_signer(key_file=key_file, pubkey_out=pubkey_out)
    writer = ReceiptLogWriter(
        log_path,
        signer,
        truncate=truncate_log,
        include_aer1=include_aer1,
    )
    proxy = McpStdioProxy(writer)

    child_env = dict(os.environ)
    child_env.pop(_ENV_SIGNING_KEY_HEX, None)

    in_stream = client_in if client_in is not None else sys.stdin.buffer
    out_stream = client_out if client_out is not None else sys.stdout.buffer

    proc = subprocess.Popen(
        list(server_cmd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        env=child_env,
    )
    assert proc.stdin is not None  # noqa: S101
    assert proc.stdout is not None  # noqa: S101
    child_stdin = proc.stdin
    child_stdout = proc.stdout

    def _forward_stdin() -> None:
        try:
            while True:
                line = in_stream.readline()
                if not line:
                    break
                proxy.observe_client_line(line)
                child_stdin.write(line)
                child_stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass
        finally:
            with contextlib.suppress(OSError, ValueError):
                child_stdin.close()

    pump = threading.Thread(target=_forward_stdin, name="mcp-proxy-stdin", daemon=True)
    pump.start()

    try:
        while True:
            line = child_stdout.readline()
            if not line:
                break
            proxy.observe_server_line(line)
            out_stream.write(line)
            out_stream.flush()
    except (BrokenPipeError, OSError):
        pass
    finally:
        with contextlib.suppress(OSError, ValueError):
            child_stdout.close()

    rc = proc.wait()
    pump.join(timeout=0.5)
    proxy.flush_abandoned_calls(rc)
    return int(rc)
