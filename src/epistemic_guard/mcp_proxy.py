"""Out-of-process stdio MCP JSON-RPC 2.0 receipt proxy and Merkle log verifier."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Final
from uuid import uuid4

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import BaseModel, Field

from epistemic_guard.aer1 import export_receipts_to_aer1_chain_v07
from epistemic_guard.exceptions import (
    MerkleTamperError,
    ProvenanceError,
    SignatureTamperError,
    ToolPinMismatchError,
)
from epistemic_guard.provenance import DomainSigner, ProvenanceGate
from epistemic_guard.schemas import Provenance, ToolReceipt
from epistemic_guard.store import MerkleReceiptLog
from epistemic_guard.web_filters import clean_scraped_passage, cluster_near_duplicate_sources

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
_ENV_MASTER_KEY: Final[str] = "EPISTEMIC_GUARD_MASTER_KEY"
_ENV_SESSION_ID: Final[str] = "EPISTEMIC_GUARD_SESSION_ID"
_SCRUBBED_EXACT_ENV_KEYS: Final[frozenset[str]] = frozenset(
    {_ENV_SIGNING_KEY_HEX, _ENV_MASTER_KEY}
)
_GENESIS_CHAIN_HEX: Final[str] = "0" * 64


def _scrub_child_env(env: Mapping[str, str]) -> dict[str, str]:
    """Strip proxy signing keys and master secrets (case-insensitively) from child env."""
    cleaned: dict[str, str] = {}
    for key, val in env.items():
        ku = key.upper()
        if ku in _SCRUBBED_EXACT_ENV_KEYS:
            continue
        if ku.startswith("EPISTEMIC_GUARD_") and any(
            tok in ku for tok in ("KEY", "SECRET", "SEED", "TOKEN")
        ):
            continue
        cleaned[key] = val
    return cleaned


@contextlib.contextmanager
def _exclusive_log_lock(log_path: Path, *, timeout_s: float = 10.0) -> Iterator[None]:
    """Acquire an OS-level exclusive lock on `<log_path>.lock` across worker processes."""
    lock_file = log_path.parent / f"{log_path.name}.lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_file.open("a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\x00")
            handle.flush()
        deadline = time.monotonic() + timeout_s
        locked = False
        while True:
            try:
                handle.seek(0)
                if sys.platform == "win32":
                    import msvcrt  # noqa: PLC0415

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl  # noqa: PLC0415

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise ProvenanceError(
                        f"Timed out waiting for exclusive lock on receipt log '{log_path}'."
                    ) from exc
                time.sleep(0.005)
        try:
            yield
        finally:
            if locked:
                with contextlib.suppress(OSError):
                    handle.seek(0)
                    if sys.platform == "win32":
                        import msvcrt  # noqa: PLC0415

                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl  # noqa: PLC0415

                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        with contextlib.suppress(OSError):
            handle.close()


class ReceiptLogEntry(BaseModel):
    """Single JSONL record written by `ReceiptLogWriter`."""

    seq: int = Field(ge=0)
    receipt: ToolReceipt
    public_key_hex: str
    leaf_sha256: str
    chain_sha256: str
    merkle_root: str
    entry_signature: str = ""
    prev_chain_sha256: str | None = None
    aer1: dict[str, object] | None = None
    operation_key: str | None = None
    operation_input_sha256: str | None = None
    cached_rpc_result: dict[str, object] | None = None


@dataclass(frozen=True)
class ReceiptLogVerification:
    """Result of verifying an append-only `mcp-proxy` JSONL receipt log."""

    entry_count: int
    merkle_root: str
    head_chain_hash: str
    public_key_hex: str
    key_pinned: bool
    receipts: tuple[ToolReceipt, ...]
    session_id: str = ""
    prev_chain_sha256: str = _GENESIS_CHAIN_HEX


@dataclass(frozen=True)
class _PendingToolCall:
    call_id: str
    tool: str
    args: dict[str, str | int | float | bool]
    tool_def_sha256: str = ""


def _coerce_scalar(val: object) -> str | int | float | bool:
    if isinstance(val, bool) or isinstance(val, (str, int)):
        return val
    if isinstance(val, float):
        return val if math.isfinite(val) else str(val)
    try:
        return json.dumps(val, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return str(val)


def _rpc_id_key(raw_id: object) -> tuple[str, str] | None:
    if raw_id is None or isinstance(raw_id, bool):
        return None
    if isinstance(raw_id, int):
        return ("int", str(raw_id))
    if isinstance(raw_id, str):
        return ("str", raw_id)
    if isinstance(raw_id, float):
        return ("float", repr(raw_id)) if math.isfinite(raw_id) else None
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
        if kpath.is_symlink():
            raise ProvenanceError(f"Refusing symlinked key_file: {kpath}")
        if kpath.exists():
            seed = _decode_seed_bytes(kpath.read_bytes(), source=str(kpath))
        else:
            seed = os.urandom(32)
            kpath.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            try:
                fd = os.open(kpath, flags, 0o600)
                with os.fdopen(fd, "w", encoding="ascii") as fh:
                    fh.write(seed.hex() + "\n")
                with contextlib.suppress(OSError):
                    os.chmod(kpath, 0o600)
            except FileExistsError:
                if kpath.is_symlink():
                    raise ProvenanceError(f"Refusing symlinked key_file: {kpath}") from None
                seed = _decode_seed_bytes(kpath.read_bytes(), source=str(kpath))
    elif os.environ.get(_ENV_SIGNING_KEY_HEX):
        env_val = os.environ[_ENV_SIGNING_KEY_HEX].encode("ascii")
        seed = _decode_seed_bytes(env_val, source=_ENV_SIGNING_KEY_HEX)
    else:
        seed = os.urandom(32)

    priv = Ed25519PrivateKey.from_private_bytes(seed)
    signer = DomainSigner(Provenance.TOOL_VERIFIED, priv)

    if pubkey_out is not None:
        ppath = Path(pubkey_out)
        if ppath.is_symlink():
            raise ProvenanceError(f"Refusing symlinked pubkey_out: {ppath}")
        ppath.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "provenance": Provenance.TOOL_VERIFIED.value,
            "public_key_hex": signer.public_key_bytes.hex(),
        }
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{ppath.name}.", suffix=".tmp", dir=str(ppath.parent)
        )
        tmp_p = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(doc, indent=2, allow_nan=False) + "\n")
            os.replace(tmp_p, ppath)
        except Exception:
            with contextlib.suppress(OSError):
                tmp_p.unlink()
            raise

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


def _verify_embedded_aer1(aer1_doc: Mapping[str, object], receipt: ToolReceipt, seq: int) -> None:
    parsed = ToolReceipt.from_aer1(aer1_doc)
    expected_err = receipt.error_code or (
        "invalid_tool_arguments"
        if not receipt.executed
        else ("execution_error" if not receipt.ok else "")
    )
    if (
        parsed.tool != receipt.tool
        or parsed.call_id != receipt.call_id
        or parsed.ok != receipt.ok
        or parsed.executed != receipt.executed
        or parsed.error_code != expected_err
        or parsed.args != receipt.args
        or parsed.output != receipt.output
        or parsed.output_sha256 != receipt.output_sha256
        or parsed.session_id != receipt.session_id
    ):
        raise SignatureTamperError(
            f"Embedded AER-1 envelope disagrees with signed ToolReceipt at seq={seq}."
        )


class ReceiptLogWriter:
    """Append-only JSONL writer maintaining an Ed25519 + RFC 6962 Merkle chain on disk."""

    def __init__(
        self,
        log_path: Path | str,
        signer: DomainSigner,
        *,
        truncate: bool = False,
        include_aer1: bool = False,
        session_id: str | None = None,
        prev_log_path: Path | str | None = None,
        prev_pubkey_file: Path | str | None = None,
    ) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._signer = signer
        self._pubkey_hex = signer.public_key_bytes.hex()
        self._include_aer1 = include_aer1
        self._lock = threading.Lock()
        self._gate = ProvenanceGate({Provenance.TOOL_VERIFIED: signer.public_key_bytes})
        self._explicit_session_id = (
            session_id.strip() if session_id is not None and session_id.strip() else None
        )
        env_session = os.environ.get(_ENV_SESSION_ID, "").strip()
        self._session_id = (
            self._explicit_session_id or env_session or f"sess-{uuid4().hex[:12]}"
        )
        self._initial_chain_hash = _GENESIS_CHAIN_HEX
        if prev_log_path is not None:
            prev_ver = verify_receipt_log_file(prev_log_path, pubkey_file=prev_pubkey_file)
            if prev_ver.entry_count == 0:
                raise ProvenanceError("prev_log_path cannot be an empty receipt log.")
            self._initial_chain_hash = prev_ver.head_chain_hash

        self._merkle = MerkleReceiptLog(
            gate=self._gate,
            initial_chain_hash=self._initial_chain_hash,
        )
        self._seen_call_ids: set[str] = set()
        self._op_index: dict[str, ReceiptLogEntry] = {}
        self._id_index: dict[str, ReceiptLogEntry] = {}
        self._synced_lines = 0

        with _exclusive_log_lock(self.log_path):
            if truncate and self.log_path.exists():
                self.log_path.unlink()
            elif self.log_path.exists() and self.log_path.stat().st_size > 0:
                self._sync_from_disk_locked(prev_log_requested=prev_log_path is not None)

    @property
    def merkle_log(self) -> MerkleReceiptLog:
        return self._merkle

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def prev_chain_sha256(self) -> str:
        return self._initial_chain_hash

    def _sync_from_disk_locked(self, *, prev_log_requested: bool = False) -> None:
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            if self._synced_lines > 0:
                raise MerkleTamperError("Active receipt log was truncated or deleted externally.")
            return
        raw_lines = [
            ln for ln in self.log_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        if len(raw_lines) < self._synced_lines:
            raise MerkleTamperError("Active receipt log shrank on disk; tamper detected.")
        for idx in range(self._synced_lines, len(raw_lines)):
            entry = ReceiptLogEntry.model_validate_json(raw_lines[idx])
            if entry.seq != idx:
                raise MerkleTamperError(
                    f"Existing log sequence mismatch at line {idx}: "
                    f"expected {idx}, got {entry.seq}."
                )
            if entry.public_key_hex != self._pubkey_hex:
                raise ProvenanceError(
                    "Existing receipt log was signed by a different Ed25519 key. "
                    "Pass matching --key-file, --prev-log to a new log file, or --truncate."
                )
            if idx == 0:
                disk_prev = entry.prev_chain_sha256 or _GENESIS_CHAIN_HEX
                if prev_log_requested and disk_prev != self._initial_chain_hash:
                    raise MerkleTamperError(
                        "Existing log prev_chain_sha256 does not match supplied prev_log_path."
                    )
                if disk_prev != self._initial_chain_hash:
                    self._initial_chain_hash = disk_prev
                    self._merkle = MerkleReceiptLog(
                        gate=self._gate,
                        initial_chain_hash=self._initial_chain_hash,
                    )
                existing_sess = entry.receipt.session_id
                if self._explicit_session_id is not None:
                    if existing_sess != self._explicit_session_id:
                        raise ProvenanceError(
                            f"Existing receipt log session_id={existing_sess!r} conflicts with "
                            f"requested session_id={self._explicit_session_id!r}."
                        )
                else:
                    self._session_id = existing_sess
            else:
                if entry.prev_chain_sha256 is not None:
                    raise MerkleTamperError(
                        f"Unexpected prev_chain_sha256 on non-initial entry at seq={idx}."
                    )
                if entry.receipt.session_id != self._session_id:
                    raise ProvenanceError(
                        f"Session mismatch inside existing receipt log at seq={idx}: "
                        f"expected {self._session_id!r}, got {entry.receipt.session_id!r}."
                    )

            if entry.receipt.call_id in self._seen_call_ids:
                raise MerkleTamperError(
                    f"Duplicate receipt call_id {entry.receipt.call_id!r} at seq={idx}."
                )
            if entry.aer1 is not None:
                _verify_embedded_aer1(entry.aer1, entry.receipt, idx)
            self._gate.verify_log_entry(
                seq=idx,
                session_id=entry.receipt.session_id,
                leaf_sha256=entry.leaf_sha256,
                prev_chain_sha256=entry.prev_chain_sha256 or _GENESIS_CHAIN_HEX,
                chain_sha256=entry.chain_sha256,
                merkle_root=entry.merkle_root,
                entry_signature=entry.entry_signature,
            )
            chain_sha = self._merkle.append(entry.receipt)
            proof = self._merkle.inclusion_proof(idx)
            if (
                proof.leaf_sha256 != entry.leaf_sha256
                or chain_sha != entry.chain_sha256
                or proof.root_sha256 != entry.merkle_root
            ):
                raise MerkleTamperError(f"Existing receipt log failed Merkle check at seq={idx}.")
            self._seen_call_ids.add(entry.receipt.call_id)
            self._id_index[entry.receipt.call_id] = entry
            if isinstance(entry.aer1, Mapping) and isinstance(entry.aer1.get("id"), str):
                self._id_index[str(entry.aer1["id"])] = entry
            if entry.operation_key and entry.operation_key not in self._op_index:
                self._op_index[entry.operation_key] = entry
        self._synced_lines = len(raw_lines)

    def lookup_operation(self, operation_key: str) -> ReceiptLogEntry | None:
        """Look up a committed operation entry by durable `operation_key` (syncing disk first)."""
        key = operation_key.strip()
        if not key:
            return None
        with self._lock, _exclusive_log_lock(self.log_path):
            if self.log_path.exists() and self.log_path.stat().st_size > 0:
                self._sync_from_disk_locked()
            return self._op_index.get(key)

    def lookup_receipt_or_operation(self, identifier: str) -> ReceiptLogEntry | None:
        """Resolve a log entry by `operation_key`, `receipt.call_id`, or `aer1.id`."""
        ident = identifier.strip()
        if not ident:
            return None
        with self._lock, _exclusive_log_lock(self.log_path):
            if self.log_path.exists() and self.log_path.stat().st_size > 0:
                self._sync_from_disk_locked()
            return self._op_index.get(ident) or self._id_index.get(ident)

    def export_aer1_chain_v07(
        self, *, close: bool = True, published_at: str | None = None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Export this writer's verified `ToolReceipt` sequence as an AER-1 `-07` chain."""
        with self._lock:
            rcpts = list(self._merkle.receipts)
        return export_receipts_to_aer1_chain_v07(
            rcpts,
            job_id=self._session_id,
            close=close,
            published_at=published_at,
        )

    def append(
        self,
        receipt: ToolReceipt,
        *,
        operation_key: str | None = None,
        operation_input_sha256: str | None = None,
        cached_rpc_result: dict[str, object] | None = None,
    ) -> ReceiptLogEntry:
        """Sign `receipt`, update the Merkle log, and atomically flush one JSONL line to disk."""
        with self._lock, _exclusive_log_lock(self.log_path):
            self._sync_from_disk_locked()
            if receipt.session_id and receipt.session_id != self._session_id:
                raise ProvenanceError(
                    f"Receipt session_id={receipt.session_id!r} conflicts with writer "
                    f"session_id={self._session_id!r}."
                )
            if receipt.signature.startswith("ed25519:"):
                signed = receipt
            else:
                prepared = (
                    receipt
                    if receipt.session_id
                    else receipt.model_copy(update={"session_id": self._session_id})
                )
                signed = self._signer.sign_receipt(prepared)
            if signed.call_id in self._seen_call_ids:
                raise MerkleTamperError(
                    f"Duplicate receipt call_id {signed.call_id!r} in active receipt log."
                )
            seq = len(self._merkle)
            chain_sha = self._merkle.append(signed)
            self._seen_call_ids.add(signed.call_id)
            proof = self._merkle.inclusion_proof(seq)
            prev_anchor = (
                self._initial_chain_hash
                if seq == 0 and self._initial_chain_hash != _GENESIS_CHAIN_HEX
                else None
            )
            entry_sig = self._signer.sign_log_entry(
                seq=seq,
                session_id=signed.session_id,
                leaf_sha256=proof.leaf_sha256,
                prev_chain_sha256=prev_anchor or _GENESIS_CHAIN_HEX,
                chain_sha256=chain_sha,
                merkle_root=proof.root_sha256,
            )
            clean_op = operation_key.strip() if operation_key and operation_key.strip() else None
            entry = ReceiptLogEntry(
                seq=seq,
                receipt=signed,
                public_key_hex=self._pubkey_hex,
                leaf_sha256=proof.leaf_sha256,
                chain_sha256=chain_sha,
                merkle_root=proof.root_sha256,
                entry_signature=entry_sig,
                prev_chain_sha256=prev_anchor,
                aer1=signed.to_aer1() if self._include_aer1 else None,
                operation_key=clean_op,
                operation_input_sha256=operation_input_sha256 if clean_op else None,
                cached_rpc_result=cached_rpc_result if clean_op else None,
            )
            if clean_op and clean_op not in self._op_index:
                self._op_index[clean_op] = entry
            self._id_index[signed.call_id] = entry
            if isinstance(entry.aer1, Mapping) and isinstance(entry.aer1.get("id"), str):
                self._id_index[str(entry.aer1["id"])] = entry
            serialized = json.dumps(
                entry.model_dump(exclude_none=True),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(serialized + "\n")
                handle.flush()
                with contextlib.suppress(OSError):
                    os.fsync(handle.fileno())
            self._synced_lines += 1
            return entry


def verify_receipt_log_file(
    log_path: Path | str,
    *,
    expected_pubkey_hex: str | None = None,
    pubkey_file: Path | str | None = None,
    expected_session_id: str | None = None,
    prev_log_path: Path | str | None = None,
    prev_expected_pubkey_hex: str | None = None,
    prev_pubkey_file: Path | str | None = None,
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

    expected_prev_chain: str | None = None
    if prev_log_path is not None:
        prev_verified = verify_receipt_log_file(
            prev_log_path,
            expected_pubkey_hex=prev_expected_pubkey_hex,
            pubkey_file=prev_pubkey_file,
        )
        if prev_verified.entry_count == 0:
            raise MerkleTamperError("prev_log_path cannot be empty when verifying rotation link.")
        expected_prev_chain = prev_verified.head_chain_hash

    path = Path(log_path)
    raw_lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not raw_lines:
        if expected_prev_chain is not None:
            raise MerkleTamperError("Rotated receipt log is empty and lacks prev_chain_sha256.")
        if expected_session_id is not None:
            raise ProvenanceError(
                f"Receipt log is empty; expected session_id={expected_session_id!r}."
            )
        empty_log = MerkleReceiptLog()
        return ReceiptLogVerification(
            entry_count=0,
            merkle_root=empty_log.merkle_root(),
            head_chain_hash=empty_log.head_chain_hash,
            public_key_hex=pinned_hex or "",
            key_pinned=pinned_hex is not None,
            receipts=(),
            session_id="",
            prev_chain_sha256=_GENESIS_CHAIN_HEX,
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

    init_chain = entries[0].prev_chain_sha256 or _GENESIS_CHAIN_HEX
    if expected_prev_chain is not None and init_chain != expected_prev_chain:
        raise MerkleTamperError(
            f"Rotation chain link mismatch: log prev_chain_sha256={init_chain!r} "
            f"!= predecessor head_chain_hash={expected_prev_chain!r}."
        )

    gate = ProvenanceGate({Provenance.TOOL_VERIFIED: pub_bytes})
    merkle = MerkleReceiptLog(gate=gate, initial_chain_hash=init_chain)
    log_session_id = entries[0].receipt.session_id
    seen_call_ids: set[str] = set()

    for idx, entry in enumerate(entries):
        if entry.seq != idx:
            raise MerkleTamperError(
                f"Receipt sequence gap or reordering at index {idx}: found seq={entry.seq}."
            )
        if idx > 0 and entry.prev_chain_sha256 is not None:
            raise MerkleTamperError(
                f"Unexpected prev_chain_sha256 on non-initial entry at seq={idx}."
            )
        if entry.public_key_hex.strip().lower() != active_pub_hex:
            raise SignatureTamperError(
                f"Public key mismatch at seq={idx}: expected {active_pub_hex}, "
                f"got {entry.public_key_hex}."
            )
        if entry.receipt.session_id != log_session_id:
            raise ProvenanceError(
                f"Mixed session_id values inside receipt log at seq={idx}: "
                f"expected {log_session_id!r}, got {entry.receipt.session_id!r}."
            )
        if expected_session_id is not None and entry.receipt.session_id != expected_session_id:
            raise ProvenanceError(
                f"Session mismatch at seq={idx}: expected {expected_session_id!r}, "
                f"got {entry.receipt.session_id!r}."
            )
        if entry.receipt.call_id in seen_call_ids:
            raise MerkleTamperError(
                f"Duplicate receipt call_id {entry.receipt.call_id!r} at seq={idx}."
            )
        if entry.aer1 is not None:
            _verify_embedded_aer1(entry.aer1, entry.receipt, idx)
        gate.verify_log_entry(
            seq=idx,
            session_id=entry.receipt.session_id,
            leaf_sha256=entry.leaf_sha256,
            prev_chain_sha256=entry.prev_chain_sha256 or _GENESIS_CHAIN_HEX,
            chain_sha256=entry.chain_sha256,
            merkle_root=entry.merkle_root,
            entry_signature=entry.entry_signature,
        )
        chain_sha = merkle.append(entry.receipt)
        seen_call_ids.add(entry.receipt.call_id)
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
        session_id=log_session_id,
        prev_chain_sha256=init_chain,
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


_PROMPT_INJECTION_BANNER_RE: Final[re.Pattern[str]] = re.compile(
    r"(?im)^.*(?:"
    r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?|"
    r"system\s+override\s*:|"
    r"<\s*(?:system|developer|important_instruction)\s*>.*<\s*/\s*(?:system|developer|important_instruction)\s*>|"
    r"you\s+are\s+now\s+in\s+developer\s+mode"
    r").*$"
)
_WEB_SEARCH_TOOL_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:search|fetch|scrape|browse|crawl|web|firecrawl|tavily|brave|exa|serp)",
    re.IGNORECASE,
)


def compute_mcp_tool_sha256(tool_def: Mapping[str, Any]) -> str:
    """Compute canonical SHA-256 over `(name, description, inputSchema[, annotations])`."""
    canonical: dict[str, Any] = {
        "name": str(tool_def.get("name") or ""),
        "description": str(tool_def.get("description") or ""),
        "inputSchema": tool_def.get("inputSchema") or {},
    }
    if "annotations" in tool_def and tool_def["annotations"] is not None:
        canonical["annotations"] = tool_def["annotations"]
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(blob).hexdigest()


def sanitize_untrusted_web_text(raw_text: str) -> tuple[str, int]:
    """Strip indirect prompt-injection directives, anti-bot stubs, and syndicated wire copies.

    Returns `(sanitized_text, collapsed_or_stripped_count)`.
    """
    removed = 0

    def _strip_inj(match: re.Match[str]) -> str:
        nonlocal removed
        removed += 1
        return "[STRIPPED_UNTRUSTED_DIRECTIVE]"

    scrubbed = _PROMPT_INJECTION_BANNER_RE.sub(_strip_inj, raw_text)
    stripped = scrubbed.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        with contextlib.suppress(Exception):
            parsed = json.loads(stripped)
            items = (
                parsed
                if isinstance(parsed, list)
                else (parsed.get("results") if isinstance(parsed, Mapping) else None)
            )
            if isinstance(items, list) and items and all(isinstance(x, Mapping) for x in items):
                raw_tuples: list[tuple[str, str, str, str]] = []
                for idx, entry in enumerate(items):
                    url = str(entry.get("url") or entry.get("link") or f"source-{idx}.example")
                    snippet = str(
                        entry.get("snippet") or entry.get("content") or entry.get("text") or ""
                    )
                    title = str(entry.get("title") or "")
                    cleaned, is_stub = clean_scraped_passage(snippet, title=title)
                    if is_stub and not cleaned:
                        removed += 1
                        continue
                    dom = (
                        url.rsplit("//", maxsplit=1)[-1].split("/", maxsplit=1)[0].lower()
                        or f"src-{idx}"
                    )
                    raw_tuples.append((dom, url, f"{title} {cleaned}".strip(), cleaned))
                if raw_tuples:
                    clusters, collapsed = cluster_near_duplicate_sources(raw_tuples, "")
                    removed += collapsed
                    deduped_docs = [
                        {
                            "domain": cl.primary_domain,
                            "url": cl.primary_url,
                            "snippet": cl.primary_body_text,
                            "collapsed_syndicated_copies": cl.collapsed_duplicates,
                        }
                        for cl in clusters
                    ]
                    if isinstance(parsed, Mapping):
                        out_doc = dict(parsed)
                        out_doc["results"] = deduped_docs
                        out_doc["deduplicated_copies_collapsed"] = collapsed
                        return (
                            json.dumps(out_doc, sort_keys=True, separators=(",", ":")),
                            removed,
                        )
                    return (
                        json.dumps(deduped_docs, sort_keys=True, separators=(",", ":")),
                        removed,
                    )
    return (scrubbed, removed)


def filter_mcp_result_object(
    tool_name: str, response_obj: Mapping[str, object]
) -> tuple[dict[str, object], int]:
    """Filter `result.content` text blocks on web-search/fetch MCP tools."""
    if not _WEB_SEARCH_TOOL_RE.search(tool_name):
        return (dict(response_obj), 0)
    res = response_obj.get("result")
    if not isinstance(res, Mapping) or bool(res.get("isError", False)):
        return (dict(response_obj), 0)
    content = res.get("content")
    if not isinstance(content, list):
        return (dict(response_obj), 0)

    total_changes = 0
    new_content: list[object] = []
    for block in content:
        if isinstance(block, Mapping) and block.get("type") == "text" and isinstance(
            block.get("text"), str
        ):
            clean_txt, delta = sanitize_untrusted_web_text(block["text"])
            total_changes += delta
            new_blk = dict(block)
            new_blk["text"] = clean_txt
            new_content.append(new_blk)
        else:
            new_content.append(block)
    if total_changes == 0:
        return (dict(response_obj), 0)
    new_res = dict(res)
    new_res["content"] = new_content
    updated = dict(response_obj)
    updated["result"] = new_res
    return (updated, total_changes)


class McpStdioProxy:
    """Out-of-process stdio JSON-RPC 2.0 proxy recording signed `ToolReceipt`s."""

    def __init__(
        self,
        writer: ReceiptLogWriter,
        *,
        pin_tools: bool = False,
        raise_on_rugpull: bool = False,
        filter_web_results: bool = False,
        endpoint_id: str = "",
    ) -> None:
        self._writer = writer
        self._pin_tools = pin_tools
        self._raise_on_rugpull = raise_on_rugpull
        self._filter_web_results = filter_web_results
        self._endpoint_id = endpoint_id
        self._pending: dict[tuple[str, str], list[_PendingToolCall]] = {}
        self._pending_tools_list: set[tuple[str, str]] = set()
        self._seen_rpc_counts: dict[tuple[str, str], int] = {}
        self._tool_pins: dict[str, str] = {}
        self._compromised_tools: set[str] = set()
        self._observed_tools: dict[str, dict[str, Any]] = {}
        self._rugpull_count = 0
        self._lock = threading.Lock()

    @property
    def observed_tools(self) -> list[dict[str, Any]]:
        """Snapshot of MCP `tools/list` definitions observed by this proxy."""
        with self._lock:
            return [dict(v) for v in self._observed_tools.values()]

    @property
    def tool_pins(self) -> dict[str, str]:
        """Map of `tool_name -> canonical SHA-256` pinned from initial `tools/list`."""
        with self._lock:
            return dict(self._tool_pins)

    def process_client_line(self, raw_line: bytes) -> tuple[bytes, list[bytes]]:
        """Inspect a client->server JSON-RPC line.

        Returns `(bytes_to_forward_to_server, immediate_client_error_lines)`.
        When `pin_tools=True` and a tool's definition previously mutated mid-session,
        `tools/call` for that tool is refused pre-execution without forwarding upstream.
        """
        stripped = raw_line.strip()
        if not stripped or stripped[:1] not in (b"{", b"["):
            return (raw_line, [])
        try:
            payload = json.loads(stripped.decode("utf-8"))
        except Exception:
            return (raw_line, [])
        is_batch = isinstance(payload, list)
        items = payload if is_batch else [payload]
        fwd_items: list[object] = []
        refusal_lines: list[bytes] = []
        intercepted = False

        for item in items:
            if not isinstance(item, Mapping):
                fwd_items.append(item)
                continue
            method = item.get("method")
            id_key = _rpc_id_key(item.get("id"))
            if id_key is None:
                fwd_items.append(item)
                continue
            if method == "tools/list":
                with self._lock:
                    self._pending_tools_list.add(id_key)
                fwd_items.append(item)
                continue
            if method != "tools/call":
                fwd_items.append(item)
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
                occ = self._seen_rpc_counts.get(id_key, 0) + 1
                self._seen_rpc_counts[id_key] = occ
                cid = f"mcp-{id_key[1]}" if occ == 1 else f"mcp-{id_key[1]}#{occ}"
                pinned_sha = self._tool_pins.get(tool_name, "")
                is_compromised = self._pin_tools and tool_name in self._compromised_tools
                if not is_compromised:
                    self._pending.setdefault(id_key, []).append(
                        _PendingToolCall(
                            call_id=cid,
                            tool=tool_name,
                            args=args_map,
                            tool_def_sha256=pinned_sha,
                        )
                    )

            if is_compromised:
                intercepted = True
                self._writer.append(
                    ToolReceipt(
                        call_id=cid,
                        tool=tool_name,
                        ok=False,
                        executed=False,
                        error_code="mcp_tool_pin_refused",
                        args=args_map,
                        output=(
                            f"Refused tools/call for '{tool_name}': tool definition changed "
                            "mid-session from approved pin."
                        ),
                        tool_definition_sha256=pinned_sha,
                        endpoint_id=self._endpoint_id,
                    )
                )
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": item.get("id"),
                    "error": {
                        "code": -32603,
                        "message": (
                            f"MCP tool '{tool_name}' invocation refused: definition digest "
                            "differs from approved session pin."
                        ),
                    },
                }
                refusal_lines.append(
                    (json.dumps(err_resp, separators=(",", ":")) + "\n").encode("utf-8")
                )
            else:
                fwd_items.append(item)

        if not intercepted:
            return (raw_line, [])
        if not fwd_items:
            return (b"", refusal_lines)
        out_fwd = fwd_items if is_batch else fwd_items[0]
        return ((json.dumps(out_fwd, separators=(",", ":")) + "\n").encode("utf-8"), refusal_lines)

    def observe_client_line(self, raw_line: bytes) -> None:
        """Inspect a client->server JSON-RPC line and register `tools/call` or `tools/list`."""
        self.process_client_line(raw_line)

    def _inspect_tools_list_response(
        self, item: Mapping[str, object]
    ) -> list[tuple[str, str, str]]:
        """Record `tools/list` definitions and return `(name, old_sha, new_sha)` mutations."""
        res = item.get("result")
        if not isinstance(res, Mapping):
            return []
        tools = res.get("tools")
        if not isinstance(tools, list):
            return []
        mutations: list[tuple[str, str, str]] = []
        with self._lock:
            for tdef in tools:
                if not isinstance(tdef, Mapping):
                    continue
                raw_name = tdef.get("name")
                if not isinstance(raw_name, str) or not raw_name:
                    continue
                digest = compute_mcp_tool_sha256(tdef)
                prev_digest = self._tool_pins.get(raw_name)
                if prev_digest is None:
                    self._tool_pins[raw_name] = digest
                    self._observed_tools[raw_name] = dict(tdef)
                elif prev_digest != digest:
                    mutations.append((raw_name, prev_digest, digest))
                    if self._pin_tools:
                        self._compromised_tools.add(raw_name)
                    else:
                        self._tool_pins[raw_name] = digest
                        self._observed_tools[raw_name] = dict(tdef)
        for tname, old_h, new_h in mutations:
            with self._lock:
                self._rugpull_count += 1
                rp_seq = self._rugpull_count
            self._writer.append(
                ToolReceipt(
                    call_id=f"mcp-rugpull-{tname}-{rp_seq}-{uuid4().hex[:8]}",
                    tool=tname,
                    ok=False,
                    executed=False,
                    error_code="mcp_tool_rugpull",
                    args={"pinned_sha256": old_h, "observed_sha256": new_h},
                    output=(
                        f"MCP tools/list schema mutation detected for '{tname}': "
                        f"pinned={old_h[:16]}... != observed={new_h[:16]}..."
                    ),
                    tool_definition_sha256=old_h,
                    endpoint_id=self._endpoint_id,
                )
            )
            if self._raise_on_rugpull:
                raise ToolPinMismatchError(
                    f"MCP tool '{tname}' mutated mid-session: {old_h} -> {new_h}"
                )
        return mutations

    def process_server_line(self, raw_line: bytes) -> bytes:
        """Inspect and optionally filter/block a server->client JSON-RPC line."""
        stripped = raw_line.strip()
        if not stripped or stripped[:1] not in (b"{", b"["):
            return raw_line
        try:
            payload = json.loads(stripped.decode("utf-8"))
        except Exception:
            return raw_line
        is_batch = isinstance(payload, list)
        items = payload if is_batch else [payload]
        rewritten = False
        out_items: list[object] = []

        for item in items:
            if not isinstance(item, Mapping) or "method" in item:
                out_items.append(item)
                continue
            id_key = _rpc_id_key(item.get("id"))
            if id_key is None:
                out_items.append(item)
                continue

            is_tools_list = False
            with self._lock:
                if id_key in self._pending_tools_list:
                    self._pending_tools_list.discard(id_key)
                    is_tools_list = True
            if is_tools_list:
                mutations = self._inspect_tools_list_response(item)
                if mutations and self._pin_tools:
                    mut_names = ", ".join(m[0] for m in mutations)
                    err_item = {
                        "jsonrpc": "2.0",
                        "id": item.get("id"),
                        "error": {
                            "code": -32603,
                            "message": (
                                f"MCP tools/list pin mismatch (rug-pull blocked): {mut_names}"
                            ),
                        },
                    }
                    out_items.append(err_item)
                    rewritten = True
                else:
                    out_items.append(item)
                continue

            with self._lock:
                queue = self._pending.get(id_key)
                pending = queue.pop(0) if queue else None
                if queue is not None and not queue:
                    self._pending.pop(id_key, None)
            if pending is None:
                out_items.append(item)
                continue

            active_item: Mapping[str, object] = item
            if self._filter_web_results:
                filtered_item, changes = filter_mcp_result_object(pending.tool, item)
                if changes > 0:
                    active_item = filtered_item
                    rewritten = True

            receipt = self._build_receipt_from_response(pending, active_item)
            self._writer.append(receipt)
            out_items.append(active_item)

        if not rewritten:
            return raw_line
        out_payload = out_items if is_batch else out_items[0]
        return (json.dumps(out_payload, separators=(",", ":")) + "\n").encode("utf-8")

    def observe_server_line(self, raw_line: bytes) -> None:
        """Inspect a server->client JSON-RPC line and commit receipts for completed calls."""
        self.process_server_line(raw_line)

    def flush_abandoned_calls(self, exit_code: int) -> None:
        """Record `ok=False` receipts for in-flight calls if the MCP server exits early."""
        with self._lock:
            remaining = [p for q in self._pending.values() for p in q]
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
                    tool_definition_sha256=pending.tool_def_sha256,
                    endpoint_id=self._endpoint_id,
                )
            )

    def _build_receipt_from_response(
        self,
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
                tool_definition_sha256=pending.tool_def_sha256,
                endpoint_id=self._endpoint_id,
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
                tool_definition_sha256=pending.tool_def_sha256,
                endpoint_id=self._endpoint_id,
            )
        return ToolReceipt(
            call_id=pending.call_id,
            tool=pending.tool,
            ok=False,
            executed=not is_validation,
            error_code="invalid_tool_arguments" if is_validation else "mcp_tool_error",
            args=pending.args,
            output=output_text,
            tool_definition_sha256=pending.tool_def_sha256,
            endpoint_id=self._endpoint_id,
        )


def run_mcp_stdio_proxy(
    server_cmd: Sequence[str],
    *,
    log_path: Path | str,
    key_file: Path | str | None = None,
    pubkey_out: Path | str | None = None,
    truncate_log: bool = False,
    include_aer1: bool = False,
    pin_tools: bool = False,
    filter_web_results: bool = False,
    session_id: str | None = None,
    prev_log_path: Path | str | None = None,
    prev_pubkey_file: Path | str | None = None,
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
        session_id=session_id,
        prev_log_path=prev_log_path,
        prev_pubkey_file=prev_pubkey_file,
    )
    proxy = McpStdioProxy(
        writer,
        pin_tools=pin_tools,
        filter_web_results=filter_web_results,
        endpoint_id=f"stdio:{Path(server_cmd[0]).name}",
    )

    child_env = _scrub_child_env(os.environ)

    in_stream = client_in if client_in is not None else sys.stdin.buffer
    out_stream = client_out if client_out is not None else sys.stdout.buffer
    out_lock = threading.Lock()

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
                fwd_line, immediate_replies = proxy.process_client_line(line)
                if immediate_replies:
                    with out_lock:
                        for rep in immediate_replies:
                            out_stream.write(rep)
                        out_stream.flush()
                if fwd_line:
                    child_stdin.write(fwd_line)
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
            out_line = proxy.process_server_line(line)
            with out_lock:
                out_stream.write(out_line)
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
