"""Append-only `MerkleReceiptLog` (RFC 6962 domain-separated SHA-256) and immutable
`SQLiteProvenanceStore` backed by SQLite WAL + Ed25519 `ProvenanceGate` verification.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from epistemic_guard.exceptions import MerkleTamperError, SignatureTamperError
from epistemic_guard.provenance import ProvenanceGate, canonical_receipt_bytes
from epistemic_guard.schemas import Provenance, SignedPayload, ToolReceipt

_LEAF_PREFIX: Final[bytes] = b"\x00"
_NODE_PREFIX: Final[bytes] = b"\x01"
_GENESIS_CHAIN: Final[str] = "0" * 64


def _leaf_hash(data: bytes) -> bytes:
    return hashlib.sha256(_LEAF_PREFIX + data).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(_NODE_PREFIX + left + right).digest()


@dataclass(frozen=True)
class MerkleInclusionProof:
    """RFC 6962 audit path proving a `ToolReceipt` belongs to a published Merkle root."""

    leaf_index: int
    tree_size: int
    leaf_sha256: str
    siblings: tuple[tuple[str, str], ...]  # (side: "L"|"R", hex_digest)
    root_sha256: str


class MerkleReceiptLog:
    """Tamper-evident append-only `ToolReceipt` log with hash chaining and Merkle proofs."""

    def __init__(
        self,
        gate: ProvenanceGate | None = None,
        *,
        initial_chain_hash: str | None = None,
    ) -> None:
        if initial_chain_hash is None:
            init_chain = _GENESIS_CHAIN
        else:
            norm = initial_chain_hash.strip().lower()
            if len(norm) != 64:
                raise MerkleTamperError("initial_chain_hash must be a 64-character hex SHA-256.")
            try:
                bytes.fromhex(norm)
            except ValueError as exc:
                raise MerkleTamperError("initial_chain_hash must be valid hexadecimal.") from exc
            init_chain = norm
        self._gate = gate
        self._initial_chain_hash = init_chain
        self._receipts: list[ToolReceipt] = []
        self._leaf_hashes: list[bytes] = []
        self._chain_hashes: list[str] = []

    def __len__(self) -> int:
        return len(self._receipts)

    @property
    def receipts(self) -> tuple[ToolReceipt, ...]:
        return tuple(self._receipts)

    @property
    def initial_chain_hash(self) -> str:
        return self._initial_chain_hash

    @property
    def head_chain_hash(self) -> str:
        return self._chain_hashes[-1] if self._chain_hashes else self._initial_chain_hash

    def append(self, receipt: ToolReceipt) -> str:
        """Append a receipt (verifying its Ed25519 signature if a `ProvenanceGate` is attached)."""
        if self._gate is not None:
            self._gate.verify_receipt(receipt)
        raw = canonical_receipt_bytes(receipt) + b"|" + receipt.signature.encode("utf-8")
        lh = _leaf_hash(raw)
        prev = self.head_chain_hash.encode("ascii")
        chain_digest = hashlib.sha256(prev + b":" + lh).hexdigest()
        self._receipts.append(receipt)
        self._leaf_hashes.append(lh)
        self._chain_hashes.append(chain_digest)
        return chain_digest

    def merkle_root(self) -> str:
        """Compute the RFC 6962 binary Merkle root over all recorded receipts."""
        if not self._leaf_hashes:
            return hashlib.sha256(b"").hexdigest()
        level = list(self._leaf_hashes)
        while len(level) > 1:
            nxt: list[bytes] = []
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    nxt.append(_node_hash(level[i], level[i + 1]))
                else:
                    nxt.append(level[i])
            level = nxt
        return level[0].hex()

    def inclusion_proof(self, index: int) -> MerkleInclusionProof:
        """Generate an `O(log N)` inclusion proof for `self.receipts[index]`."""
        n = len(self._leaf_hashes)
        if index < 0 or index >= n:
            raise IndexError(f"Receipt index {index} out of range for log of size {n}.")
        siblings: list[tuple[str, str]] = []
        idx = index
        level = list(self._leaf_hashes)
        while len(level) > 1:
            nxt: list[bytes] = []
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    if i == idx:
                        siblings.append(("R", level[i + 1].hex()))
                        idx = len(nxt)
                    elif i + 1 == idx:
                        siblings.append(("L", level[i].hex()))
                        idx = len(nxt)
                    nxt.append(_node_hash(level[i], level[i + 1]))
                else:
                    if i == idx:
                        idx = len(nxt)
                    nxt.append(level[i])
            level = nxt
        return MerkleInclusionProof(
            leaf_index=index,
            tree_size=n,
            leaf_sha256=self._leaf_hashes[index].hex(),
            siblings=tuple(siblings),
            root_sha256=level[0].hex(),
        )

    @staticmethod
    def _expected_sibling_sides(leaf_index: int, tree_size: int) -> tuple[str, ...]:
        sides: list[str] = []
        idx = leaf_index
        width = tree_size
        while width > 1:
            if idx % 2 == 0:
                if idx + 1 < width:
                    sides.append("R")
            else:
                sides.append("L")
            idx //= 2
            width = (width + 1) // 2
        return tuple(sides)

    @staticmethod
    def verify_inclusion_proof(receipt: ToolReceipt, proof: MerkleInclusionProof) -> bool:
        """Verify that `receipt` hashes to `proof.root_sha256` without needing the full log."""
        if proof.tree_size < 1 or proof.leaf_index < 0 or proof.leaf_index >= proof.tree_size:
            raise MerkleTamperError(
                f"Invalid Merkle proof bounds: leaf_index={proof.leaf_index}, "
                f"tree_size={proof.tree_size}."
            )
        expected_sides = MerkleReceiptLog._expected_sibling_sides(
            proof.leaf_index, proof.tree_size
        )
        if len(proof.siblings) != len(expected_sides):
            raise MerkleTamperError(
                f"Merkle sibling count {len(proof.siblings)} does not match expected "
                f"{len(expected_sides)} for leaf_index={proof.leaf_index}, "
                f"tree_size={proof.tree_size}."
            )
        raw = canonical_receipt_bytes(receipt) + b"|" + receipt.signature.encode("utf-8")
        cur = _leaf_hash(raw)
        if cur.hex() != proof.leaf_sha256:
            raise MerkleTamperError("Receipt bytes do not match proof leaf SHA-256.")
        for (side, hex_sib), exp_side in zip(proof.siblings, expected_sides, strict=True):
            if side not in ("L", "R") or side != exp_side:
                raise MerkleTamperError(
                    f"Merkle sibling side {side!r} does not match expected {exp_side!r} "
                    f"for leaf_index={proof.leaf_index}."
                )
            try:
                sib = bytes.fromhex(hex_sib)
            except ValueError as exc:
                raise MerkleTamperError("Merkle sibling is not valid hex.") from exc
            if len(sib) != 32 or hex_sib != sib.hex():
                raise MerkleTamperError(
                    "Merkle sibling must be a canonical 64-char lowercase SHA-256 hex digest."
                )
            cur = _node_hash(sib, cur) if side == "L" else _node_hash(cur, sib)
        if cur.hex() != proof.root_sha256:
            raise MerkleTamperError("Merkle inclusion path does not match expected root.")
        return True

    def verify_integrity(self) -> bool:
        """Re-compute all leaf, chain, and signature invariants across the entire log."""
        prev = self._initial_chain_hash.encode("ascii")
        for idx, rcpt in enumerate(self._receipts):
            if self._gate is not None:
                self._gate.verify_receipt(rcpt)
            raw = canonical_receipt_bytes(rcpt) + b"|" + rcpt.signature.encode("utf-8")
            lh = _leaf_hash(raw)
            if lh != self._leaf_hashes[idx]:
                raise MerkleTamperError(f"Leaf hash mismatch at receipt index {idx}.")
            expected_chain = hashlib.sha256(prev + b":" + lh).hexdigest()
            if expected_chain != self._chain_hashes[idx]:
                raise MerkleTamperError(f"Hash chain broken at receipt index {idx}.")
            prev = expected_chain.encode("ascii")
        return True


class SQLiteProvenanceStore:
    """Append-only SQLite memory store gated by `ProvenanceGate` on both write and read.

    Installs `BEFORE UPDATE` and `BEFORE DELETE` SQLite triggers so stored memory records
    cannot be mutated via SQL, and re-verifies Ed25519 signatures on every read to catch
    raw disk tampering.
    """

    def __init__(self, db_path: Path | str, gate: ProvenanceGate) -> None:
        self.db_path = str(db_path)
        self.gate = gate
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("PRAGMA trusted_schema=OFF;")
            if self.db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS provenance_envelopes (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    payload_id TEXT NOT NULL UNIQUE,
                    content TEXT NOT NULL,
                    provenance TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    trace_id TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    signature TEXT NOT NULL,
                    signer_domain TEXT NOT NULL,
                    chain_sha256 TEXT NOT NULL
                );
                """
            )
            self._conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS prevent_provenance_update
                BEFORE UPDATE ON provenance_envelopes
                BEGIN
                    SELECT RAISE(ABORT, 'provenance_envelopes is append-only (UPDATE forbidden)');
                END;
                """
            )
            self._conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS prevent_provenance_delete
                BEFORE DELETE ON provenance_envelopes
                BEGIN
                    SELECT RAISE(ABORT, 'provenance_envelopes is append-only (DELETE forbidden)');
                END;
                """
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def append(
        self,
        envelope: SignedPayload,
        *,
        expected_provenance: Provenance | None = None,
    ) -> str:
        """Verify `envelope` via `ProvenanceGate` and append it to the hash-chained SQLite log."""
        self.gate.verify(envelope, expected_provenance=expected_provenance)
        meta_json = json.dumps(
            envelope.metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
        )

        with self._lock:
            if not self._conn.in_transaction:
                self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT chain_sha256 FROM provenance_envelopes ORDER BY seq DESC LIMIT 1"
                ).fetchone()
                prev_chain = str(row["chain_sha256"]) if row is not None else _GENESIS_CHAIN
                chain_input = f"{prev_chain}:{envelope.payload_id}:{envelope.signature}".encode()
                chain_sha = hashlib.sha256(chain_input).hexdigest()
                self._conn.execute(
                    """
                    INSERT INTO provenance_envelopes (
                        payload_id, content, provenance, confidence, trace_id,
                        created_at_utc, metadata_json, signature, signer_domain, chain_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        envelope.payload_id,
                        envelope.content,
                        envelope.provenance.value,
                        float(envelope.confidence),
                        envelope.trace_id,
                        envelope.created_at_utc,
                        meta_json,
                        envelope.signature,
                        envelope.signer_domain,
                        chain_sha,
                    ),
                )
                self._conn.commit()
            except Exception:
                if self._conn.in_transaction:
                    self._conn.rollback()
                raise
        return chain_sha

    @staticmethod
    def _row_to_envelope(row: sqlite3.Row) -> SignedPayload:
        try:
            raw_meta = json.loads(str(row["metadata_json"]))
            if not isinstance(raw_meta, dict):
                raise ValueError("metadata_json must decode to an object")
            return SignedPayload(
                payload_id=str(row["payload_id"]),
                content=str(row["content"]),
                provenance=Provenance(str(row["provenance"])),
                confidence=float(row["confidence"]),
                trace_id=str(row["trace_id"]),
                created_at_utc=str(row["created_at_utc"]),
                metadata=raw_meta,
                signature=str(row["signature"]),
                signer_domain=str(row["signer_domain"]),
            )
        except Exception as exc:
            raise SignatureTamperError(
                "Corrupted provenance envelope row in SQLite store."
            ) from exc

    def list_verified(
        self,
        *,
        provenance: Provenance | None = None,
        limit: int = 100,
    ) -> list[SignedPayload]:
        """Read envelopes and re-verify every Ed25519 signature before returning."""
        with self._lock:
            if provenance is not None:
                rows = self._conn.execute(
                    "SELECT * FROM provenance_envelopes "
                    "WHERE provenance = ? ORDER BY seq ASC LIMIT ?",
                    (provenance.value, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM provenance_envelopes ORDER BY seq ASC LIMIT ?",
                    (limit,),
                ).fetchall()
        verified: list[SignedPayload] = []
        for r in rows:
            env = self._row_to_envelope(r)
            self.gate.verify(env)
            verified.append(env)
        return verified

    def verify_chain(self) -> int:
        """Verify immutability triggers, Ed25519 signatures, and sequential SHA-256 chain links."""
        with self._lock:
            trig_rows = self._conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'trigger' AND tbl_name = 'provenance_envelopes'"
            ).fetchall()
            rows = self._conn.execute(
                "SELECT * FROM provenance_envelopes ORDER BY seq ASC"
            ).fetchall()
        trig_names = {str(tr["name"]) for tr in trig_rows}
        if not {"prevent_provenance_update", "prevent_provenance_delete"}.issubset(trig_names):
            raise SignatureTamperError("SQLite immutability triggers missing from schema.")
        prev_chain = _GENESIS_CHAIN
        for r in rows:
            env = self._row_to_envelope(r)
            self.gate.verify(env)
            expected_chain = hashlib.sha256(
                f"{prev_chain}:{env.payload_id}:{env.signature}".encode()
            ).hexdigest()
            if str(r["chain_sha256"]) != expected_chain:
                raise SignatureTamperError(
                    f"SQLite provenance hash-chain mismatch at payload_id={env.payload_id}."
                )
            prev_chain = expected_chain
        return len(rows)
