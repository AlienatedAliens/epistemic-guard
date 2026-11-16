# `epistemic-guard`

Signed tool receipts, out-of-process MCP receipt proxying, Ed25519 memory write-gating, and RAG source deduplication for autonomous agents.  
Runs locally on CPU, GPU, or NPU in **3–11 ms**. Zero LLM API calls. MIT License.

[PyPI](https://pypi.org/project/epistemic-guard/) · [Interactive Demo](https://huggingface.co/spaces/RollSolo/epistemic-guard) · [INT8 Weights](https://huggingface.co/RollSolo/epistemic-guard-int8) · [Held-Out Benchmarks (`N=973`)](https://huggingface.co/datasets/RollSolo/epistemic-guard-benchmarks)

---

## What It Fixes

1. **Phantom tool claims in assistant replies:** A tool threw an HTTP 500, failed schema validation before execution, or was never called, yet the model replies *"I refunded order #8821 and emailed the receipt."* Pre-execution firewalls do not catch this when no tool call occurred.
2. **Memory self-poisoning and prompt injection:** Writing raw assistant replies or untrusted web text into long-term memory (`vector_db.add(assistant_reply)`) lets hallucinations and injected pages persist across sessions.
3. **Syndicated RAG echo chambers:** Multiple retrieved URLs reprinting the same wire-service paragraph get counted as independent sources on medical, legal, or security queries.

---

## Install

```bash
pip install epistemic-guard

# Optional: download the 4-model OpenVINO INT8 pack (~227 MB download / ~279 MB unpacked, SHA-256 verified)
epistemic-guard download-models

# Run the local 4-part verification showcase
epistemic-guard demo
```

---

## 1. Out-of-Process MCP Receipt Proxy (`mcp-proxy` + `verify-log`)

Wrap any stdio MCP server (`Claude Desktop`, `Cursor`, `Cline`, `Windsurf`, or custom agents in any language). The Ed25519 signing key stays inside the `mcp-proxy` OS process; `EPISTEMIC_GUARD_MCP_KEY_HEX` is stripped before spawning the child MCP server. Every `tools/call` completion, tool runtime error (`isError: true`), JSON-RPC pre-execution rejection (`-32602`), and mid-call server crash is signed into an append-only RFC 6962 Merkle JSONL log.

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "epistemic-guard",
      "args": [
        "mcp-proxy",
        "--log", "/var/log/agent/receipts.jsonl",
        "--key-file", "/etc/agent-keys/mcp.key",
        "--pubkey-out", "/etc/agent-keys/mcp.pub.json",
        "--aer1",
        "--",
        "npx", "-y", "@modelcontextprotocol/server-filesystem", "/srv/workspace"
      ]
    }
  }
}
```

Verify the receipt log and check an assistant reply from any language or CI job:

```bash
# Verify Ed25519 signatures, sequential chain hashes, and Merkle roots (exit code 0 = valid, 2 = tampered)
epistemic-guard verify-log /var/log/agent/receipts.jsonl \
  --pubkey-file /etc/agent-keys/mcp.pub.json

# Also verify an assistant reply against those verified receipts (exit code 1 = ungrounded claim)
epistemic-guard verify-log /var/log/agent/receipts.jsonl \
  --pubkey-file /etc/agent-keys/mcp.pub.json \
  --narration "I saved the checklist to notes.txt; I also deleted staging.db." \
  --prompt "Save notes.txt and delete staging.db." \
  --json
```

**Security boundary note:** If the untrusted agent process has filesystem write access to `--log`, always pass `--pubkey-file` (stored outside the agent's writable directory) when running `verify-log` so an attacker cannot replace the entire log file under a newly generated keypair.

---

## 2. In-Process Python Supervision (`@recorder.tool` + `ReceiptSupervisor`)

```python
import os
from epistemic_guard import (
    MerkleReceiptLog,
    ProvenanceGate,
    ReceiptRecorder,
    ReceiptSupervisor,
    SignerFactory,
)

factory = SignerFactory.from_master_key(os.urandom(32))
gate = ProvenanceGate(factory.public_keys())
merkle = MerkleReceiptLog(gate=gate)
recorder = ReceiptRecorder(signer=factory.tool_verified(), merkle_log=merkle)


@recorder.tool  # Supports sync and async functions
def write_file(path: str, content: str) -> str:
    return f"wrote {len(content)} bytes"


write_file("notes.txt", "release checklist")

sup = ReceiptSupervisor(device="CPU")
verdict = sup.verify_narration(
    "I saved your checklist to notes.txt; I also deleted staging.db.",
    receipts=recorder.receipts,
    user_prompt="Save the checklist to notes.txt.",
)

assert verdict.is_valid is False
print(verdict.unsupported_spans[0])
# UnsupportedClaimSpan(text='I also deleted staging.db.', start_char=37, end_char=63, claimed_family='CLAIM_DELETE', ...)

# Correction prompt surfacing executed (ok=True), failed (ok=False), and pre-execution rejected calls:
print(verdict.to_retry_prompt(recorder.receipts))
```

- **Streaming SSE / voice pipelines:** `sup.stream(receipts=...)` checks each clause as `.`, `!`, `?`, `;`, or `\n` arrives so generation can stop before trailing tokens reach the user.
- **Router dispatch & AER-1 interop:** `recorder.dispatch(fn, raw_json_args)` records signed `executed=False, ok=False` receipts when JSON parsing or signature binding fails before entering the tool body. `ToolReceipt.from_aer1(...)` and `receipt.to_aer1()` import and export `AER-1` (`draft-zambo-aer1`) receipts with `canonical_bytes` SHA-256 verification.
- **Chat message lists:** `verify_chat_turn(messages)` checks OpenAI/Anthropic `tool_calls` + `role="tool"` histories directly.

---

## 3. Ed25519 Memory Write-Gating (`ProvenanceGate` + `SQLiteProvenanceStore`)

`ProvenanceGate` holds only Ed25519 public keys. Even if prompt injection compromises the agent loop, the gate holds no private key capable of forging `USER_EXPLICIT` or `TOOL_VERIFIED` records, and `LLM_NARRATION` is rejected by construction.

```python
import os
from epistemic_guard import Provenance, ProvenanceGate, SQLiteProvenanceStore, SignerFactory

factory = SignerFactory.from_master_key(os.urandom(32))
gate = ProvenanceGate(factory.public_keys())  # Public keys only
store = SQLiteProvenanceStore("agent_memory.sqlite", gate)  # Append-only SQLite triggers + hash chain

env = factory.user_explicit().sign("Deploy freeze starts Friday 18:00 UTC.", trace_id="turn-42")
store.append(env, expected_provenance=Provenance.USER_EXPLICIT)

assert len(store.list_verified()) == 1
assert store.verify_chain() == 1
store.close()
```

---

## 4. RAG Source Deduplication & Hazard Scaling (`corroborate_sources`)

```python
from epistemic_guard import corroborate_sources

wire_copy = (
    "Clinical guidance notes that initial adult warfarin dosing typically starts at "
    "2 mg to 5 mg once daily with INR monitoring."
)
burden = corroborate_sources(
    "What is the recommended starting adult warfarin dosage in mg?",
    [
        {"url": "https://reuters-health.org/a", "snippet": wire_copy},
        {"url": "https://clinical-digest.net/b", "snippet": wire_copy},
        {"url": "https://pharma-bulletin.io/c", "snippet": wire_copy},
    ],
    device="CPU",
)

assert burden.deduplicated_duplicates_collapsed == 2
assert burden.independent_sources_count == 1
assert burden.meets_casual_burden is False  # High-hazard query requires >=3 independent sources
```

---

## Measured CPU Performance & Held-Out Accuracy (OpenVINO INT8)

All four optional neural classifiers run via OpenVINO INT8 (`CPU`, `GPU`, or `NPU`, English-only) with pinned SHA-256 manifests. When weights are absent, `epistemic-guard` emits `EpistemicModelUnavailableWarning` (or raises `ModelIntegrityError` when `require_neural=True`) rather than degrading silently.

| Component | INT8 Size | CPU p50 Latency | Held-Out Result | Known Limitations |
| :--- | :---: | :---: | :--- | :--- |
| **`web_evidence_int8` + `nli_int8`** (`corroborate_sources`) | `23.1 MB` + `91.1 MB` | `3.0 ms` / `9.9 ms` | **`47/64 (73.4%)` burden accuracy & `56/64 (87.5%)` dispute accuracy** on pre-registered live-scraped web (`real_web_v2`, `246` archived `.html.gz` pages, 3 blinded LLM raters $\kappa = 0.91\text{–}1.00$); `60/64 (93.8%)` dispute accuracy on `real_web_v1`; `30/30 (100%)` wire/sybil collapse on `R-33`; `88/120 (73.3%)` on `nli_holdout_v2_120` | Implicit cross-article disagreements without lexical or single-sentence NLI opposition remain `17/64` (`26.6%`) of open-web cases. |
| **`action_claim_int8`** (`ReceiptSupervisor`) | `82.7 MB` | `10.9 ms` (clause) / `27.4 ms` (turn) | **`236/250 (94.4%)` across 3 held-out sets** (`91/100` live LLM completions `H100`, `95/100` `BLIND42_COH_100`, `50/50` `LOCKED_COH_50`); `297/300 (99.0%)` 10-class speech-act val (`99.7%` FP16 parity) | Code blocks, blockquotes, attributed dialogue, and creative/translation `user_prompt` contexts are masked automatically; unprompted first-person fiction (`user_prompt=""`) requires `user_prompt` or `[ACTION: verb target]` tags. |
| **`hazard_int8`** (`OpenVINOHazardClassifier`) | `82.7 MB` | `3.2 ms` | **`128/130 (98.5%)` safe tier assignment** on multi-clause `R-21` (2 blinded LLM raters); `44/50 (88.0%)` exact on `HZD_50` | Over-escalates ~12–15% of benign technical queries containing clinical or security nouns to higher proof tiers. |
| **`ProvenanceGate` (Ed25519)** | `0 MB` | `< 0.3 ms` | **`0/100 (0.0%)` natural self-poisoning** (vs `100/100` unguarded RAG and `11/100` natural / `100/100` adversarial under soft metadata tags) with **`0/200 (0.0%)` false rejections on valid writes** | Verifies cryptographic origin (`USER_EXPLICIT`, `TOOL_VERIFIED`), not whether an authenticated user stated a true fact. |

---

## Action-Receipt Benchmark (`N=250` Held-Out Turns)

Evaluated across `LOCKED_COH_50` (`50`), `BLIND42_COH_100` (`100`), and `Organic_Heldout_H100` (`100` live LLM completions; `0` training items; [`benchmarks/external_baselines_benchmark.json`](benchmarks/external_baselines_benchmark.json)) on `Core Ultra 9 275HX` + `RTX 5080 Laptop GPU`:

| System | Hardware & Memory | Accuracy (`[95% Wilson CI]`) | Recall / FPR | Turn p50 Latency | `Organic_Heldout_H100` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`epistemic-guard` (`action_claim_int8`)** | **CPU (`0 MiB` VRAM, `82.7 MB` RAM)** | **`236/250 (94.4% [90.8%, 96.6%])`** | `96.7%` / `7.8%` | **`27.4 ms`** | **`91/100 (91.0%)`** |
| **`Mistral-Nemo-Instruct-12B` (`Q5_K_M` Judge)** | GPU (`8,844 MiB` VRAM) | `217/250 (86.8% [82.0%, 90.4%])` | `87.7%` / `14.1%` | `77.9 ms` | `84/100 (84.0%)` |
| **`Microsoft Phi-4-14B` (`Q6_K` Judge)** | GPU (`12,214 MiB` VRAM) | `216/250 (86.4% [81.6%, 90.1%])` | **`100.0%`** / `26.6%` | `75.1 ms` | `79/100 (79.0%)` |
| **Off-the-shelf `DeBERTa-v3-xsmall` NLI** | CPU (`0 MiB` VRAM, `91.1 MB` RAM) | `132/250 (52.8% [46.6%, 58.9%])` | `24.6%` / `20.3%` | `27.5 ms` | `51/100 (51.0%)` |

- **Why off-the-shelf SNLI/MNLI (`52.8%`) misses phantom actions:** Standard NLI classifies `"Zero tools ran"` vs `"I saved checklist_backup.md"` as `NEUTRAL` rather than `CONTRADICTION` on ~75% of phantom action claims.
- **Tradeoff vs local LLM judges:** `Phi-4-14B` reached `100.0%` recall (`0` false negatives vs `4` for `epistemic-guard`) at the cost of `12.2 GiB` VRAM, `2.7x` latency, and a `26.6%` false-positive rate on offers and idioms. Cloud frontier judges (`GPT-4o`, `Claude`) were not benchmarked here.

---

## Ecosystem Comparison & Disclosures

- **Scope vs adjacent projects:**
  - **[NabaOS](https://github.com/nabaos/nabaos):** Full Rust agent runtime with OS sandboxing and Ed25519-signed static constitutions. `epistemic-guard` is a standalone Python package and MCP stdio proxy (`pip install epistemic-guard`) for existing stacks.
  - **Pre-execution firewalls (`NeMo Guardrails`, `LlamaFirewall`, `Guardrails AI`):** Gate inputs and tool calls *before* execution; `epistemic-guard` verifies post-execution assistant prose against signed `ToolReceipt` records.
  - **RAG faithfulness (`Vectara HHEM`, `Bespoke-MiniCheck`, `Patronus Lynx`):** Score summary-to-chunk entailment (`epistemic-guard` has not been benchmarked on RAG summarization leaderboards).
  - **Receipt formats (`AER-1` / `draft-zambo-aer1`):** Define portable hash-committed tool receipts; `epistemic-guard` signs receipts locally with Ed25519 (`mcp-proxy` or `@recorder.tool`), supports `AER-1` import/export, and checks assistant prose against those receipts.
- **Disclosures:** Neural models are English-only. Benchmarks ran on a single workstation; pre-registrations (`WO-46`) use git + OpenTimestamps SHA-256 locks ([dataset & `.ots` proofs](https://huggingface.co/datasets/RollSolo/epistemic-guard-benchmarks)), and $\kappa$ agreement figures come from blinded LLM raters rather than human crowdworkers.

---

## License

- **Source Code:** [MIT License](LICENSE) (`Copyright (c) 2026 Sean Maxwell` · [ORCID `0009-0004-0133-0463`](https://orcid.org/0009-0004-0133-0463)).
- **INT8 Model Weights:** Derived from `Apache-2.0` / `MIT` upstream checkpoints (`cross-encoder/nli-MiniLM2-L6-H768`, `cross-encoder/ms-marco-MiniLM-L-6-v2`, `cross-encoder/nli-deberta-v3-xsmall`); see [`MODEL_NOTICES.md`](MODEL_NOTICES.md).
- **Citation & Preprints:** [`CITATION.cff`](CITATION.cff) · Zenodo DOIs [`10.5281/zenodo.23649049`](https://doi.org/10.5281/zenodo.23649049), [`10.5281/zenodo.23901452`](https://doi.org/10.5281/zenodo.23901452), [`10.5281/zenodo.23901605`](https://doi.org/10.5281/zenodo.23901605), [`10.5281/zenodo.23901683`](https://doi.org/10.5281/zenodo.23901683).
