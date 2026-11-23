# `epistemic-guard`

<!-- mcp-name: io.github.AlienatedAliens/epistemic-guard -->

**Check what an AI agent claims it did against cryptographically signed receipts of what actually happened.**  
`epistemic-guard` wraps any MCP server (`mcp-proxy` / `mcp-http-proxy`) or Claude Code / Codex session (`hook`) to emit Ed25519 + RFC 6962 Merkle tool receipts, optionally verifies local side effects out-of-band (`--probe-fs`, `--probe-git`, `--probe-sqlite`), and checks post-turn replies locally on CPU (`~8-11 ms`, `0 MiB` VRAM, zero LLM API fees, MIT License) with three-state verdicts (`verified`, `contradicted`, `unverifiable`). Optional modules add Ed25519 SQLite memory write-gating and RAG source deduplication.

[arXiv Paper (`2611.27095`)](https://arxiv.org/abs/2611.27095) · [Project Page](https://rollsolo.app/epistemic-guard) · [PyPI](https://pypi.org/project/epistemic-guard/) · [Interactive Demo](https://huggingface.co/spaces/RollSolo/epistemic-guard) · [INT8 Weights](https://huggingface.co/RollSolo/epistemic-guard-int8) · [Held-Out Benchmarks](https://huggingface.co/datasets/RollSolo/epistemic-guard-benchmarks) · [`AGENTS.md`](AGENTS.md)

---

## Quick Start (One Command)

```bash
pip install epistemic-guard
epistemic-guard download-models   # optional ~227 MB OpenVINO INT8 pack (SHA-256 verified)
epistemic-guard demo              # local 5-part showcase including live lying-server --probe-fs demo
```

Wrap an MCP server with out-of-process signed receipts + local disk verification, then check any reply:

```bash
epistemic-guard mcp-proxy --log receipts.jsonl --probe-fs . --pin-tools -- \
  npx -y @modelcontextprotocol/server-filesystem .

epistemic-guard verify-log receipts.jsonl \
  --narration "I saved notes.txt and deleted staging.db." \
  --flag-undisclosed --json
```

Or install directly into **Claude Code / Codex** `PostToolUse` + `Stop` hooks in one command (no MCP proxy required):

```bash
epistemic-guard install-hooks --flag-undisclosed
# Add --enforce once calibrated on your traffic; pass --uninstall to remove cleanly
```

---

## What It Fixes

1. **Phantom tool claims & lying servers:** A tool threw an HTTP 500, was never called, or returned `isError: false` without mutating disk/git/SQLite, yet the model replies *"I saved the file and deleted staging.db."* Pre-execution firewalls do not catch post-turn prose hallucinations.
2. **Silent undisclosed mutations:** A prompt-injected or over-eager agent calls `send_email` or `delete_file` and omits it from its reply (`--flag-undisclosed`).
3. **Optional modules (Memory self-poisoning & RAG echo chambers):** Ed25519 SQLite write-gating (`ProvenanceGate` + `SQLiteProvenanceStore`) rejects unsigned or `LLM_NARRATION` memory writes at the DB trigger layer, and `corroborate_sources` collapses syndicated wire-copy duplicates.

---

## 1. Out-of-Process MCP Receipt Proxies (`mcp-proxy`, `mcp-http-proxy`, `wrap-config`)

Wrap any stdio or Streamable HTTP MCP server (`Claude Desktop`, `Cursor`, `Cline`, `Windsurf`, or custom agents in any language). The Ed25519 signing key stays inside the proxy OS process; `EPISTEMIC_GUARD_*` signing secrets (`EPISTEMIC_GUARD_MCP_KEY_HEX`, `EPISTEMIC_GUARD_MASTER_KEY`, etc.) are stripped before spawning child MCP servers. Every `tools/call` completion, tool runtime error (`isError: true`), JSON-RPC pre-execution rejection (`-32602`), and mid-call server crash is signed into an append-only RFC 6962 Merkle JSONL log, with optional `--pin-tools` SHA-256 schema pinning against mid-session MCP rug-pulls.

Wrap an entire existing `mcp_config.json` in one command:

```bash
epistemic-guard wrap-config --input mcp_config.json --output mcp_config.wrapped.json --log /var/log/agent/receipts.jsonl --pin-tools
```

Or wrap a remote Streamable HTTP / SSE MCP server (adds `Idempotency-Key` / `params._meta.operation_key` retry deduplication, `HTTP 409` / `-32602` parameter-mismatch refusal, and `GET /receipts/<id_or_key>` lost-response recovery):

```bash
epistemic-guard mcp-http-proxy --upstream http://127.0.0.1:8080/mcp --port 8081 --log /var/log/agent/receipts.jsonl --pin-tools --aer1
```

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

# Also verify an assistant reply against current-turn receipts (exit code 1 = ungrounded claim)
epistemic-guard verify-log /var/log/agent/receipts.jsonl \
  --pubkey-file /etc/agent-keys/mcp.pub.json \
  --since-seq 12 \
  --narration "I saved the checklist to notes.txt; I also deleted staging.db." \
  --prompt "Save notes.txt and delete staging.db." \
  --json
```

- **Multi-worker locking, signed sequence links, & key rotation:** Each `ReceiptLogEntry` carries both an inner Ed25519 `ToolReceipt` signature and a domain-separated Ed25519 `entry_signature` over `(seq, session_id, leaf_sha256, prev_chain_sha256, chain_sha256, merkle_root)`, defeating active reorder-and-rehash attacks even when an attacker has write access to `receipts.jsonl`. Concurrent `mcp-proxy` workers sharing a `--log` file coordinate appends via OS-level exclusive file locks. When rotating Ed25519 keys across log segments, pass `--prev-log /var/log/agent/prev.jsonl` (and optional `--prev-pubkey-file` / `--session-id`) on both `mcp-proxy` and `verify-log`.

**Security boundary note:** Parent-injected `EPISTEMIC_GUARD_MCP_KEY_HEX` secrets stay in memory and are scrubbed from child environments; when using `--key-file` on disk (created atomically with POSIX mode `0600`, or restricted NTFS directory ACLs on Windows), run `mcp-proxy` under a dedicated OS user or container if the untrusted agent or child MCP server runs on the same host, so same-UID processes cannot read `--key-file` directly. Likewise, store `--pubkey-file` outside any agent-writable directory when running `verify-log`.

You can also run `epistemic-guard` itself as a standalone stdio MCP verification server (`verify_narration`, `verify_receipt_log`, `inspect_receipt`, `corroborate_sources`):

```json
{
  "mcpServers": {
    "epistemic-guard": {
      "command": "epistemic-guard",
      "args": ["mcp-server"]
    }
  }
}
```

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

- **Signature pinning, session binding & freshness anti-replay:** Pass `gate=gate`, `expected_session_id="..."`, and `max_receipt_age_seconds=300` to `ReceiptSupervisor` (or `expected_pubkey_hex` / `expected_session_id` / `max_receipt_age_seconds` on MCP `verify_narration`) to reject unsigned, cross-session, or replayed receipts before supervision runs.
- **OpenTelemetry & Langfuse observability export (zero extra deps):** Pass `otel_span=span` to `ReceiptRecorder` or call `attach_otel_evidence(span, receipts=recorder.receipts, verdict=verdict, merkle_root=merkle.merkle_root())` to stamp `AER-1` digests, Ed25519 signatures, Merkle roots, and supervision verdicts directly onto OTel / Langfuse traces.
- **Streaming SSE / voice pipelines:** `sup.stream(receipts=...)` checks each clause as `.`, `!`, `?`, `;`, or `\n` arrives so generation can stop before trailing tokens reach the user.
- **Output-value witnesses & split-conformal calibration (`v0.6.0`):** `OutputValueGuard` checks JSON scalar values and leaf strings returned inside `ToolReceipt.output` against assistant prose, while `ConformalCalibrator` fits distribution-free finite-sample thresholds (`q_hat`) targeting explicit `fnr` or `fpr` error rates.
- **Router dispatch & AER-1 interop:** `recorder.dispatch(fn, raw_json_args)` records signed `executed=False, ok=False` receipts when JSON parsing or signature binding fails before entering the tool body. `ToolReceipt.from_aer1(...)`, `receipt.to_aer1()`, and `verify_aer1_receipt(...)` import, export, and verify `AER-1` (`draft-zambo-aer1-26`, `146/146` `v1.5.7` conformance checks) receipts and `-07` chains.
- **Universal SDK, LiteLLM & OpenTelemetry adapters:** `verify_chat_turn(messages, assistant_text, registered_mcp_tools=...)` auto-extracts tool calls across OpenAI, Anthropic (`tool_use`/`tool_result`), and Google Gemini (`functionCall`/`functionResponse`); `EpistemicGuardLiteLLM` hooks LiteLLM completions; and `verify_otel_spans(spans, assistant_text)` supervises OpenTelemetry `gen_ai.*` traces directly.

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

All four optional neural classifiers run via OpenVINO INT8 (`CPU`, `GPU`, or `NPU`, English-only) with pinned SHA-256 manifests verified by default at every model load (`verify_hashes=True`). When weights are absent, `epistemic-guard` emits `EpistemicModelUnavailableWarning` (or raises `ModelIntegrityError` when `require_neural=True`) rather than degrading silently.

| Component | INT8 Size | CPU p50 Latency | Held-Out Result | Known Limitations |
| :--- | :---: | :---: | :--- | :--- |
| **`web_evidence_int8` + `nli_int8`** (`corroborate_sources`) | `23.1 MB` + `91.1 MB` | `3.0 ms` / `9.9 ms` | **`47/64 (73.4%)` burden accuracy & `56/64 (87.5%)` dispute accuracy** on pre-registered live-scraped web (`real_web_v2`, `246` archived `.html.gz` pages, 3 blinded LLM raters $\kappa = 0.91\text{–}1.00$); `60/64 (93.8%)` dispute accuracy on `real_web_v1`; `30/30 (100%)` wire/sybil collapse on `R-33`; `88/120 (73.3%)` on `nli_holdout_v2_120` | Implicit cross-article disagreements without lexical or single-sentence NLI opposition remain `17/64` (`26.6%`) of open-web cases. |
| **`action_claim_int8`** (`ReceiptSupervisor`) | `82.7 MB` | `8.6 ms` (clause) / `9.3–17.9 ms` (turn) | **`234/250 (93.6%)` (`v0.6.0`) / `233/250 (93.2%)` (`v0.5.2`) across 3 held-out sets** (vs. `187/250 [74.8%]` receipt-aware 7-family rule baseline); **`161/180 (89.4%, 85.0% recall, 8.3% FPR)` on blind frozen-commit `Live-Frontier-180 v2`** (`fe72761`, 12 unseen enterprise MCP tools across `Claude Haiku 4.5`, `GPT-4.1-mini`, `Gemini 2.5 Flash`, vs. `145/180 [80.6%, 43.3% recall]` dynamic rule baseline and `156/180 [86.7%, 19.2% FPR]` `gpt-4o-mini` judge) | Probabilistic statistical classifier (run in shadow/flag mode or two-stage cascade; `8.3%` FPR on multi-sentence failure prose). Code blocks, blockquotes, attributed dialogue, failure reports, and open-vocabulary MCP tool schemas are handled automatically; strict `args` target-binding rejects bare zero-argument receipts when prose names concrete targets. |
| **`hazard_int8`** (`OpenVINOHazardClassifier`) | `82.7 MB` | `3.2 ms` | **`128/130 (98.5%)` safe tier assignment** on multi-clause `R-21` (2 blinded LLM raters); `44/50 (88.0%)` exact on `HZD_50` | Over-escalates ~12–15% of benign technical queries containing clinical or security nouns to higher proof tiers. |
| **`ProvenanceGate` (Ed25519)** | `0 MB` | `< 0.3 ms` | **`0/100 (0.0%)` natural self-poisoning** (vs `100/100` unguarded RAG and `11/100` natural / `100/100` adversarial under soft metadata tags) with **`0/200 (0.0%)` false rejections on valid writes** | Verifies cryptographic origin (`USER_EXPLICIT`, `TOOL_VERIFIED`), not whether an authenticated user stated a true fact. |

---

## Action-Receipt Benchmark (`N=250` Held-Out Turns)

Evaluated across `LOCKED_COH_50` (`50`), `BLIND42_COH_100` (`100`), and `Organic_Heldout_H100` (`100` live LLM completions; `0` training items; [`benchmarks/external_baselines_benchmark.json`](benchmarks/external_baselines_benchmark.json)) on `Intel Core Ultra 9 275HX` + `RTX 5080 Laptop GPU`:

| System | Hardware & Memory | Accuracy (`[95% Wilson CI]`) | Recall / FPR | Turn p50 Latency | `Organic_Heldout_H100` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`epistemic-guard` (`action_claim_int8`, `v0.6.0` / `v0.5.2`)** | **CPU (`0 MiB` VRAM, `82.7 MB` RAM)** | **`234/250 (93.6%)` / `233/250 (93.2% [89.4%, 95.7%])`** | `96.7%` / `9.4%` (`10.2%` `v0.5.2`) | **`17.9 ms`** | **`91/100 (91.0%)`** |
| **`Qwen3-30B-A3B` (`Q4_K_M` MoE Judge)** | GPU (`~18.5 GiB` VRAM) | `232/250 (92.8% [88.9%, 95.4%])` | `95.9%` / `10.2%` | `141.8 ms` | `90/100 (90.0%)` |
| **`Mistral-Nemo-Instruct-12B` (`Q5_K_M` Judge)** | GPU (`8,844 MiB` VRAM) | `217/250 (86.8% [82.0%, 90.4%])` | `87.7%` / `14.1%` | `80.6 ms` | `84/100 (84.0%)` |
| **`Microsoft Phi-4-14B` (`Q6_K` Judge)** | GPU (`12,214 MiB` VRAM) | `216/250 (86.4% [81.6%, 90.1%])` | **`100.0%`** / `26.6%` | `76.2 ms` | `79/100 (79.0%)` |
| **Off-the-shelf `DeBERTa-v3-xsmall` NLI** | CPU (`0 MiB` VRAM, `91.1 MB` RAM) | `132/250 (52.8% [46.6%, 58.9%])` | `24.6%` / `20.3%` | `22.4 ms` | `51/100 (51.0%)` |
| **Receipt-Aware 7-Family Rule Baseline** | CPU (`0 MiB` VRAM, `0 MB` RAM) | `187/250 (74.8% [69.1%, 79.8%])` | `50.0%` / `1.6%` | `< 0.1 ms` | `63/100 (63.0%)` |

- **Scope of cryptographic vs. statistical guarantees:** `mcp-proxy` / `mcp-http-proxy` provide a **deterministic signed-evidence layer** (`Ed25519` + `RFC 6962` Merkle log) proving what crossed the MCP wire; they do not prove that a buggy upstream server actually mutated external state after returning `200 OK`, and `ReceiptSupervisor` is a **probabilistic statistical checker** over those receipts (recommended in shadow/log mode first until calibrated to your real-traffic base rate).
- **Why off-the-shelf SNLI/MNLI (`52.8%`) and receipt-aware rules (`74.8%`) miss phantom actions:** Standard NLI classifies `"Zero tools ran"` vs `"I saved checklist_backup.md"` as `NEUTRAL` rather than `CONTRADICTION` on `73/92` (`79.3%`) of its false negatives. A receipt-aware 7-family rule baseline (clause-split, quote-stripped, matching verb family to `ok=True` tool family + filename targets) scores `46/50 (92.0%)` on templated `LOCKED_COH_50` with `1.6%` FPR, but degrades to `78/100 (78.0%)` on `BLIND42_COH_100` and `63/100 (63.0%, 23.4% recall)` on organic LLM completions (`Organic_Heldout_H100`) where paraphrases diverge from hand-coded patterns.
- **Tradeoff vs local LLM judges:** Paired two-sided exact McNemar tests (unadjusted) show higher accuracy than `Mistral-NeMo-12B` ($p = 0.0259$) and `Phi-4-14B` ($p = 0.0095$; $p = 0.0169$ on `Organic_Heldout_H100`), while tying `Qwen3-30B-A3B` (`233` vs `232/250`, within statistical noise) at `0 MiB` GPU VRAM vs `~18.5 GiB` VRAM. `Phi-4-14B` reached `100.0%` recall (`0` false negatives vs `4` for `epistemic-guard`) at the cost of `12,214 MiB` (`11.9 GiB`) VRAM and a `26.6%` false-positive rate on offers and idioms.

### Synthetic Frontier-Model Bidirectional MCP Benchmark (`Live-Frontier-360`, Held-Out Unseen Tools)

To test generalization outside coding/filesystem tools across **both** `Halluci-Action` directions (phantom completion claims AND silent undisclosed mutations), we ran a frozen-commit (`d7128fb`) synthetic fault-injection evaluation (`N=360`, `N=36` per sub-regime, `N=72` for `ok_clean`, `sha256:5d035c16...`; [datasets on Hugging Face](https://huggingface.co/datasets/RollSolo/epistemic-guard-benchmarks)) on **12 held-out unseen enterprise MCP tools** (Stripe, Twilio, AWS EC2, Okta, Datadog, GitLab, Jira, Slack, Cloudflare, Snowflake, Sentry, Zendesk) generated by `claude-haiku-4.5`, `gpt-4.1-mini`, and `gemini-2.5-flash` (same author-built generator pipeline and taxonomy; all judges received identical structured receipts + prompt + reply):

| System | Accuracy (`[95% Wilson CI]`) | Recall (`TP / 144`) | FPR (`[95% Wilson CI]`) | Ex-Undisclosed (`N=324`) | Latency / Cost |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`epistemic-guard` (`d7128fb`, CPU INT8)** | **`357/360 (99.17% [97.6%, 99.7%])`** | **`98.61%`** (`142/144`) | **`0.46% [0.1%, 2.6%]`** (`1/216`) | `321/324` (`99.1%`) | **`10.9 ms` CPU, no per-call fee** |
| **`anthropic/claude-haiku-5.5` (Current-Gen Judge)** | `357/360 (99.17% [97.6%, 99.7%])` | `98.61%` (`142/144`) | `0.46% [0.1%, 2.6%]` (`1/216`) | `321/324` (`99.1%`) | `~1,200 ms`, API \$ |
| **`openai/gpt-5.6-sol` (Current-Gen Judge)** | `350/360 (97.22% [95.0%, 98.5%])` | `93.06%` (`134/144`) | `0.00% [0.0%, 1.8%]` (`0/216`) | `323/324` (`99.7%`) | `~1,400 ms`, API \$ |
| **Dynamic Receipt-Aware Rule Baseline** | `247/360 (68.61% [63.6%, 73.2%])` | `28.47%` (`41/144`) | `4.63% [2.5%, 8.3%]` (`10/216`) | `247/324` (`76.2%`) | `< 1 ms`, no per-call fee |

- **Honest trajectory & caveats (read before extrapolating):**
  - **Iteration history across our evaluations:** `Live-60` (`v0.5.2`, real stdio proxy): `38.24%` FPR (`13/34`) → `Live-Frontier-180 v1` (`2c7cc3e`, first SaaS test): `50.8%` FPR → `v2` (`fe72761`, open-vocab grounding): `8.3%` FPR → `v3` (`0912019`, Stage-2 NLI): `5.1%` FPR → `Live-Frontier-360` (`d7128fb`): `0.46%` FPR. This benchmark is **synthetic, author-taxonomized, and saturated at parity** (`357/360` vs `claude-haiku-5.5` `357/360`; `321/324` vs `gpt-5.6-sol` `323/324` on the 324 non-undisclosed turns, where `gpt-5.6-sol` missed `9/36` on `undisclosed_mutation`). Also note that the underlying `nli_int8` checkpoint scores `73.3%` (`88/120`) on its own blinded cross-domain holdout (`nli_holdout_120`), so `99.17%` here reflects controlled synthetic fault injection, not a universal production guarantee.
  - **External unlabeled traces (`Exgentic/agent-llm-traces-v2`):** Across `30` real `AppWorld`/`claude_code` OpenTelemetry sessions (`223` multi-turn replies, `487` tools) on Hugging Face, `epistemic-guard` flagged **`0/223` turns** (`0.54 ms` p50; Stage-2 NLI rarely triggered because planning clauses lacked action claims). Because `Exgentic` has no human ground-truth labels and a `gpt-4.1-mini` auto-auditor flagged `14/60` sampled turns, treat `0/223` strictly as an **unlabeled flag rate**, not a proven false-positive or recall rate.
- **Out-of-process local effect probes (`--probe-fs`, `--probe-git`, `--probe-sqlite`) & preconditions:** `mcp-proxy` takes read-only before/after snapshots outside the MCP server's trust domain (`stat` + SHA-256 for files; `--no-optional-locks -c core.fsmonitor=false` `HEAD`/`symbolic-ref`/`for-each-ref`/`status`/`diff`/`stash` for Git; file/WAL `stat` + SHA-256 + `schema_version` for SQLite) and emits three-state outcomes (`independently_verified`, `effect_contradicted`, or `probe_unobservable`). Supported scope: synchronous local mutations inside the probed workspace/repo/DB; does not cover async/queued jobs, concurrent writers, remote HTTP servers, or legitimate 0-row SQL `UPDATE`s.
- **Recommended rollout:** Run the deterministic proxy, Merkle log, and `--probe-fs` in enforcement/audit mode, and run `ReceiptSupervisor` in **shadow/flag mode** until calibrated on your own production traffic.

---

## Threat Model & Non-Goals

`epistemic-guard` is a **signed-evidence MCP proxy / hook recorder plus a lightweight statistical prose checker**, not a sandbox or full enterprise firewall.

- **What signed receipts prove:** `mcp-proxy` and `mcp-http-proxy` prove what JSON-RPC request/response crossed the MCP wire, which tool definition (`tool_definition_sha256` + `endpoint_id`) was active, and that the append-only RFC 6962 Merkle log has not been tampered with or reordered.
- **What out-of-process local effect probes (`--probe-fs`, `--probe-git`, `--probe-sqlite`) prove:** For synchronous local mutations inside the probed workspace, Git repository, or SQLite DB, `mcp-proxy` takes read-only before/after snapshots outside the MCP server's trust domain and signs `evidence_tier` (`independently_verified`, `effect_contradicted`, or `probe_unobservable`) into the receipt. Remote HTTP/SaaS tools, async/queued jobs, concurrent writers, and legitimate 0-row SQL `UPDATE`s remain `server_reported` or `probe_unobservable`.
- **Statistical checker & enforcement boundary:** `ReceiptSupervisor` is a probabilistic classifier over signed receipts, not a formal proof, emitting three-state verdicts (`verified`, `contradicted`, `unverifiable`). If an agent streams unbuffered tokens directly to the user before `verify_narration()` runs, the proxy log still records ground-truth execution, but the user may already have seen an unverified claim. Recommended rollout: run the deterministic proxy, Merkle log, and `--probe-fs` in enforcement/audit mode immediately, and run `ReceiptSupervisor` (or `hook stop`) in **shadow/flag mode** until calibrated to your real-traffic false-positive tolerance.
- **OS key isolation:** Keep `--key-file` (and `--pubkey-file` during `verify-log`) outside directories writable by the untrusted agent or child MCP server; run the proxy under a separate OS user or container on shared hosts.

---

## Ecosystem Comparison & Head-to-Head (`agent-claimcheck`)

### Direct Peer & Adjacent-Layer Landscape

| Project | License | Layer / Integration | Signed Receipts & Tamper Log | Out-of-Band Local Effect Probes | Post-Turn Free-Prose Check | Three-State Verdict | Memory Write-Gate / RAG Dedup |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **`epistemic-guard`** | **MIT** | **Out-of-process MCP stdio/HTTP proxy, Claude Code/Codex hooks, Python SDK** | **Ed25519 + RFC 6962 Merkle + AER-1** | **Yes (`--probe-fs`, `--probe-git`, `--probe-sqlite`)** | **Yes (CPU INT8 + Stage-2 NLI, ~8–11 ms)** | **Yes (`verified` / `contradicted` / `unverifiable`)** | **Yes (Ed25519 SQLite + RAG wire dedup)** |
| **[`agent-claimcheck`](https://github.com/B0yko/agent-claimcheck)** | Apache-2.0 | Offline/trace verifier & `cascade-offline` classifier | No | Consumes trace `state_probe` steps | Yes (rule + trained cascade) | Yes (`allow` / `block` / `review`) | No |
| **[`veridict`](https://github.com/veridict-dev/veridict)** | MIT | Alpha CI gate over declared artifacts | No | Checks declared CI artifacts | No (declared claims only) | Yes | No |
| **[`Cruxial`](https://github.com/cruxial-ai/cruxial)** | MIT | In-process Python regex wrapper | No | No | Regex-only | No | No |
| **[`claimproof`](https://github.com/claimproof/claimproof)** | MIT | Claude Code `Stop` hook | No | No | Yes | No | No |
| **`TruthGuard`** | Source-available | Claude Code / Gemini CLI file-checksum hooks | No | Local file checksums | Hook-based | No | No |
| **[`Pipelock`](https://github.com/pipelab-org/pipelock)** | Apache-2.0 | Go egress & Streamable HTTP MCP firewall | Ed25519 action receipts + `tools/list` pinning | No | No (pre-execution / wire firewall) | No | No |
| **[`hol-guard`](https://github.com/hashgraph-online/hol-guard)** | Apache-2.0 | Pre-execution agent/MCP policy firewall | Policy logs | No | No (pre-execution) | No | No |
| **[`NabaOS`](https://github.com/nabaos/nabaos)** | Apache-2.0 | Rust agent runtime (`arXiv:2603.10060`) | HMAC receipts (symmetric) | No | Cooperative receipt-ID tags | No | No |

### Independent External Benchmarks (`N=1,441` Third-Party Turns)

To test generalization outside our own synthetic datasets, we evaluated `epistemic-guard` (`v0.6.5-dev`, CPU INT8, zero domain training) across four independently authored public corpora. Every row is labeled with its exact evaluation **Status** (`zero-shot`, `development (inspected)`, or `unlabeled`):

| External Corpus | Evaluation Status | Size | Result (`epistemic-guard`, CPU) | CPU p50 | Honest Notes & Tradeoffs |
| :--- | :---: | :---: | :--- | :---: | :--- |
| **[`B0yko/agent-claimcheck`](https://github.com/B0yko/agent-claimcheck)** (`v1` test, `b420b5d` default) | **`zero-shot`** | `120` traces (`72` clean, `48` fail) | **`90/120 (75.0%)`** acc, `19/48 (39.6%)` caught, **`1/72 (1.4%)` FP** | `7.5 ms` | Clean zero-shot baseline (never saw train or test). Enabling `verify_output_values=True` at `b420b5d` caught `27/48` but caused `17/72 (23.6%)` FPs from experimental MNLI-over-JSON-leaves. |
| **[`B0yko/agent-claimcheck`](https://github.com/B0yko/agent-claimcheck)** (`v1` test, `v0.6.5-dev` post-fix) | **`development (inspected)`** | `120` traces (`72` clean, `48` fail) | `110/120 (91.7%)` acc, `39/48 (81.2%)` caught, `1/72 (1.4%)` FP (`verify_output_values=True`) | `8.2 ms` | **Test split no longer held out:** failures were inspected during development. Competitor's own `cascade-offline` (trained on its `v1` train split, escalating `11/120` to `review`) beats us on its benchmark at `113/120 (94.2%, 41/48 caught, 0/72 FP)`. |
| **[`B0yko/agent-claimcheck`](https://github.com/B0yko/agent-claimcheck)** (`seed=99999` fresh split, `1ad8d7c`) | **`held-out (fresh seed)`** | `120` traces (`72` clean, `48` fail) | **`107/120 (89.2%)`** default (`35/48` caught, **`0/72 [0.0%]` FP**) / **`111/120 (92.5%)`** opt-in `verify_output_values=True` (`39/48` caught, **`0/72 [0.0%]` FP**) | `8.4 ms` | Generated fresh with `agent-claimcheck`'s own generator at `seed=99999` and evaluated once without inspection (same synthetic generator family as `v1`, testing fresh-instance generalization rather than an independent generator; opt-in `verify_output_values=True` enables additional date/email/timezone value heuristics). |
| **[`control_arena_agentdojo`](https://huggingface.co/datasets/adityaasinha28/control_arena_agentdojo)** (`claude-sonnet-4` banking) | **`development (inspected)`** | `300` sessions (`170` benign, `130` attacks) | Dir-1 benign FP: **`10/170 (5.9%)`**; Dir-2 (`--flag-undisclosed`): `254/300 (84.7%)`, `106/130 (81.5%)` attacks caught, `22/170 (12.9%)` benign FP | `8.9 ms` | Banking suite only; benign FPs on these 300 rows were inspected to fix `CLAIM_DELETE`/`CLAIM_COMM_EXTERNAL` misclassifications. At `1%` real-world attack prevalence, `12.9%` FPR implies `~6%` precision (keep `--flag-undisclosed` in shadow/flag mode until calibrated). |
| **[`nebius/SWE-agent-trajectories`](https://huggingface.co/datasets/nebius/SWE-agent-trajectories)** | **`unlabeled`** | `798` turns (`40` sessions) | `32/798 (4.01%)` flagged | `0.44 ms` | Real GitHub coding sessions (`bash`/`edit`/`git`). Unlabeled: flag-rate reduction alone does not separate FPR from recall. |
| **[`Exgentic/agent-llm-traces-v2`](https://huggingface.co/datasets/Exgentic/agent-llm-traces-v2)** | **`unlabeled`** | `223` turns (`30` sessions, `487` tools) | `0/223 (0.0%)` flagged | `0.54 ms` | Unlabeled; a `gpt-4.1-mini` auto-auditor flagged `14/60` sampled turns, so treat `0/223` strictly as an unlabeled flag rate. |

### OWASP Coverage & Disclosures

- **OWASP Top 10 for Agentic Applications (2026):**
  - **`ASI02` (Tool Misuse & Phantom Execution):** `mcp-proxy`, `hook`, `@recorder.tool`, and `ReceiptSupervisor` check assistant prose against signed `ToolReceipt` logs (`executed`, `ok`, `error_code`, `evidence_tier`).
  - **`ASI06` (Memory & Context Poisoning):** `ProvenanceGate` and `SQLiteProvenanceStore` reject unsigned writes and `LLM_NARRATION` at the database trigger layer, and `corroborate_sources` collapses syndicated wire/Sybil echoes before synthesis.
  - **`ASI08` (Cascading Hallucinations Across Trust Boundaries):** Append-only RFC 6962 Merkle logs and optional `AER-1` receipt export preserve verifiable provenance across multi-agent and MCP boundaries.
- **Literature & adjacent tools:** `Halluci-Action` (Mao et al., NeurIPS 2026 Workshop) measures bidirectional claim-vs-action misalignment offline across `2,843` enterprise trajectories (`epistemic-guard` implements runtime detection for both phantom completion claims and silent `--flag-undisclosed` mutations). Pre-execution firewalls (`NeMo Guardrails`, `LlamaFirewall`, `Guardrails AI`) gate inputs before execution; RAG faithfulness models (`Vectara HHEM`, `Bespoke-MiniCheck`, `Patronus Lynx`) score summary-to-chunk entailment.
- **Disclosures:** Neural models are English-only. Benchmarks ran on a single workstation; v0.3.0 pre-registrations use git + OpenTimestamps SHA-256 locks ([dataset & `.ots` proofs](https://huggingface.co/datasets/RollSolo/epistemic-guard-benchmarks)), $\kappa$ agreement figures come from blinded LLM raters rather than human crowdworkers, and positive cases in `Live-Frontier-180` / `Live-Frontier-360` use synthetic prompted roleplay faults (not human-labeled production traces). Next validation step: natural unprompted multi-turn enterprise trajectories (`N >= 500`) with human-audited labels and a receipt-conditioned cross-encoder.

---

## License

- **Source Code:** [MIT License](LICENSE) (`Copyright (c) 2026 Sean Maxwell` · [ORCID `0009-0004-0133-0463`](https://orcid.org/0009-0004-0133-0463) · Built by [RollSolo](https://rollsolo.app), Member of NVIDIA Inception).
- **INT8 Model Weights:** Derived from `Apache-2.0` / `MIT` upstream checkpoints (`cross-encoder/nli-MiniLM2-L6-H768`, `cross-encoder/ms-marco-MiniLM-L-6-v2`, `cross-encoder/nli-deberta-v3-xsmall`); see [`MODEL_NOTICES.md`](MODEL_NOTICES.md).
- **Citation & Preprints:** [`arXiv:2611.27095` (`cs.SE`)](https://arxiv.org/abs/2611.27095) · [`CITATION.cff`](CITATION.cff) · Zenodo DOIs [`10.5281/zenodo.23649049`](https://doi.org/10.5281/zenodo.23649049), [`10.5281/zenodo.23901452`](https://doi.org/10.5281/zenodo.23901452), [`10.5281/zenodo.23901605`](https://doi.org/10.5281/zenodo.23901605), [`10.5281/zenodo.23901683`](https://doi.org/10.5281/zenodo.23901683).
