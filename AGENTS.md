# AGENTS.md — Instructions for Coding Agents Integrating `epistemic-guard`

`epistemic-guard` (MIT License, Python >= 3.11) provides out-of-process cryptographic tool receipts (`HKDF-SHA256` + `Ed25519` + `RFC 6962` Merkle audit logs), zero-VRAM CPU/NPU output supervision (`OpenVINO INT8`), provenance-gated SQLite memory, and source-wire deduplication for MCP servers and LLM tool loops.

## 1. Install & Fetch Models

```bash
pip install epistemic-guard
epistemic-guard download-models
```

- Requires `0 MiB` GPU VRAM (runs on CPU or Intel NPU via OpenVINO INT8; `82.7 MB` `action_claim` supervisor, `279.5 MB` full 4-model bundle, SHA-256 verified before load).
- Cryptographic receipt signing (`ToolReceipt`, `mcp-proxy`, `mcp-http-proxy`, `SQLiteProvenanceStore`) works offline immediately without downloading OpenVINO weights.

## 2. Wrap Existing MCP Configs in One Command (Claude Desktop, Cursor, Windsurf, Cline)

To wrap every configured stdio MCP server in an out-of-process Ed25519 receipt proxy (scrubbing signing keys from child environments and pinning `tools/list` SHA-256 schemas against mid-session rug-pulls):

```bash
epistemic-guard wrap-config --input mcp_config.json --output mcp_config.wrapped.json --log receipts.jsonl --pin-tools
```

Or wrap a single stdio MCP server manually:

```json
{
  "mcpServers": {
    "filesystem-guarded": {
      "command": "epistemic-guard",
      "args": ["mcp-proxy", "--log", "receipts.jsonl", "--pin-tools", "--", "npx", "-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
    },
    "epistemic-verifier": {
      "command": "epistemic-guard",
      "args": ["mcp-server"]
    }
  }
}
```

For remote Streamable HTTP / SSE MCP servers (adds `Idempotency-Key` / `params._meta.operation_key` deduplication, `HTTP 409` / `-32602` parameter-mismatch refusal, and `GET /receipts/<id_or_key>` recovery):

```bash
epistemic-guard mcp-http-proxy --upstream http://127.0.0.1:8080/mcp --port 8081 --log receipts.jsonl --pin-tools
```

## 3. Verify Assistant Turns in Python (OpenAI, Anthropic, Gemini, LiteLLM, OpenTelemetry)

### Universal Chat History (`verify_chat_turn`)
Automatically extracts executed/failed tool calls from OpenAI (`tool_calls` / `role="tool"`), Anthropic (`tool_use` / `tool_result`), or Google Gemini (`functionCall` / `functionResponse`) message lists and checks final assistant prose against receipts:

```python
from epistemic_guard import verify_chat_turn

verdict = verify_chat_turn(
    messages=conversation_messages,
    assistant_text=final_reply,
    registered_mcp_tools=mcp_tools_list,  # optional: dynamic tools/list auto-binding
)
if not verdict.passed:
    # Block unverified completion claim or re-prompt with verdict.violations
    raise RuntimeError(f"Unverified action claim: {verdict.violations}")
```

### LiteLLM Post-Call Guardrail (`EpistemicGuardLiteLLM`)
```python
from epistemic_guard import EpistemicGuardLiteLLM

guard = EpistemicGuardLiteLLM(mode="raise")  # or mode="tag"
guard.check_completion(messages=messages, response=litellm_response)
```

### OpenTelemetry `gen_ai.*` Spans (`verify_otel_spans`)
```python
from epistemic_guard import verify_otel_spans

verdict = verify_otel_spans(spans=finished_spans, assistant_text=final_reply)
```

### Provenance-Gated Episodic Memory (`SQLiteProvenanceStore`)
Prevents multi-session memory self-poisoning by rejecting unsigned or `ok=False` writes at the SQLite trigger boundary:

```python
from epistemic_guard import SignerFactory, SQLiteProvenanceStore

factory = SignerFactory.from_master_key(master_key_32_bytes)
store = SQLiteProvenanceStore("agent_memory.sqlite", factory.public_key_registry())
signed_fact = factory.tool_verified().sign_dict({"fact": "deployed v1.4.2"})
store.insert_verified_fact("deploy-1", signed_fact)
```

## 4. Audit & Verify Receipt Logs Offline

```bash
epistemic-guard verify-log --log receipts.jsonl --pubkey receipts.jsonl.pub
```

## Boundaries & Honest Scope
- Pre-execution authorization/sandboxing decides whether a tool *may* run; `epistemic-guard` verifies *what actually ran on the wire* and blocks post-turn prose hallucinations (`ok=False` / `-32602` / zero-call fabrications) and unverified memory persistence.
- Optional export/verification for IETF `draft-zambo-aer1` envelopes is available via `--aer1` and `epistemic_guard.verify_aer1_receipt`, alongside native Ed25519 `ToolReceipt`s and OpenTelemetry `gen_ai.epistemic.*` span attributes.
