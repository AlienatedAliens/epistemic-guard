# Changelog

This file starts at 0.6.7. Earlier releases are described by their git tags and GitHub release notes.

## 0.6.8 (local release candidate)

Narrow follow-up to 0.6.7. No new performance or accuracy figures.

### Changed
- **Result classifier reads unknown-outcome shapes by value, not field name** (`result_status.py`,
  shared by both proxies and the transcript extractor). For a non-read-only tool, each of these is now
  `indeterminate` (previously `ok` or `failed`): a `status`/`outcome`/`state`/`result_status` value that
  reads unknown, indeterminate, uncertain or maybe-applied (for example `submission_uncertain`,
  `outcome_unknown`); an `error`/`error_code`/`code` string ending `_unknown` or `_uncertain` (for
  example `rent_outcome_unknown`), including when a `_failed` sibling is present; and a
  `side_effect_status` that reads unknown. `retry_safe: false` alone is not a signal. The same checks
  run on `structuredContent`, JSON text blocks and one `result`/`data` level. Read-only tools stay
  exempt, and honest `failed`/`ok` results are unchanged.
- **Deterministic claim rules no longer need the INT8 models for these wordings.** When the models
  are absent, claims are found by regex rules. 0.6.8 adds: first-person overwrite and destructive
  verbs on a file, table or database (`overwrote`, `replaced`, `clobbered`, `rewrote`, `truncated`,
  `dropped`, `reset`, `emptied`, `zeroed out`, `blew away`, `nuked`), including `I've`, "went ahead
  and" and terse lines such as "Replaced prod.db."; passive forms ("prod.db was overwritten", "the
  users table was dropped"); ticket-like creation claims ("I set up/logged/opened ticket T-1",
  "Ticket T-1 has been created"), so a `created: false` replay can no longer ground them; passive
  code-edit claims ("The `fail.py` file has been updated", "... updated and executed"); and
  "I changed/patched fail.py". Every rule needs a file, table, database or ticket-like object, and
  self-referential or third-party wording ("in my answer", "upstream", "last year") is excluded.
  The eleven tests that were skipped without the models now run without them. The neural path is
  unchanged.
  Limits: first-person and the listed passive wordings only; a claim with no file-like target
  ("I clobbered the config file") can only be contradicted by a receipt for a different named file,
  never by an unrelated write; some wordings still miss ("I swapped prod.db for an empty one",
  "I cleaned out the logs folder", "prod.db is now empty"). These misses are pinned as strict xfail
  in `tests/test_deterministic_paraphrases.py`; wording rules are open-ended, so they are stated as a
  limit, not chased. Written against development examples that were
  inspected and tuned on; no held-out measurement, so no rates are claimed.
- **Neural path: overwrite and destructive verbs are routed by the same rules.** With the models
  installed, the classifier's own label for "I overwrote prod.db" or "I truncated prod.db" was often
  a different family (a delete, an exec, a system-state change), so an honest claim grounded by a
  `write_file` or `truncate_file` receipt naming `prod.db` was flagged as having no matching receipt.
  `resolve_clause_claim_family` now applies the deterministic destructive/overwrite rules before
  falling back to the model's label, so both paths agree on these verbs. Idioms the model still
  treats as claims ("I dropped the idea", "I set up my notes") are not changed; the cause is the
  same (a first-person verb with no receipt), and they are recorded as baseline in
  `tests/test_deterministic_negatives.py`.
- Two idiom false positives of the new deterministic rules are fixed: an object noun followed by an
  abstract noun ("the repo idea") and a passive creation claim about someone else's earlier work
  ("was created yesterday by Dana").
- **HTTP proxy receipts carry an evidence tier.** `mcp-http-proxy` has no local probes, so every
  executed result is now signed `evidence_tier="server_reported"` (it was empty), and a call refused
  before execution (JSON-RPC -32700/-32600/-32601/-32602, or invalid arguments) is `attempted`. This
  matches the README, which already described remote HTTP tools as `server_reported`. It does not
  change any verdict: the supervisor reads only `independently_verified` and `probe_unobservable`.
- **`--pin-file PATH`** (`mcp-proxy` and `mcp-http-proxy`, or `pin_file=` in the Python API): keeps the
  first-seen SHA-256 of each tool definition on disk (trust on first use). A server whose definitions
  changed since an earlier connection takes the existing rug-pull path (`mcp_tool_rugpull` receipt;
  blocked with `--pin-tools`). Without the flag nothing changes. The file holds digests only, is
  written atomically, and loading fails closed on an unreadable, malformed or other-server file. Use
  one file per server; the stdio endpoint id is the executable name only. Not wired into `wrap-mcp-config`.
- The stdio MCP server reported `serverInfo.version` 0.5.2 and `server.json` (the MCP registry
  manifest) said 0.6.0; both now say 0.6.8, and a test fails if any release version drifts.
- The source distribution now includes `CHANGELOG.md` and `benchmarks/aer1_full_corpus.json`, so the
  test suite runs from an unpacked sdist.

### Not verified
- The classifier, evidence-tier, pin-file and deterministic-rule changes are covered by unit tests only. It has not been run against real MCP servers
  or any benchmark, and its false-positive rate on real traffic is unmeasured.
- The `_unknown`/`_uncertain` error-code and `side_effect_status` shapes follow a report of what real
  servers emit; they were not found by name in the public codebook of the 100-server study.

## 0.6.7

Covers everything on `main` after the v0.6.5 tag plus the `fix/claim-family-fallback` branch. No new
performance or accuracy figures: none of the changes below has been re-benchmarked on a held-out set.

0.6.7 shipped with a known limit for installs without the INT8 models; 0.6.8 closes it (see below).

### Added
- **Indeterminate tool outcomes.** A call whose effect is unknown (transport drop, EOF mid-call, HTTP
  502/504, or a result reporting an unknown or partly applied write) yields an indeterminate receipt.
  A reply that reports a definite failure is flagged `INDETERMINATE_OUTCOME_MISSTATED`; a reply that
  discloses the uncertainty passes with `verdict_status="unverifiable"`. Read-only tools are exempt.
  The MCP stdio/HTTP proxies and the transcript extractor share one classifier (`result_status.py`).
- **`unverifiable` for unconfirmable named targets.** When a file-write or system-state claim names a
  target and every matching write receipt names none, the turn is `unverifiable` instead of flagged.
  The tools are listed in `SupervisionVerdict.unconfirmed_target_tools`. A receipt that names a
  different target still contradicts; delete, external and exec claims stay strict.
- **`created: false` replay marker.** A successful result marked `created: false` or
  `replayed`/`idempotent_replay`/`deduplicated` true no longer grounds a claim that this turn created
  the object (it still counts as `ok`).
- **Passive code-edit claims** ("The `fail.py` file has been updated") resolve to `CLAIM_FILE_WRITE`;
  a neural-only `CLAIM_MEMORY` label on a clause with no memory vocabulary is treated as a generic
  state change.
- **OpenTelemetry:** receipt `evidence_tier`, narration `verdict_status` and a redacted `endpoint_id`
  (URL userinfo, query and fragment dropped) are exported as span attributes.
- **`--strip-untrusted-annotations`** (`mcp-proxy` and `mcp-http-proxy`): strips privilege-raising tool
  annotations (`readOnlyHint=true`, `destructiveHint=false`, `idempotentHint=true`,
  `openWorldHint=false`) from `tools/list` responses before they reach the client.
- **Provider tool-error parity:** OpenAI, Anthropic and Gemini transcripts are read through the same
  classifier, including each provider's `is_error` flag and structured JSON failures.

### Changed (decisions for review, see below)
- **Result classifier, `error` field (defect fix).** Previously any non-empty `error` meant `failed`.
  Rule now: `null`, `""`, `false`, `0`, `[]`, `{}` and placeholders such as `"none"`/`"null"` are no
  error. A non-empty `error` next to per-item statuses (`results`, `items`, `responses`,
  `operations` or `batch`) is decided by the items: all succeeded is `ok`, all failed is `failed`,
  a mix is `indeterminate`. A non-empty `error` next to `ok: true`, `success: true` or a success
  `status` with no per-item statuses is ambiguous: `indeterminate` for a mutating tool, `failed`
  for a read-only tool. A bare non-empty `error` is still `failed`.
- **Destructive-verb list (defect fix).** "I replaced the contents of prod.db" passed as verified when
  an unrelated write succeeded, because `replaced` was missing. Added: replaced, overwrote/overwritten,
  truncated, clobbered, reset, emptied. `truncate`, `empty` and `reset` tools now also ground delete
  claims. Existing: deleted, removed, dropped, wiped, purged, erased, cleared and others.
- **Retry supersession (decision).** An ok retry only supersedes an indeterminate call when it is in
  the same turn (`receipts`, not `prior_receipts`), after the indeterminate call, the same tool,
  `ok` and `executed`, and has identical args or the same idempotency key (`idempotency_key`,
  `Idempotency-Key`, `X-Idempotency-Key`). Otherwise the indeterminate call stays disclosed. Ordinary
  failed calls keep the older rule (any later ok call of the same tool name).

### Fixed
- README AER-1 claim: removed the `146/146 ... conformance checks` wording. That figure is this
  repository's own tally over a pinned local corpus, not the published kit's count.
- mypy error in `McpStdioProxy` (optional request id when dropping undelivered calls).

### Behaviour changes that can surprise existing users
- More turns can come out `contradicted` for destructive-sounding claims (replaced, overwrote,
  truncated, clobbered, reset, emptied) when no matching receipt exists.
- A result with `ok: true` plus a non-empty `error` on a mutating tool is now indeterminate, not failed;
  a reply that says it succeeded is flagged, a reply that discloses the uncertainty passes `unverifiable`.
- Batch results with a non-empty top-level `error` and all-successful items are no longer failures.
- A retry that changes the arguments no longer hides an earlier indeterminate call.
- More turns are `unverifiable` (not flagged) when a write receipt names no target.
