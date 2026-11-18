"""Deterministic architecture and specification metrics for epistemic-guard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import epistemic_guard  # noqa: E402
from epistemic_guard.corroboration import POLICY_TABLE_2D  # noqa: E402
from epistemic_guard.mcp_server import _SUPPORTED_PROTOCOLS, MCP_TOOLS  # noqa: E402
from epistemic_guard.models import MODEL_SPECS  # noqa: E402
from epistemic_guard.schemas import (  # noqa: E402
    ELIGIBLE_PROVENANCE,
    PROOF_STANDARDS,
    Hazard,
    Provenance,
)


def main() -> None:
    src_files = sorted((ROOT / "src" / "epistemic_guard").glob("*.py"))
    test_files = sorted((ROOT / "tests").glob("test_*.py"))
    test_fn_count = sum(
        sum(1 for line in tf.read_text(encoding="utf-8").splitlines() if line.startswith("def test_"))
        for tf in test_files
    )

    per_model_mb = {k: round(spec.total_size_mb, 2) for k, spec in MODEL_SPECS.items()}
    total_int8_mb = round(sum(spec.total_size_mb for spec in MODEL_SPECS.values()), 2)
    pinned_file_count = sum(len(spec.files_sha256) for spec in MODEL_SPECS.values())

    payload = {
        "package_version": epistemic_guard.__version__,
        "source_module_count": len(src_files),
        "test_file_count": len(test_files),
        "test_function_count": test_fn_count,
        "import_linter_contracts": 1,
        "mcp_tool_count": len(MCP_TOOLS),
        "supported_mcp_protocols": len(_SUPPORTED_PROTOCOLS),
        "provenance_domain_count": len(Provenance),
        "hkdf_signer_count": len(ELIGIBLE_PROVENANCE),
        "eligible_write_provenance_count": len(ELIGIBLE_PROVENANCE),
        "hazard_tier_count": len(Hazard),
        "proof_standard_count": len(PROOF_STANDARDS),
        "policy_table_cells": len(POLICY_TABLE_2D),
        "corroboration_bucket_count": len(POLICY_TABLE_2D) // len(Hazard),
        "int8_model_count": len(MODEL_SPECS),
        "pinned_artifact_files": pinned_file_count,
        "files_per_model": pinned_file_count // len(MODEL_SPECS),
        "model_sizes_mb": per_model_mb,
        "total_int8_unpacked_mb": total_int8_mb,
        "ed25519_key_bytes": 32,
        "ed25519_signature_bytes": 64,
        "shingle_window_words": 5,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
