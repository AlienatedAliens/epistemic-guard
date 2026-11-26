"""Test helper: tells tests whether the INT8 models are installed (the neural path then differs)."""

from __future__ import annotations

from epistemic_guard.models import MODEL_SPECS, resolve_managed_model_dir


def _models_installed() -> bool:
    try:
        return all(
            resolve_managed_model_dir(k, verify_hashes=False, warn_on_missing=False) is not None
            for k in MODEL_SPECS
        )
    except Exception:
        return False
