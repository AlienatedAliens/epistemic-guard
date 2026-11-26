"""Test helper: some regression tests pin behaviour that only holds with the INT8 models installed."""

from __future__ import annotations

import pytest

from epistemic_guard.models import MODEL_SPECS, resolve_managed_model_dir


def _models_installed() -> bool:
    try:
        return all(
            resolve_managed_model_dir(k, verify_hashes=False, warn_on_missing=False) is not None
            for k in MODEL_SPECS
        )
    except Exception:
        return False


requires_models = pytest.mark.skipif(
    not _models_installed(),
    reason="needs the INT8 models (epistemic-guard download-models); the deterministic path does not yet cover this case",
)
