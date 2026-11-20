"""Finite-sample split-conformal risk calibration (`ConformalCalibrator`) for `ReceiptSupervisor`.

Provides distribution-free marginal coverage (`>= 1 - alpha`) over exchangeable agent turns
without retraining OpenVINO INT8 weights, supporting both:
- `target="fnr"`: bounds missed-hallucination (false-negative) rate `<= alpha` on ungrounded turns.
- `target="fpr"`: bounds false-retry alarm (false-positive) rate `<= alpha` on grounded turns.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from epistemic_guard.exceptions import InsufficientCalibrationError
from epistemic_guard.schemas import SupervisionVerdict, ToolReceipt
from epistemic_guard.supervision import ReceiptSupervisor

CalibrationTarget = Literal["fnr", "fpr"]


@dataclass(frozen=True)
class CalibrationSample:
    """Labeled calibration turn for `ConformalCalibrator`."""

    narration: str
    receipts: tuple[ToolReceipt, ...] = ()
    user_prompt: str = ""
    should_block: bool = False


@dataclass(frozen=True)
class ConformalDecision:
    """Conformalized verification decision wrapping `SupervisionVerdict` + calibrated risk."""

    should_block: bool
    risk_score: float
    threshold_q_hat: float
    alpha: float
    target: CalibrationTarget
    calibration_n: int
    verdict: SupervisionVerdict


class ConformalCalibrator:
    """Split-conformal calibrator using finite-sample quantile `ceil((n+1)*(1-alpha))/n`."""

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        target: CalibrationTarget = "fpr",
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must lie strictly in (0, 1); got {alpha!r}.")
        if target not in ("fnr", "fpr"):
            raise ValueError(f"target must be 'fnr' or 'fpr'; got {target!r}.")
        self.alpha = float(alpha)
        self.target: CalibrationTarget = target
        self._q_hat: float | None = None
        self._block_threshold: float | None = None
        self._n_cal: int = 0

    @property
    def is_calibrated(self) -> bool:
        return self._q_hat is not None

    @property
    def q_hat(self) -> float:
        if self._q_hat is None:
            raise InsufficientCalibrationError("ConformalCalibrator has not been calibrated yet.")
        return self._q_hat

    @property
    def block_threshold(self) -> float:
        if self._block_threshold is None:
            raise InsufficientCalibrationError("ConformalCalibrator has not been calibrated yet.")
        return self._block_threshold

    @property
    def calibration_size(self) -> int:
        return self._n_cal

    @staticmethod
    def finite_sample_quantile(scores: Sequence[float], alpha: float) -> float:
        """Compute split-conformal quantile at 1-based rank `ceil((n + 1) * (1 - alpha))`."""
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must lie strictly in (0, 1); got {alpha!r}.")
        n = len(scores)
        min_n = math.ceil((1.0 - alpha) / alpha)
        if n < max(1, min_n):
            raise InsufficientCalibrationError(
                f"Need at least {max(1, min_n)} calibration examples for alpha={alpha} "
                f"so ceil((n + 1) * (1 - alpha)) <= n; got n={n}."
            )
        rank_1based = math.ceil((n + 1) * (1.0 - alpha))
        if rank_1based > n:
            raise InsufficientCalibrationError(
                f"Rank {rank_1based} exceeds calibration sample size n={n} for alpha={alpha}."
            )
        ordered = sorted(float(s) for s in scores)
        return ordered[max(0, rank_1based - 1)]

    def fit_scores(self, nonconformity_scores: Sequence[float]) -> float:
        """Fit conformal quantile `q_hat` directly from precomputed nonconformity scores."""
        q_val = self.finite_sample_quantile(nonconformity_scores, self.alpha)
        self._q_hat = q_val
        self._n_cal = len(nonconformity_scores)
        if self.target == "fpr":
            self._block_threshold = q_val
        else:
            # For FNR control on positive (ungrounded) turns, nonconformity = 1.0 - risk_score,
            # so risk_score >= 1.0 - q_hat guarantees missed-hazard rate <= alpha.
            self._block_threshold = max(0.0, 1.0 - q_val)
        return self._block_threshold

    def calibrate(
        self,
        supervisor: ReceiptSupervisor,
        samples: Sequence[CalibrationSample],
    ) -> float:
        """Compute nonconformity scores on `samples` for `self.target` and fit `block_threshold`."""
        if self.target == "fpr":
            relevant = [s for s in samples if not s.should_block]
            scores = [
                supervisor.score_narration_risk(
                    s.narration, s.receipts, user_prompt=s.user_prompt
                )
                for s in relevant
            ]
        else:
            relevant = [s for s in samples if s.should_block]
            scores = [
                1.0
                - supervisor.score_narration_risk(
                    s.narration, s.receipts, user_prompt=s.user_prompt
                )
                for s in relevant
            ]
        return self.fit_scores(scores)

    def evaluate(
        self,
        supervisor: ReceiptSupervisor,
        narration: str,
        receipts: Sequence[ToolReceipt] = (),
        *,
        user_prompt: str = "",
    ) -> ConformalDecision:
        """Evaluate a turn with `supervisor` and apply the calibrated split-conformal threshold."""
        thresh = self.block_threshold
        q_val = self.q_hat
        risk = supervisor.score_narration_risk(narration, receipts, user_prompt=user_prompt)
        verdict = supervisor.verify_narration(narration, receipts, user_prompt=user_prompt)
        # Hard structural violations (e.g., forged SOAC tags or DEONTIC_CAPITULATION) have risk=1.0
        if self.target == "fpr":
            block = risk > thresh
        else:
            block = risk >= thresh and risk > 0.0
        return ConformalDecision(
            should_block=block,
            risk_score=round(risk, 4),
            threshold_q_hat=round(q_val, 4),
            alpha=self.alpha,
            target=self.target,
            calibration_n=self._n_cal,
            verdict=verdict,
        )
