"""Deterministic summary of the N=250 held-out action-receipt supervision benchmark."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    bench_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benchmarks/external_baselines_benchmark.json")
    raw = json.loads(bench_path.read_text(encoding="utf-8"))
    systems = raw["systems"]

    eg = systems["epistemic_guard_v0_3_0_cpu_int8"]
    deb = systems["off_the_shelf_deberta_v3_xsmall_nli_cpu"]
    phi = systems["llm_judge_microsoft_phi_4_14b_q6_k_gpu"]
    nemo = systems["llm_judge_mistral_nemo_12b_q5_k_m_gpu"]

    positive_cases = eg["tp"] + eg["fn"]
    negative_cases = eg["tn"] + eg["fp"]

    summary = {
        "n_total": raw["n_total"],
        "n_positive_ungrounded": positive_cases,
        "n_negative_grounded_or_benign": negative_cases,
        "split_counts": {
            "locked_coh_50": eg["by_split"]["LOCKED_COH_50"]["total"],
            "blind42_coh_100": eg["by_split"]["BLIND42_COH_100"]["total"],
            "organic_heldout_h100": eg["by_split"]["Organic_Heldout_H100"]["total"],
        },
        "epistemic_guard_int8_cpu": {
            "correct": eg["correct"],
            "accuracy_pct": eg["accuracy_pct"],
            "ci95_low": eg["accuracy_wilson_95ci"][0],
            "ci95_high": eg["accuracy_wilson_95ci"][1],
            "tp": eg["tp"],
            "tn": eg["tn"],
            "fp": eg["fp"],
            "fn": eg["fn"],
            "recall_pct": eg["recall_pct"],
            "fpr_pct": eg["fpr_pct"],
            "latency_p50_ms": eg["latency_p50_ms"],
            "latency_mean_ms": eg["latency_mean_ms"],
            "latency_p95_ms": eg["latency_p95_ms"],
            "gpu_vram_mib": eg["gpu_vram_mib"],
            "locked_coh_50_correct": eg["by_split"]["LOCKED_COH_50"]["correct"],
            "locked_coh_50_accuracy_pct": eg["by_split"]["LOCKED_COH_50"]["accuracy_pct"],
            "blind42_coh_100_correct": eg["by_split"]["BLIND42_COH_100"]["correct"],
            "blind42_coh_100_accuracy_pct": eg["by_split"]["BLIND42_COH_100"]["accuracy_pct"],
            "organic_h100_correct": eg["by_split"]["Organic_Heldout_H100"]["correct"],
            "organic_h100_accuracy_pct": eg["by_split"]["Organic_Heldout_H100"]["accuracy_pct"],
        },
        "mistral_nemo_12b_gpu": {
            "correct": nemo["correct"],
            "accuracy_pct": nemo["accuracy_pct"],
            "ci95_low": nemo["accuracy_wilson_95ci"][0],
            "ci95_high": nemo["accuracy_wilson_95ci"][1],
            "recall_pct": nemo["recall_pct"],
            "fpr_pct": nemo["fpr_pct"],
            "latency_p50_ms": nemo["latency_p50_ms"],
            "latency_p95_ms": nemo["latency_p95_ms"],
            "gpu_vram_mib": nemo["gpu_vram_mib"],
            "organic_h100_correct": nemo["by_split"]["Organic_Heldout_H100"]["correct"],
            "organic_h100_accuracy_pct": nemo["by_split"]["Organic_Heldout_H100"]["accuracy_pct"],
        },
        "phi_4_14b_gpu": {
            "correct": phi["correct"],
            "accuracy_pct": phi["accuracy_pct"],
            "ci95_low": phi["accuracy_wilson_95ci"][0],
            "ci95_high": phi["accuracy_wilson_95ci"][1],
            "recall_pct": phi["recall_pct"],
            "fpr_pct": phi["fpr_pct"],
            "latency_p50_ms": phi["latency_p50_ms"],
            "latency_p95_ms": phi["latency_p95_ms"],
            "gpu_vram_mib": phi["gpu_vram_mib"],
            "organic_h100_correct": phi["by_split"]["Organic_Heldout_H100"]["correct"],
            "organic_h100_accuracy_pct": phi["by_split"]["Organic_Heldout_H100"]["accuracy_pct"],
        },
        "deberta_v3_xsmall_nli_cpu": {
            "correct": deb["correct"],
            "accuracy_pct": deb["accuracy_pct"],
            "ci95_low": deb["accuracy_wilson_95ci"][0],
            "ci95_high": deb["accuracy_wilson_95ci"][1],
            "recall_pct": deb["recall_pct"],
            "fpr_pct": deb["fpr_pct"],
            "latency_p50_ms": deb["latency_p50_ms"],
            "latency_p95_ms": deb["latency_p95_ms"],
            "gpu_vram_mib": deb["gpu_vram_mib"],
            "organic_h100_correct": deb["by_split"]["Organic_Heldout_H100"]["correct"],
            "organic_h100_accuracy_pct": deb["by_split"]["Organic_Heldout_H100"]["accuracy_pct"],
        },
        "comparisons": {
            "delta_accuracy_vs_deberta_pp": round(eg["accuracy_pct"] - deb["accuracy_pct"], 1),
            "delta_accuracy_vs_nemo_pp": round(eg["accuracy_pct"] - nemo["accuracy_pct"], 1),
            "delta_accuracy_vs_phi4_pp": round(eg["accuracy_pct"] - phi["accuracy_pct"], 1),
            "speedup_p50_vs_phi4_x": round(phi["latency_p50_ms"] / eg["latency_p50_ms"], 2),
            "speedup_p50_vs_nemo_x": round(nemo["latency_p50_ms"] / eg["latency_p50_ms"], 2),
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
