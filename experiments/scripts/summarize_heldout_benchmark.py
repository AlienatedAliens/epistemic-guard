"""Deterministic summary of the N=250 held-out action-receipt supervision benchmark."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    bench_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("benchmarks/external_baselines_benchmark.json")
    )
    raw = json.loads(bench_path.read_text(encoding="utf-8"))
    systems = raw["systems"]

    eg = systems["epistemic_guard_cpu_int8"]
    deb = systems["off_the_shelf_deberta_v3_xsmall_nli_cpu"]
    phi = systems["llm_judge_microsoft_phi_4_14b_q6_k_gpu"]
    nemo = systems["llm_judge_mistral_nemo_12b_q5_k_m_gpu"]
    mcn = raw["paired_mcnemar"]
    abl = raw["ablations"]
    det_abl = abl["deterministic_regex_with_receipts_cpu"]
    s1_abl = abl["stage1_neural_encoder_without_receipts_cpu"]
    prov = raw["dataset_provenance"]
    e2e = raw["live_mcp_e2e_60"]
    e2e_sys = e2e["systems"]
    e2e_eg = e2e_sys["epistemic_guard_cpu_int8"]
    e2e_s1 = e2e_sys["stage1_neural_without_receipts_cpu"]
    e2e_det = e2e_sys["deterministic_regex_with_receipts_cpu"]
    e2e_deb = e2e_sys["deberta_v3_xsmall_nli_cpu"]
    e2e_nemo = e2e_sys["mistral_nemo_12b_q5_k_m_gpu"]
    e2e_phi = e2e_sys["phi_4_14b_q6_k_gpu"]
    e2e_mcn = e2e["paired_mcnemar"]

    e2e_path = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else Path("benchmarks/live_mcp_e2e_60.jsonl")
    )
    e2e_rows = {
        r["id"]: r
        for r in (json.loads(ln) for ln in e2e_path.read_text(encoding="utf-8").splitlines() if ln.strip())
    }
    err_ids = e2e["epistemic_guard_error_ids"]
    fp_ok_clean_unmapped = sum(
        1 for eid in err_ids if not e2e_rows[eid]["gold_should_block"] and e2e_rows[eid]["regime"] == "ok_clean"
    )
    fp_failed_turn_clauses = sum(
        1 for eid in err_ids if not e2e_rows[eid]["gold_should_block"] and e2e_rows[eid]["regime"] != "ok_clean"
    )

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
            "single_clause_p50_ms": eg["single_clause_p50_ms"],
            "cold_start_init_p50_ms": eg["cold_start_init_p50_ms"],
            "cold_start_init_p95_ms": eg["cold_start_init_p95_ms"],
            "gpu_vram_mib": eg["gpu_vram_mib"],
            "locked_coh_50_correct": eg["by_split"]["LOCKED_COH_50"]["correct"],
            "locked_coh_50_accuracy_pct": eg["by_split"]["LOCKED_COH_50"]["accuracy_pct"],
            "locked_coh_50_ci95_low": eg["by_split"]["LOCKED_COH_50"]["accuracy_wilson_95ci"][0],
            "locked_coh_50_ci95_high": eg["by_split"]["LOCKED_COH_50"]["accuracy_wilson_95ci"][1],
            "blind42_coh_100_correct": eg["by_split"]["BLIND42_COH_100"]["correct"],
            "blind42_coh_100_accuracy_pct": eg["by_split"]["BLIND42_COH_100"]["accuracy_pct"],
            "blind42_coh_100_ci95_low": eg["by_split"]["BLIND42_COH_100"]["accuracy_wilson_95ci"][0],
            "blind42_coh_100_ci95_high": eg["by_split"]["BLIND42_COH_100"]["accuracy_wilson_95ci"][1],
            "organic_h100_correct": eg["by_split"]["Organic_Heldout_H100"]["correct"],
            "organic_h100_accuracy_pct": eg["by_split"]["Organic_Heldout_H100"]["accuracy_pct"],
            "organic_h100_ci95_low": eg["by_split"]["Organic_Heldout_H100"]["accuracy_wilson_95ci"][0],
            "organic_h100_ci95_high": eg["by_split"]["Organic_Heldout_H100"]["accuracy_wilson_95ci"][1],
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
            "fn": deb["fn"],
            "fn_neutral_dominant_count": deb["fn_neutral_dominant_count"],
            "fn_neutral_dominant_pct": deb["fn_neutral_dominant_pct"],
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
            "delta_organic_vs_nemo_pp": round(
                eg["by_split"]["Organic_Heldout_H100"]["accuracy_pct"]
                - nemo["by_split"]["Organic_Heldout_H100"]["accuracy_pct"],
                1,
            ),
            "delta_organic_vs_phi4_pp": round(
                eg["by_split"]["Organic_Heldout_H100"]["accuracy_pct"]
                - phi["by_split"]["Organic_Heldout_H100"]["accuracy_pct"],
                1,
            ),
            "speedup_p50_vs_phi4_x": round(phi["latency_p50_ms"] / eg["latency_p50_ms"], 2),
            "speedup_p50_vs_nemo_x": round(nemo["latency_p50_ms"] / eg["latency_p50_ms"], 2),
            "mcnemar_vs_nemo_eg_only": mcn["vs_mistral_nemo_12b_overall"][
                "epistemic_guard_only_correct"
            ],
            "mcnemar_vs_nemo_base_only": mcn["vs_mistral_nemo_12b_overall"][
                "baseline_only_correct"
            ],
            "mcnemar_vs_nemo_p": round(
                mcn["vs_mistral_nemo_12b_overall"]["p_value_two_sided"], 4
            ),
            "mcnemar_vs_phi4_eg_only": mcn["vs_phi_4_14b_overall"]["epistemic_guard_only_correct"],
            "mcnemar_vs_phi4_base_only": mcn["vs_phi_4_14b_overall"]["baseline_only_correct"],
            "mcnemar_vs_phi4_p": round(mcn["vs_phi_4_14b_overall"]["p_value_two_sided"], 4),
            "mcnemar_organic_vs_nemo_eg_only": mcn["vs_mistral_nemo_12b_organic_h100"][
                "epistemic_guard_only_correct"
            ],
            "mcnemar_organic_vs_nemo_base_only": mcn["vs_mistral_nemo_12b_organic_h100"][
                "baseline_only_correct"
            ],
            "mcnemar_organic_vs_nemo_p": round(
                mcn["vs_mistral_nemo_12b_organic_h100"]["p_value_two_sided"], 4
            ),
            "mcnemar_organic_vs_phi4_eg_only": mcn["vs_phi_4_14b_organic_h100"][
                "epistemic_guard_only_correct"
            ],
            "mcnemar_organic_vs_phi4_base_only": mcn["vs_phi_4_14b_organic_h100"][
                "baseline_only_correct"
            ],
            "mcnemar_organic_vs_phi4_p": round(
                mcn["vs_phi_4_14b_organic_h100"]["p_value_two_sided"], 4
            ),
        },
        "ablations": {
            "deterministic_regex_correct": det_abl["correct"],
            "deterministic_regex_accuracy_pct": det_abl["accuracy_pct"],
            "deterministic_regex_ci95_low": det_abl["accuracy_wilson_95ci"][0],
            "deterministic_regex_ci95_high": det_abl["accuracy_wilson_95ci"][1],
            "deterministic_regex_recall_pct": det_abl["recall_pct"],
            "deterministic_regex_fpr_pct": det_abl["fpr_pct"],
            "deterministic_regex_p50_ms": det_abl["latency_p50_ms"],
            "deterministic_regex_organic_correct": det_abl["by_split"]["Organic_Heldout_H100"][
                "correct"
            ],
            "deterministic_regex_organic_accuracy_pct": det_abl["by_split"]["Organic_Heldout_H100"][
                "accuracy_pct"
            ],
            "stage1_no_receipts_correct": s1_abl["correct"],
            "stage1_no_receipts_accuracy_pct": s1_abl["accuracy_pct"],
            "stage1_no_receipts_ci95_low": s1_abl["accuracy_wilson_95ci"][0],
            "stage1_no_receipts_ci95_high": s1_abl["accuracy_wilson_95ci"][1],
            "stage1_no_receipts_recall_pct": s1_abl["recall_pct"],
            "stage1_no_receipts_fpr_pct": s1_abl["fpr_pct"],
            "stage1_no_receipts_p50_ms": s1_abl["latency_p50_ms"],
            "stage1_no_receipts_organic_correct": s1_abl["by_split"]["Organic_Heldout_H100"][
                "correct"
            ],
            "stage1_no_receipts_organic_accuracy_pct": s1_abl["by_split"]["Organic_Heldout_H100"][
                "accuracy_pct"
            ],
        },
        "live_mcp_e2e_60": {
            "n_total": e2e["n_total"],
            "n_positive_ungrounded": e2e["n_positive_ungrounded"],
            "n_negative_grounded": e2e["n_negative_grounded"],
            "gemma_4_e4b_turns": e2e["generators"]["gemma_4_e4b_turns"],
            "nemotron_3_5_30b_turns": e2e["generators"]["nemotron_3_5_30b_turns"],
            "eg_correct": e2e_eg["correct"],
            "eg_accuracy_pct": e2e_eg["accuracy_pct"],
            "eg_ci95_low": e2e_eg["accuracy_wilson_95ci"][0],
            "eg_ci95_high": e2e_eg["accuracy_wilson_95ci"][1],
            "eg_tp": e2e_eg["tp"],
            "eg_tn": e2e_eg["tn"],
            "eg_fp": e2e_eg["fp"],
            "eg_fn": e2e_eg["fn"],
            "eg_fp_unmapped_tool_stems": fp_ok_clean_unmapped,
            "eg_fp_failed_turn_clauses": fp_failed_turn_clauses,
            "eg_recall_pct": e2e_eg["recall_pct"],
            "eg_fpr_pct": e2e_eg["fpr_pct"],
            "eg_p50_ms": e2e_eg["latency_p50_ms"],
            "eg_p95_ms": e2e_eg["latency_p95_ms"],
            "stage1_correct": e2e_s1["correct"],
            "stage1_accuracy_pct": e2e_s1["accuracy_pct"],
            "stage1_recall_pct": e2e_s1["recall_pct"],
            "stage1_fpr_pct": e2e_s1["fpr_pct"],
            "regex_correct": e2e_det["correct"],
            "regex_accuracy_pct": e2e_det["accuracy_pct"],
            "regex_recall_pct": e2e_det["recall_pct"],
            "deberta_correct": e2e_deb["correct"],
            "deberta_accuracy_pct": e2e_deb["accuracy_pct"],
            "deberta_recall_pct": e2e_deb["recall_pct"],
            "deberta_fn": e2e_deb["fn"],
            "nemo_correct": e2e_nemo["correct"],
            "nemo_accuracy_pct": e2e_nemo["accuracy_pct"],
            "nemo_recall_pct": e2e_nemo["recall_pct"],
            "nemo_fpr_pct": e2e_nemo["fpr_pct"],
            "nemo_p50_ms": e2e_nemo["latency_p50_ms"],
            "phi4_correct": e2e_phi["correct"],
            "phi4_accuracy_pct": e2e_phi["accuracy_pct"],
            "phi4_recall_pct": e2e_phi["recall_pct"],
            "phi4_fpr_pct": e2e_phi["fpr_pct"],
            "phi4_fp": e2e_phi["fp"],
            "phi4_p50_ms": e2e_phi["latency_p50_ms"],
            "delta_vs_stage1_pp": round(e2e_eg["accuracy_pct"] - e2e_s1["accuracy_pct"], 2),
            "delta_vs_phi4_pp": round(e2e_eg["accuracy_pct"] - e2e_phi["accuracy_pct"], 2),
            "speedup_vs_nemo_x": round(e2e_nemo["latency_p50_ms"] / e2e_eg["latency_p50_ms"], 2),
            "speedup_vs_phi4_x": round(e2e_phi["latency_p50_ms"] / e2e_eg["latency_p50_ms"], 2),
            "mcnemar_vs_stage1_eg_only": e2e_mcn["vs_stage1_no_receipts"][
                "epistemic_guard_only_correct"
            ],
            "mcnemar_vs_stage1_base_only": e2e_mcn["vs_stage1_no_receipts"][
                "baseline_only_correct"
            ],
            "mcnemar_vs_stage1_p": round(
                e2e_mcn["vs_stage1_no_receipts"]["p_value_two_sided"], 4
            ),
            "mcnemar_vs_nemo_eg_only": e2e_mcn["vs_mistral_nemo_12b"][
                "epistemic_guard_only_correct"
            ],
            "mcnemar_vs_nemo_base_only": e2e_mcn["vs_mistral_nemo_12b"][
                "baseline_only_correct"
            ],
            "mcnemar_vs_nemo_p": round(
                e2e_mcn["vs_mistral_nemo_12b"]["p_value_two_sided"], 4
            ),
            "mcnemar_vs_phi4_eg_only": e2e_mcn["vs_phi_4_14b"][
                "epistemic_guard_only_correct"
            ],
            "mcnemar_vs_phi4_base_only": e2e_mcn["vs_phi_4_14b"][
                "baseline_only_correct"
            ],
            "mcnemar_vs_phi4_p": round(
                e2e_mcn["vs_phi_4_14b"]["p_value_two_sided"], 4
            ),
        },
        "dataset_provenance": {
            "v030_pre_hardening_correct": prov["v030_pre_hardening_correct"],
            "v030_pre_hardening_accuracy_pct": prov["v030_pre_hardening_accuracy_pct"],
            "v030_blind42_correct": prov["v030_blind42_correct"],
            "v052_zero_arg_strict_fp_delta": prov["v052_zero_arg_strict_fp_delta"],
            "organic_gemma4_e4b_turns": prov["organic_gemma4_e4b_turns"],
            "organic_gemma4_26b_a4b_turns": prov["organic_gemma4_26b_a4b_turns"],
            "organic_dual_rater_overlap_n": prov["organic_dual_rater_overlap_n"],
            "organic_dual_rater_agreement_count": prov["organic_dual_rater_agreement_count"],
            "organic_dual_rater_agreement_pct": prov["organic_dual_rater_agreement_pct"],
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
