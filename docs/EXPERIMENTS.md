# Experiments: where to find each protocol

The scripts are retained under their original names because they encode different evaluation versions. A suffix such as `v1`, `v2`, or `v3` is a research iteration, **not** an assurance that one script supersedes every previous experimental protocol.

| Question | Representative source |
|---|---|
| Basic Mid20 Top-K MIL baseline | `research_7b/train_mid20_topk_mil_v1.py` |
| Region-aware dropout + fake-only boundary penalty + AUC-constrained selection | `research_7b/train_mid20_edge_debiased_mil_v2.py` |
| Sharded full-SD1.4 training and checkpoint selection | `research_7b/train_mid20_edge_debiased_mil_sharded_fullsd14_v1.py` |
| Cross-source GenImage / AIGI-Now evaluation | `research_7b/eval_fullsd14_genimage_aiginow_v1.py` |
| Method-component ablations | `research_7b/eval_component_ablation_full_shards_v1.py`, `research_7b/finalize_component_ablation_v1.py` |
| Input-space causal/evidence-drop interventions | `research_7b/make_causal_perturbations_clean_v2.py`, `research_7b/evaluate_causal_drops_clean_baseline_v2.py` |
| Feature-space causal interventions | `research_7b/evaluate_feature_space_causal_v1.py` |
| Evidence-region faithfulness | `research_7b/run_evidence_region_faithfulness_v1.py`, `research_7b/run_evidence_region_feature_faithfulness_v2.py` |
| Classifier-aware faithfulness | `research_7b/run_classifier_aware_faithfulness_v3.py` |
| Border and crop boundary diagnostic | `research_7b/run_border_band_ablation_v1.py`, `research_7b/run_crop_boundary_diagnostic_v1.py` |
| Flip-equivariance stability | `research_7b/run_causal_flip_equivariance_v1.py` |
| Low-level evidence characterization | `research_7b/analyze_lowlevel_red_blue_v2.py`, `research_7b/analyze_lowlevel_plausibility_v1.py` |
| Visual evidence overlays | `research_7b/export_unseen_top10_overlays_v3_multisource.py` |
| ViT-B Mid6 clean inference | `edge_vitb/inference/infer_pytorch.py` |

## Output and claims

No numeric result table was present in the uploaded source collection. The code and source manifest support claims about **implemented procedures**, but do not independently substantiate any particular AUC, Top-1 boundary reduction, or robustness statistic. Such metrics should be taken from the submitted paper and original verified logs when published.

## Baseline and protocol separation

The 7B study works primarily with **pre-extracted features** and can run head training without loading the entire 7B backbone. The ViT-B path is a **different, smaller network** with distinct checkpoint parameters and feature normalization. Do not point the ViT-B loader at a 7B checkpoint.
