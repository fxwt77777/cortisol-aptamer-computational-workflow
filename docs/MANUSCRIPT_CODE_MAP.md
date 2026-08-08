# Manuscript-to-code map

This file maps the major computational components described in the manuscript to the corresponding repository locations.

| Manuscript component | Repository location |
|---|---|
| Boltz-2 metric audit and data aggregation | `ml/boltz2_audit/`|
| Top/Bottom 5% extreme-label dataset preparation | `ml/similarity_split/prepare_top5_extremes.py` |
| Sequence-similarity analysis | `ml/similarity_split/run_mmseqs_similarity_batch.sh` |
| MMseqs2 Linclust analysis | `ml/similarity_split/run_mmseqs_linclust_batch.sh` |
| Similarity-aware train/validation/test splitting | `ml/similarity_split/create_similarity_aware_split.py` |
| Logistic Regression baseline | `ml/baselines/train_baselines.py` |
| Random Forest baseline | `ml/baselines/train_baselines.py` |
| CNN-BiLSTM training | `ml/cnn_bilstm/train_cnn_bilstm_5seeds.py` |
| Test-set correlation analysis | `ml/cnn_bilstm/calculate_test_pearson.py` |
| Paired bootstrap comparison with Random Forest | `ml/cnn_bilstm/paired_bootstrap_vs_kmer_rf.py` |
| ML result table generation | `ml/cnn_bilstm/build_ml_results_tables.py` |
| Original CNN-BiLSTM classification model | `ml/original_generation/model_cls.py` |
| Original model training | `ml/original_generation/train_cls.py` |
| Evolutionary sequence generation | `ml/original_generation/evolve_cls.py` |
| Diverse candidate selection | `ml/original_generation/select_diverse_candidates.py` |
| GA-generated candidate rescoring | `ml/ga_rescore/ga60_boltz2_rescore.py` |
| GA candidate rescore analysis | `ml/ga_rescore/analyze_ga60_rescore.py` |
| Independent MD replicate preparation | `md/setup/setup_md_replicates.py` |
| MD replicate seeds and run metadata | `md/setup/replicate_manifest.csv` |
| NVT equilibration parameters | `md/configs/*/rep*/nvt.mdp` |
| NPT equilibration parameters | `md/configs/*/rep*/npt.mdp` |
| 50-ns production MD parameters | `md/configs/*/rep*/md_50ns.mdp` |
| Trajectory preprocessing | `md/analysis/preprocess_trajectory_example.sh` |
| DNA RMSD, Rg, minimum distance, contacts, H-bonds, ligand RMSD, and RMSF | `md/analysis/analyze_core_metrics_example.sh` |
| MM/GBSA execution | `md/mmgbsa/run_replicate_mmgbsa.sh` |
| MM/GBSA result collection | `md/mmgbsa/collect_mmgbsa.py` |
| Derived replicate-level MD summaries | `results_summary/md/` |

## Note

The map documents the custom code included in this repository. Third-party software, models, and external services are not redistributed.
