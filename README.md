# Leakage audit of swarm feature selection in heart-disease prediction

Code and results for the paper:

> Hassan, S. M., & Alsadi, N. A. A. *Too Good to Be True? Leakage Audit of Swarm Feature Selection in Heart Disease Prediction.* Journal of Academic Research, The Libyan Academy for Postgraduate Studies (submitted).

## Environment
Python 3.13.1 on Google Colab, scikit-learn 1.6.1, imbalanced-learn 0.14.2, XGBoost 3.4.1.
```
pip install -r requirements.txt
```
All random seeds are fixed in the scripts (cross-validation `random_state=42`, models `random_state=0`, SMOTE seeded per fold).

## Data
See `datasets/README.md`. The UCI files are included; the two Kaggle files must be downloaded.

## Run order
| Script | What it does | Output (in `results/`) | Time |
|---|---|---|---|
| `step0_data_audit.py` | Lineage audit of Kaggle/Statlog files vs. UCI | `data_audit.csv` | s |
| `step2_leakage_experiment.py` | C0 vs. L1–L4, 5×10 CV, corrected t-test, external validation | `leakage_*.csv`, `external_validation.csv` | ~60 min |
| `step3_reproduce_original.py` | Published vs. leak-free protocol for PSOGWO + SVM | `reproduce_original.csv` | min |
| `step4_swarm_comparison.py` | PSO, GWO, PSOGWO, LGWO vs. all features, Friedman | `swarm_*.csv` | ~25 min |
| `step5_label_flip.py` | Logistic regression under both class codings | `label_flip_*.csv` | s |
| `step6_posthoc.py` | Holm and Nemenyi post-hoc tests | `swarm_posthoc.csv` | s |
| `step7_baselines.py` | Logistic regression, nested-tuned SVM, L5, 1-NN check | `baselines_*.csv`, `knn1_check.csv` | ~20 min |
| `step8_exhaustive.py` | Exhaustive search over 8,191 subsets; GWO with SVM fitness | `exhaustive_*.csv` | ~50 min |
| `step9_extra_stats.py` | AUC CIs, Wilson intervals, calibration, Friedman without Switzerland | several | ~15 min |
| `step10_ablation.py` | Model vs. preprocessing ablation | `ablation_*.csv` | ~30 min |
| `step11_pooled_auc.py` | Pooled patient-level AUC with bootstrap | `pooled_auc_bootstrap.csv` | ~1 min |
| `step12_seed_sensitivity.py` | Duplication effect (C0 vs. L4a) repeated with six seeds | `seed_sensitivity*.csv` | ~20 min |
| `make_figures.py`, `make_paper_figures.py` | Figures and tables | `figures/`, `paper_tables.md` | s |

`fs_algorithms.py` contains the swarm optimizers and the fitness function. `step1_check_data.py` is an optional quick duplicate check.
`results/` holds the outputs reported in the paper; re-running overwrites them.
Code comments are partly in Arabic.

## License
Code: MIT. UCI data: CC BY 4.0.
