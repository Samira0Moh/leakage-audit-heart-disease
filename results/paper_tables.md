## Table 1. Leakage experiment (5×10-fold CV)

| dataset     | clf      | condition                   |   Acc % |    SD |   Δ vs correct (pp) |   ci_low |   ci_high | p_value   |   wilcoxon_p |   auc |   sens |   spec |   brier |   n_test |
|:------------|:---------|:----------------------------|--------:|------:|--------------------:|---------:|----------:|:----------|-------------:|------:|-------:|-------:|--------:|---------:|
| cleveland   | SVM      | C0 correct                  |   80.36 |  8.82 |                0    |   nan    |    nan    |           |     nan      | 87.05 |  76.71 |  83.45 |   0.146 |     30.3 |
| cleveland   | SVM      | L1 scaling before split     |   80.56 |  7.19 |                0.2  |    -2.93 |      3.33 | 0.897     |       0.4451 | 87.75 |  76.56 |  83.95 |   0.143 |     30.3 |
| cleveland   | SVM      | L2 GWO before split         |   82.89 |  6.75 |                2.53 |    -2.28 |      7.34 | 0.296     |       0.0073 | 89.27 |  80.15 |  85.23 |   0.131 |     30.3 |
| cleveland   | SVM      | L3 SMOTE before split       |   80.64 |  6.63 |                0.28 |    -4.27 |      4.83 | 0.902     |       0.659  | 87.9  |  78.44 |  82.55 |   0.141 |     30.3 |
| cleveland   | SVM      | L4a controlled duplication  |   90.35 |  3.51 |                9.99 |     3.01 |     16.97 | 0.006     |     nan      | 95.35 |  88.14 |  92.19 |   0.075 |    102.5 |
| cleveland   | SVM      | L4b real Kaggle file (1025) |   87.28 |  3.87 |                6.92 |    -0.15 |     14    | 0.055     |     nan      | 94.34 |  87.93 |  86.61 |   0.091 |    102.5 |
| cleveland   | Stacking | C0 correct                  |   80.63 |  8.54 |                0    |   nan    |    nan    |           |     nan      | 87.22 |  77.88 |  82.97 |   0.145 |     30.3 |
| cleveland   | Stacking | L1 scaling before split     |   80.43 |  7.12 |               -0.2  |    -3.76 |      3.36 | 0.912     |       0.7088 | 87.72 |  77.14 |  83.23 |   0.143 |     30.3 |
| cleveland   | Stacking | L2 GWO before split         |   82.62 |  6.95 |                1.99 |    -2.86 |      6.83 | 0.414     |       0.0549 | 89.13 |  80.74 |  84.24 |   0.129 |     30.3 |
| cleveland   | Stacking | L3 SMOTE before split       |   80.64 |  6.51 |                0.01 |    -4.47 |      4.5  | 0.995     |       0.9107 | 88.28 |  79.16 |  81.94 |   0.138 |     30.3 |
| cleveland   | Stacking | L4a controlled duplication  |   98.56 |  1.41 |               17.93 |    11.56 |     24.29 | <0.001    |     nan      | 99.65 |  97.9  |  99.1  |   0.014 |    102.5 |
| cleveland   | Stacking | L4b real Kaggle file (1025) |   99.28 |  2.43 |               18.65 |    12.12 |     25.18 | <0.001    |     nan      | 99.82 |  99.39 |  99.16 |   0.006 |    102.5 |
| hungarian   | SVM      | C0 correct                  |   81.51 |  7.39 |                0    |   nan    |    nan    |           |     nan      | 88.06 |  78.4  |  83.27 |   0.136 |     29.4 |
| hungarian   | SVM      | L1 scaling before split     |   80.37 |  6.86 |               -1.14 |    -4.94 |      2.67 | 0.551     |       0.0907 | 87.12 |  76.84 |  82.31 |   0.142 |     29.4 |
| hungarian   | SVM      | L2 GWO before split         |   82.19 |  7.18 |                0.68 |    -3.69 |      5.06 | 0.756     |       0.5037 | 88.33 |  80.31 |  83.26 |   0.133 |     29.4 |
| hungarian   | SVM      | L3 SMOTE before split       |   83.23 |  6.58 |                1.73 |    -2.87 |      6.32 | 0.454     |       0.0229 | 89.8  |  80.13 |  84.96 |   0.126 |     29.4 |
| hungarian   | Stacking | C0 correct                  |   82.05 |  6.49 |                0    |   nan    |    nan    |           |     nan      | 88.02 |  78.75 |  83.9  |   0.132 |     29.4 |
| hungarian   | Stacking | L1 scaling before split     |   80.56 |  7.21 |               -1.49 |    -4.4  |      1.43 | 0.310     |       0.025  | 87.36 |  75.89 |  83.16 |   0.138 |     29.4 |
| hungarian   | Stacking | L2 GWO before split         |   83.28 |  6.71 |                1.23 |    -2.44 |      4.9  | 0.504     |       0.0656 | 88.5  |  80.36 |  84.97 |   0.126 |     29.4 |
| hungarian   | Stacking | L3 SMOTE before split       |   83.7  |  6.82 |                1.65 |    -2.36 |      5.67 | 0.412     |       0.0177 | 90.35 |  81.31 |  85.05 |   0.122 |     29.4 |
| switzerland | SVM      | C0 correct                  |   89.73 |  9.13 |                0    |   nan    |    nan    |           |     nan      | 78.31 |  92.73 |  38    |   0.086 |     12.3 |
| switzerland | SVM      | L1 scaling before split     |   89.73 |  9.6  |                0    |    -3.95 |      3.95 | 1.000     |       0.944  | 81.4  |  92.61 |  40    |   0.082 |     12.3 |
| switzerland | SVM      | L2 GWO before split         |   91.83 |  7.89 |                2.1  |    -3.27 |      7.48 | 0.435     |       0.0778 | 83.18 |  94.26 |  46    |   0.063 |     12.3 |
| switzerland | SVM      | L3 SMOTE before split       |   91.21 |  8.3  |                1.47 |    -3.84 |      6.79 | 0.580     |       0.4372 | 94.04 |  91.7  |  68    |   0.066 |     12.3 |
| switzerland | Stacking | C0 correct                  |   90.71 |  8.25 |                0    |   nan    |    nan    |           |     nan      | 78.98 |  93.97 |  36    |   0.079 |     12.3 |
| switzerland | Stacking | L1 scaling before split     |   90.88 |  8.23 |                0.18 |    -3.91 |      4.26 | 0.930     |       0.6166 | 81.21 |  94.32 |  34    |   0.076 |     12.3 |
| switzerland | Stacking | L2 GWO before split         |   92.99 |  6.96 |                2.28 |    -2.72 |      7.29 | 0.364     |       0.0418 | 82.08 |  95.85 |  42    |   0.06  |     12.3 |
| switzerland | Stacking | L3 SMOTE before split       |   93    |  8.19 |                2.29 |    -2.54 |      7.13 | 0.345     |       0.0403 | 93.84 |  94.33 |  60    |   0.058 |     12.3 |
| va          | SVM      | C0 correct                  |   70.6  |  9.83 |                0    |   nan    |    nan    |           |     nan      | 67.65 |  76.39 |  54.13 |   0.209 |     20   |
| va          | SVM      | L1 scaling before split     |   68.5  |  8.79 |               -2.1  |    -9.42 |      5.22 | 0.567     |       0.1059 | 65.2  |  74.65 |  50.73 |   0.215 |     20   |
| va          | SVM      | L2 GWO before split         |   72    |  9.64 |                1.4  |    -5.61 |      8.41 | 0.690     |       0.4355 | 71.52 |  78.29 |  53.93 |   0.194 |     20   |
| va          | SVM      | L3 SMOTE before split       |   70.1  | 10.51 |               -0.5  |    -7.86 |      6.86 | 0.892     |       0.8221 | 73.2  |  71.82 |  65.33 |   0.202 |     20   |
| va          | Stacking | C0 correct                  |   71    |  8.94 |                0    |   nan    |    nan    |           |     nan      | 65.95 |  79.2  |  47.33 |   0.209 |     20   |
| va          | Stacking | L1 scaling before split     |   69.4  |  7.66 |               -1.6  |    -7.45 |      4.25 | 0.585     |       0.1935 | 66.02 |  77.58 |  45.47 |   0.208 |     20   |
| va          | Stacking | L2 GWO before split         |   72    |  8.06 |                1    |    -5.7  |      7.7  | 0.765     |       0.2579 | 70.43 |  81.08 |  45.6  |   0.192 |     20   |
| va          | Stacking | L3 SMOTE before split       |   71.7  |  9.14 |                0.7  |    -7.18 |      8.58 | 0.859     |       0.6534 | 71.8  |  78.79 |  51    |   0.195 |     20   |

## Table 2. Reproduction of the original paper

| method             |   paper_reported | paper_protocol                 | correct_protocol   |   inflation |
|:-------------------|-----------------:|:-------------------------------|:-------------------|------------:|
| SVM (all features) |            91.22 | 92.49 ± 1.70 (range 88.3–95.6) | 83.14 ± 7.59       |        9.34 |
| PSOGWO + SVM       |            93.66 | 92.44 ± 1.52 (range 89.8–95.6) | 81.69 ± 7.40       |       10.75 |

## Table 3. External validation (train: Cleveland)

| train     | test        | features        |   n_features_used | clf      |   acc |   bal_acc |   auc |   sens |   spec |    f1 |   brier |   majority_acc |   prevalence |   missing_pct |   dropped |
|:----------|:------------|:----------------|------------------:|:---------|------:|----------:|------:|-------:|-------:|------:|--------:|---------------:|-------------:|--------------:|----------:|
| cleveland | hungarian   | all 13 features |                13 | SVM      | 78.23 |     71.87 | 84.07 |  49.06 |  94.68 | 61.9  |   0.162 |           63.9 |         36.1 |          20.5 |       nan |
| cleveland | hungarian   | all 13 features |                13 | Stacking | 78.23 |     72.28 | 82.03 |  50.94 |  93.62 | 62.79 |   0.162 |           63.9 |         36.1 |          20.5 |       nan |
| cleveland | switzerland | all 13 features |                13 | SVM      | 58.54 |     66.2  | 80.43 |  57.39 |  75    | 72.13 |   0.29  |           93.5 |         93.5 |          24.8 |       nan |
| cleveland | switzerland | all 13 features |                13 | Stacking | 59.35 |     66.63 | 75.43 |  58.26 |  75    | 72.83 |   0.295 |           93.5 |         93.5 |          24.8 |       nan |
| cleveland | va          | all 13 features |                13 | SVM      | 57    |     64.69 | 68.9  |  48.99 |  80.39 | 62.93 |   0.305 |           74.5 |         74.5 |          28.7 |       nan |
| cleveland | va          | all 13 features |                13 | Stacking | 58.5  |     66.34 | 69.94 |  50.34 |  82.35 | 64.38 |   0.298 |           74.5 |         74.5 |          28.7 |       nan |

## Table 4. Data audit (Kaggle vs original UCI)

| dataset                   | variable               | mapping_uci_to_kaggle                                                               |
|:--------------------------|:-----------------------|:------------------------------------------------------------------------------------|
| Kaggle 1025 (johnsmith88) | cp                     | typical angina: 1→3, atypical angina: 2→1, non-anginal pain: 3→2, asymptomatic: 4→0 |
| Kaggle 1025 (johnsmith88) | restecg                | normal: 0→1, ST-T abnormality: 1→2, LV hypertrophy: 2→0                             |
| Kaggle 1025 (johnsmith88) | slope                  | upsloping: 1→2, flat: 2→1, downsloping: 3→0                                         |
| Kaggle 1025 (johnsmith88) | thal                   | MISSING(?): ?→0, normal: 3→2, fixed defect: 6→1, reversible defect: 7→3             |
| Kaggle 1025 (johnsmith88) | ca                     | MISSING(?): ?→4, 0: 0→0, 1: 1→1, 2: 2→2, 3: 3→3                                     |
| Kaggle 1025 (johnsmith88) | target                 | label agreement with UCI = 0.0%                                                     |
| Kaggle 1025 (johnsmith88) | duplicates             | 723/1025                                                                            |
| Kaggle 303                | cp                     | typical angina: 1→3, atypical angina: 2→1, non-anginal pain: 3→2, asymptomatic: 4→0 |
| Kaggle 303                | restecg                | normal: 0→1, ST-T abnormality: 1→2, LV hypertrophy: 2→0                             |
| Kaggle 303                | slope                  | upsloping: 1→2, flat: 2→1, downsloping: 3→0                                         |
| Kaggle 303                | thal                   | MISSING(?): ?→0, normal: 3→2, fixed defect: 6→1, reversible defect: 7→3             |
| Kaggle 303                | ca                     | MISSING(?): ?→4, 0: 0→0, 1: 1→1, 2: 2→2, 3: 3→3                                     |
| Kaggle 303                | target                 | label agreement with UCI = 0.0%                                                     |
| Kaggle 303                | duplicates             | 1/303                                                                               |
| Statlog (270)             | overlap with Cleveland | 270/270                                                                             |

## Table 5. Label inversion — metrics

| view                                            |   accuracy |   recall_disease |   precision_disease |
|:------------------------------------------------|-----------:|-----------------:|--------------------:|
| Researcher's belief (Kaggle target=1 'disease') |      77.15 |            83.54 |               76.54 |
| Truth (UCI: 1 = disease)                        |      77.15 |            69.57 |               78.05 |

## Table 6. Label inversion — logistic regression coefficients

| Unnamed: 0   |   Researcher's belief (Kaggle target=1 'disease') |   Truth (UCI: 1 = disease) | interpretation_flipped   |
|:-------------|--------------------------------------------------:|---------------------------:|:-------------------------|
| age          |                                            -0.175 |                      0.175 | True                     |
| trestbps     |                                            -0.218 |                      0.218 | True                     |
| chol         |                                            -0.301 |                      0.301 | True                     |
| thalach      |                                             0.657 |                     -0.657 | True                     |
| oldpeak      |                                            -0.693 |                      0.693 | True                     |
| sex          |                                            -0.777 |                      0.777 | True                     |
| exang        |                                            -0.658 |                      0.658 | True                     |

## Table 7. Swarm feature selection comparison (3×10-fold CV)

| dataset                 | clf      | method          |   acc |   std |   n_feat |   fitness_calls |   fitness_evals |   fs_seconds |   mean_rank |   friedman_p |
|:------------------------|:---------|:----------------|------:|------:|---------:|----------------:|----------------:|-------------:|------------:|-------------:|
| cleveland               | SVM      | All features    | 82.38 |  6.4  |     13   |               0 |               0 |         0    |        1.54 |       0.0032 |
| cleveland               | SVM      | GWO             | 79.72 |  7.81 |      7.2 |            1020 |             192 |         4.22 |        3    |       0.0032 |
| cleveland               | SVM      | LGWO | 81.16 |  6.7  |      7.6 |            1070 |             193 |         4.18 |        3.17 |       0.0032 |
| cleveland               | SVM      | PSO             | 80.74 |  7.37 |      8   |            1020 |             164 |         3.84 |        4    |       0.0032 |
| cleveland               | SVM      | PSOGWO          | 80.61 |  7.17 |      7.6 |            1020 |             206 |         5.18 |        3.29 |       0.0032 |
| cleveland               | Stacking | All features    | 82.49 |  7.28 |     13   |               0 |               0 |         0    |        1.25 |       0.0006 |
| cleveland               | Stacking | GWO             | 80.17 |  6.93 |      7.2 |            1020 |             192 |         4.22 |        3.17 |       0.0006 |
| cleveland               | Stacking | LGWO | 81.06 |  6.37 |      7.6 |            1070 |             193 |         4.18 |        3.54 |       0.0006 |
| cleveland               | Stacking | PSO             | 81.07 |  6.35 |      8   |            1020 |             164 |         3.84 |        3.62 |       0.0006 |
| cleveland               | Stacking | PSOGWO          | 79.63 |  6.87 |      7.6 |            1020 |             206 |         5.18 |        3.42 |       0.0006 |
| hungarian               | SVM      | All features    | 82.55 |  6.61 |     13   |               0 |               0 |         0    |        1.54 |       0.0032 |
| hungarian               | SVM      | GWO             | 80.95 |  7.98 |      7.7 |            1020 |             186 |         4.67 |        3    |       0.0032 |
| hungarian               | SVM      | LGWO | 80.74 |  7.08 |      7.9 |            1070 |             208 |         5.06 |        3.17 |       0.0032 |
| hungarian               | SVM      | PSO             | 80.07 |  7.88 |      8.1 |            1020 |             154 |         3.47 |        4    |       0.0032 |
| hungarian               | SVM      | PSOGWO          | 80.39 |  7.92 |      7.7 |            1020 |             215 |         4.85 |        3.29 |       0.0032 |
| hungarian               | Stacking | All features    | 83.21 |  6.61 |     13   |               0 |               0 |         0    |        1.25 |       0.0006 |
| hungarian               | Stacking | GWO             | 81.75 |  7.06 |      7.7 |            1020 |             186 |         4.67 |        3.17 |       0.0006 |
| hungarian               | Stacking | LGWO | 81.33 |  7.03 |      7.9 |            1070 |             208 |         5.06 |        3.54 |       0.0006 |
| hungarian               | Stacking | PSO             | 81.19 |  6.39 |      8.1 |            1020 |             154 |         3.47 |        3.62 |       0.0006 |
| hungarian               | Stacking | PSOGWO          | 80.97 |  7.57 |      7.7 |            1020 |             215 |         4.85 |        3.42 |       0.0006 |
| kaggle1025 (duplicated) | SVM      | All features    | 93.04 |  2.3  |     13   |               0 |               0 |         0    |      nan    |     nan      |
| kaggle1025 (duplicated) | SVM      | GWO             | 87.52 |  3.5  |      7.2 |            1020 |             186 |         6.21 |      nan    |     nan      |
| kaggle1025 (duplicated) | SVM      | LGWO | 87.16 |  3.16 |      7.3 |            1070 |             201 |         7.16 |      nan    |     nan      |
| kaggle1025 (duplicated) | SVM      | PSO             | 88.48 |  3.16 |      7.8 |            1020 |             154 |         5.88 |      nan    |     nan      |
| kaggle1025 (duplicated) | SVM      | PSOGWO          | 88.43 |  3.34 |      7.6 |            1020 |             206 |         7.36 |      nan    |     nan      |
| kaggle1025 (duplicated) | Stacking | All features    | 99.87 |  0.72 |     13   |               0 |               0 |         0    |      nan    |     nan      |
| kaggle1025 (duplicated) | Stacking | GWO             | 99.09 |  3.18 |      7.2 |            1020 |             186 |         6.21 |      nan    |     nan      |
| kaggle1025 (duplicated) | Stacking | LGWO | 98.54 |  2.76 |      7.3 |            1070 |             201 |         7.16 |      nan    |     nan      |
| kaggle1025 (duplicated) | Stacking | PSO             | 98.92 |  3.19 |      7.8 |            1020 |             154 |         5.88 |      nan    |     nan      |
| kaggle1025 (duplicated) | Stacking | PSOGWO          | 98.83 |  3.34 |      7.6 |            1020 |             206 |         7.36 |      nan    |     nan      |
| switzerland             | SVM      | All features    | 90.6  |  6.23 |     13   |               0 |               0 |         0    |        1.54 |       0.0032 |
| switzerland             | SVM      | GWO             | 89.72 |  8.8  |      5.1 |            1020 |             156 |         3.55 |        3    |       0.0032 |
| switzerland             | SVM      | LGWO | 90.28 |  8.36 |      5.1 |            1070 |             166 |         3.7  |        3.17 |       0.0032 |
| switzerland             | SVM      | PSO             | 89.12 |  7.81 |      5.8 |            1020 |             163 |         3.37 |        4    |       0.0032 |
| switzerland             | SVM      | PSOGWO          | 89.7  |  8.24 |      5.4 |            1020 |             198 |         4.1  |        3.29 |       0.0032 |
| switzerland             | Stacking | All features    | 92.48 |  6.31 |     13   |               0 |               0 |         0    |        1.25 |       0.0006 |
| switzerland             | Stacking | GWO             | 90.81 |  6.97 |      5.1 |            1020 |             156 |         3.55 |        3.17 |       0.0006 |
| switzerland             | Stacking | LGWO | 90.83 |  6.97 |      5.1 |            1070 |             166 |         3.7  |        3.54 |       0.0006 |
| switzerland             | Stacking | PSO             | 91.05 |  6.51 |      5.8 |            1020 |             163 |         3.37 |        3.62 |       0.0006 |
| switzerland             | Stacking | PSOGWO          | 91.35 |  7.01 |      5.4 |            1020 |             198 |         4.1  |        3.42 |       0.0006 |
| va                      | SVM      | All features    | 71.33 |  9.64 |     13   |               0 |               0 |         0    |        1.54 |       0.0032 |
| va                      | SVM      | GWO             | 71.5  |  8.72 |      6   |            1020 |             189 |         3.92 |        3    |       0.0032 |
| va                      | SVM      | LGWO | 67.83 | 10.64 |      6   |            1070 |             187 |         3.9  |        3.17 |       0.0032 |
| va                      | SVM      | PSO             | 69.17 | 10.43 |      6.7 |            1020 |             161 |         3.81 |        4    |       0.0032 |
| va                      | SVM      | PSOGWO          | 71    | 10.03 |      6.9 |            1020 |             203 |         5.03 |        3.29 |       0.0032 |
| va                      | Stacking | All features    | 73.5  |  7.21 |     13   |               0 |               0 |         0    |        1.25 |       0.0006 |
| va                      | Stacking | GWO             | 71.83 |  9.42 |      6   |            1020 |             189 |         3.92 |        3.17 |       0.0006 |
| va                      | Stacking | LGWO | 69.33 | 10.89 |      6   |            1070 |             187 |         3.9  |        3.54 |       0.0006 |
| va                      | Stacking | PSO             | 68.83 |  9.62 |      6.7 |            1020 |             161 |         3.81 |        3.62 |       0.0006 |
| va                      | Stacking | PSOGWO          | 71.83 |  9.33 |      6.9 |            1020 |             203 |         5.03 |        3.42 |       0.0006 |

## Table 8. Friedman test (blocks = hospital × repeat, UCI only)

| clf      | method          |   mean_rank |   friedman_p |   n_blocks |
|:---------|:----------------|------------:|-------------:|-----------:|
| SVM      | All features    |        1.54 |       0.0032 |         12 |
| SVM      | PSO             |        4    |       0.0032 |         12 |
| SVM      | GWO             |        3    |       0.0032 |         12 |
| SVM      | PSOGWO          |        3.29 |       0.0032 |         12 |
| SVM      | LGWO |        3.17 |       0.0032 |         12 |
| Stacking | All features    |        1.25 |       0.0006 |         12 |
| Stacking | PSO             |        3.62 |       0.0006 |         12 |
| Stacking | GWO             |        3.17 |       0.0006 |         12 |
| Stacking | PSOGWO          |        3.42 |       0.0006 |         12 |
| Stacking | LGWO |        3.54 |       0.0006 |         12 |
