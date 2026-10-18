# Bank Marketing Subscription Prediction: Traditional Machine Learning vs. Deep Learning

**Machine Learning Final Classification Project**

This project investigates whether customer and marketing-history information can be used to predict subscription to a bank term deposit. It compares six traditional machine-learning configurations implemented with **Scikit-learn** against three feedforward neural networks built with **TensorFlow/Keras**. Nine experiments are trained and evaluated systematically, with an additional majority-class benchmark.

The project emphasizes reproducible preprocessing, hyperparameter and architecture comparisons, evaluation under class imbalance, independent test-set assessment, and analysis of classification errors.

## Research question

**How do traditional machine-learning methods compare with deep neural networks when predicting bank term-deposit subscriptions using historical customer data?**

The intended use is to support customer prioritization for marketing outreach. The dataset includes some features related to campaign activity, so the model is a research prototype rather than a deployment-ready pre-call decision system.

## Dataset

- **Source:** [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Dataset file:** `bank-full.csv` (the full, real dataset; **not** the smaller `bank.csv`)
- **Original size:** 45,211 records, with 16 input variables and one binary target (`y`)
- **Target:** `yes` = subscribed to a term deposit; `no` = did not subscribe
- **Class balance:** 5,289 subscriptions (**11.7%**) and 39,922 non-subscriptions (**88.3%**)
- **Dataset DOI:** https://doi.org/10.24432/C5K306

The dataset is **not included in the GitHub repository**. Download it from UCI and supply it locally or upload it to Google Colab as described below.

## Methods

The workflow implements:

1. **Exploratory data analysis:** Data types, missing/unknown categories, duplicate rows, the `pdays = -1` sentinel, and class distribution.
2. **Leakage-aware feature selection:** Excludes `duration`, `day`, and `month`. The remaining campaign-related features require a clearly defined prediction time for real-world use.
3. **Stratified partitioning:** 70% training (31,647 records), 15% validation (6,782), and 15% held-out testing (6,782), using random seed `42`.
4. **Feature engineering:** `contacted_before` and `log_campaign`. The `pdays = -1` sentinel is treated as missing for numerical imputation.
5. **Preprocessing:** Training-fitted numerical median imputation and standardization, plus categorical most-frequent imputation and one-hot encoding. Literal `unknown` categories are retained as categories.
6. **Model development:** Six Scikit-learn experiments and three Keras experiments with `tf.data` input pipelines, early stopping, and training logs.
7. **Evaluation:** Accuracy, precision, recall, F1-score, ROC-AUC, and **average precision (AP)**. Model selection and F1-oriented threshold tuning use validation data; the test set is reserved for final evaluation.
8. **Error analysis:** Confusion matrices, ROC and precision–recall curves, learning curves, and a false-negative breakdown by occupation.

### Experiment configurations

| ID | Approach | Main variation |
| --- | --- | --- |
| E1 | Logistic Regression | Baseline (unengineered features) |
| E2 | Logistic Regression | Engineered features and stronger L2 regularization |
| E3 | Random Forest | Baseline forest configuration |
| E4 | Random Forest | Adjusted tree settings, class weighting, engineered features |
| E5 | Histogram-based Gradient Boosting | Boosting configuration with engineered features |
| E6 | Logistic Regression | Engineered-feature ablation using E1 hyperparameters |
| D1 | Dense neural network | Two hidden layers (64, 32), no explicit dropout or L2 |
| D2 | Dense neural network | Hidden layers (128, 64), 30% dropout, L2 regularization |
| D3 | Dense neural network | Hidden layers (128, 64, 32), batch normalization, dropout, lower learning rate |

The dummy majority-class benchmark is reported separately and does **not** count as one of the nine model experiments.

## Results summary

The figures below are from the executed notebook. **Validation** performance was used for model selection; **test** performance was measured on the previously held-out test partition.

| Selected model | Validation ROC-AUC | Test ROC-AUC | Test average precision |
| --- | ---: | ---: | ---: |
| **E4 — Tuned Random Forest** | 0.7649 | 0.7558 | **0.3820** |
| **D2 — Regularized neural network** | **0.7721** | **0.7567** | 0.3714 |

On the held-out test set, the models achieved **almost identical ROC-AUC values**. The tuned Random Forest achieved higher average precision, while the neural network achieved a marginally higher ROC-AUC. The best choice depends on the cost of contacting uninterested customers versus missing potential subscribers.

The validation-selected probability thresholds were approximately **0.5513** for E4 and **0.2515** for D2. At these thresholds, the held-out test results were:

| Model | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: |
| E4 — Tuned Random Forest | 0.8431 | 0.3605 | 0.4414 | 0.3968 |
| D2 — Regularized neural network | 0.8692 | 0.4266 | 0.3443 | 0.3810 |

These threshold-based scores should not be confused with probability-ranking metrics (ROC-AUC and AP). Reported results come from one dataset and one random split; small differences should **not** be interpreted as statistically significant or as evidence that one model family is universally superior.

## Repository structure

The recommended GitHub layout is:

```text
bank-marketing-ml-vs-dl/
├── bank_marketing_ml_vs_dl.ipynb   # Executed, documented notebook
├── README.md                       # Project documentation (this file)
├── requirements.txt                # Python dependencies
└── .gitignore                      # Prevents uploading raw data and artifacts
```

Running the notebook creates directories and files similar to the following **locally / in Colab** (they are not all expected in GitHub):

```text
data/raw/bank-full.csv                  # Raw dataset: keep out of GitHub
outputs/
├── experiment_results.csv
├── experiment_results_sorted.csv
├── final_test_results.csv
├── error_analysis_by_job.csv
├── validation_model_comparison.png
├── D1_Dense_basic_learning_curve.png
├── D2_Dense_regularized_learning_curve.png
├── D3_Dense_BN_learning_curve.png
├── E4_RF_tuned_confusion_matrix.png
├── D2_Dense_regularized_confusion_matrix.png
├── test_roc_curves.png
├── test_pr_curves.png
├── tensorboard/                      # Run-specific logs and CSV histories
└── models/
    ├── best_traditional_ml_pipeline.joblib
    ├── best_deep_learning.keras
    ├── deep_learning_preprocessor.joblib
    └── metadata.json
```

## How to reproduce the experiments

### Option A — Google Colab (recommended)

1. Open `bank_marketing_ml_vs_dl.ipynb` in **Google Colab**.
2. Install dependencies if they are missing. For example, run `!pip install -r requirements.txt` if you have also uploaded `requirements.txt`, or install the individual packages listed there. Google Colab's preinstalled packages may already satisfy some dependencies.
3. Obtain **`bank-full.csv`**, not `bank.csv`, from the [UCI dataset page](https://archive.ics.uci.edu/dataset/222/bank+marketing). In the Colab **Files** sidebar, upload `bank-full.csv` to `/content/`. The notebook also supports a UCI ZIP archive containing that CSV.
4. Run notebook cells **from top to bottom**. The dataset-loading cell copies/extracts the file to `/content/data/raw/bank-full.csv`, verifies the minimum number of rows, and validates the target classes.
5. Allow all nine experiments and the final evaluation to complete. Inspect the tables, plots, and generated files in `/content/outputs/`.
6. **Before the runtime resets**, copy the `outputs/` directory to persistent Google Drive storage or download it as a ZIP. Colab runtime files are temporary.

### Option B — Local Jupyter environment

Use a compatible Python environment (Python 3.10–3.12 is a reasonable starting point; check TensorFlow compatibility for your operating system).

```bash
python -m venv .venv
# Activate the environment using your platform's standard command.
pip install -r requirements.txt
jupyter notebook bank_marketing_ml_vs_dl.ipynb
```

Before executing the dataset-loading cell locally, place the extracted `bank-full.csv` at **`data/raw/bank-full.csv` relative to the notebook's working directory**. The current notebook's automatic upload search otherwise expects Google Colab's `/content/` path.

**Reproducibility note:** The notebook sets `SEED = 42` and attempts TensorFlow deterministic execution. Identical results are not guaranteed across every hardware/software environment. The notebook fits preprocessing only on training data and reserves the test set until after validation-based model and threshold selection. Re-running the evaluation-setup cell after training resets the in-memory experiment log and model dictionaries; run from the beginning for a complete rerun.

## Saved models and inference

- `best_traditional_ml_pipeline.joblib`: The selected **E4** Scikit-learn model together with its fitted preprocessing pipeline.
- `best_deep_learning.keras`: The selected **D2** TensorFlow/Keras network.
- `deep_learning_preprocessor.joblib`: The fitted transformer required to generate D2's 39-dimensional model inputs.
- `metadata.json`: Dataset, split, random seed, excluded variables, selected runs, feature-engineering flag, and decision thresholds.

**Important:** Both selected models rely on the notebook's `make_features()` function for the engineered input variables. Keep that function, input-column schema, and compatible package versions when using the saved artifacts to score new data. Load serialized model files only from sources you trust.

## Limitations and ethical considerations

The underlying data is historical and comes from bank campaigns in Portugal; deployment performance in another location or time period is unknown. Although `duration`, `day`, and `month` are excluded, features such as `contact` and `campaign` may not be available at every pre-contact decision point. Random row-level splitting does not establish prospective temporal generalization or independence between repeat contacts. The dataset has substantial class imbalance, and subgroup analysis shows that error rates differ across occupational categories. Small subgroups support only cautious interpretation. Real-world use would require a better-defined prediction point, time-based evaluation, privacy and fairness review, and ongoing monitoring.

## Submission materials

The assessed submission consists of the documented notebook and supporting repository files on GitHub, the **raw dataset ZIP and saved best-model files** in a viewable Google Drive folder, an IEEE-style written report, and a **5–10 minute** recorded presentation. Put **working GitHub, Google Drive, and video links on the first page of the report**, as required by the assignment. Do **not** upload raw dataset files to the GitHub repository.

## Data citation

Moro, S., Cortez, P., and Rita, P., “A data-driven approach to predict the success of bank telemarketing,” *Decision Support Systems*, vol. 62, pp. 22–31, 2014. See also the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing), DOI: [10.24432/C5K306](https://doi.org/10.24432/C5K306).
