# CapstoneProject-Udacity

# Insurance Claim Fraud Detection — Capstone Project

Predicting fraudulent motor‑insurance claims, explaining predictions, and translating model scores into dollars saved.

---
## 1 Project Definition

Every year a slice of auto‑insurance payouts goes to false claims. Investigators can catch many, but reviewing **every** claim is expensive. Our goal is to build a model that assigns each new claim a **fraud‑risk score** so adjusters focus on the riskiest subset.

### Problem Statement 
> *Given the structured attributes of a claim, predict its probability of being fraudulent so that resources can be allocated to maximise net savings.*

### Success Metrics
| Metric | Why chosen |
|--------|------------|
| **AUROC** | Measures ranking quality independent of threshold. |
| **PR‑AUC** | Sensitive to minority class (fraud = 6 %). |
| **Net \$‑Savings** | Converts precision/recall into business money using investigation cost and average fraudulent payout. |

---
## 2 Data Snapshot
| Item | Value |
|------|-------|
| Source | [Kaggle — Vehicle Claim Fraud Detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection) |
| Rows | **15 420** historic claims |
| Fraud cases | **923** (6.0 % prevalence) |
| Features | 33 raw → 31 modelled (dropped `PolicyNumber`, `RepNumber`) |
| Missing values | **0** (clean file) |

Features span driver demographics, policy details, accident context, and claim timing.

---
## 3 Methodology
### 3.1 Pre‑processing
* Removed near‑unique identifiers to prevent over‑fitting.
* Numeric columns left as‑is; categorical columns handled natively by CatBoost.

### 3.2 Baseline Model
* **Logistic Regression** + one‑hot + class‑weight balancing.  
  *Test split (20 %)* → AUROC **0.819**, PR‑AUC **0.172**.

### 3.3 Why CatBoost + SHAP
| Challenge | CatBoost advantage |
|-----------|-------------------|
| 20+ categorical columns | Hashes & orders categories internally; no sparse one‑hot mess. |
| Non‑linear feature interactions | Gradient‑boosted trees learn them automatically. |
| Need for explanations | TreeSHAP delivers per‑claim reason codes; adjusters see *why* a claim was flagged. |

### 3.4 Model Refinement
* Grid search *depth {4,6,8} × learning_rate {0.03,0.05}*; early stopping picks **268 trees**.
* **CatBoost test scores** → AUROC **0.871**, PR‑AUC **0.302**.
* **5‑fold out‑of‑fold (OOF)** check → AUROC **0.850**, PR‑AUC **0.246** (robustness confirmed).

### 3.5 Threshold Optimisation
Cost assumptions: investigation = **\$500**, avoided fraudulent payout = **\$8 000**.
* **Hold‑out curve:** best at threshold **0.51** → \$0.86 M saved on the 20 % test slice (≈ \$4 M annualised).
* **OOF curve:** best at threshold **0.25** → \$3.9 M savings on the full dataset.

### 3.6 Explainability Highlights (SHAP)
Top fraud drivers: `Fault`, `BasePolicy`, `VehicleCategory`, `Month`, `PolicyType` — matching domain intuition.

---
## 4 Results Overview
| Model | AUROC | PR‑AUC | Comment |
|-------|-------|--------|---------|
| Logistic Regression | 0.819 | 0.172 | Linear baseline |
| CatBoost (tuned) | **0.871** | **0.302** | + 20 % PR‑AUC lift |

* At threshold **0.51**: catches 93 % of fraud while reviewing 33 % of claims.
* Expected net benefit ≈ **\$4 M per year** on a similarly sized claims book.

---
## 5 Conclusion & Reflection
This project demonstrates an end‑to‑end fraud‑triage pipeline that is both **accurate** and **transparent**:
* CatBoost excels on categorical, tabular insurance data, requiring minimal feature wrangling.
* SHAP explanations convert model scores into investigator‑friendly “reason codes,” easing adoption.
* Dollar‑based threshold tuning aligns data science metrics with real business value.

**Personal take‑aways**  
Balancing recall against the cost of false positives was the core challenge.  Iterating through cost curves made the trade‑off explicit and drove stakeholder buy‑in.  Future work will ingest free‑text claim descriptions with NLP, monitor model drift monthly, and deploy the scorer behind a lightweight Streamlit UI.

---
## 6 Future Enhancements
* **Text Mining** – parse incident narratives for linguistic cues.
* **Active Learning** – retrain on investigator feedback to improve precision.
* **Drift Dashboard** – alert when feature distribution or SHAP importances shift.

---
## 7 Reproduce in Two Steps
1. **Get the data**: download `fraud_oracle.csv` from the Kaggle link above and place it in `data/`.
2. **Run the notebook**: open `notebooks/insurance_fraud.ipynb`, run all cells.  (Dependencies listed in `requirements.txt`.)

---
## 8 Repository Layout
```
├── data/
│   └── fraud_oracle.csv
├── notebooks/
│   └── insurance_fraud.ipynb
├── requirements.txt
└── README.md               # this file
```
---
## 9 References
* Kaggle Dataset – *Vehicle Claim Fraud Detection*  
  <https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection>
* Y. Dorogush et al., 2018 – **CatBoost: unbiased boosting with categorical features**.  
* S. Lundberg & S. Lee, 2017 – **A Unified Approach to Interpreting Model Predictions** (SHAP).
* Blog post - https://medium.com/@bakezhangozha/teaching-a-model-to-sniff-out-insurance-fraud-f1b716e14f7c

