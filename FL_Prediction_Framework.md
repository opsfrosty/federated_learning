# Federated Learning Prediction Framework
### Pharmacy Transactional Dataset — Sulaimaniyah, Iraq (2024)

---

## Overview

This document outlines the three core prediction tasks for the hierarchical federated learning (HFL) framework applied to the Pharmacy Transactional Dataset (2.5 million records, 15 pharmacies, January–December 2024).

The framework supports analysis at **pharmacy → zone → city → national** levels using LSTM for forecasting and one-class SVM / autoencoders for anomaly detection.

---

## Dataset Summary

| Property | Detail |
|---|---|
| Records | ~2.5 million (after preprocessing) |
| Pharmacies | 15 (privately operated) |
| Location | Sulaimaniyah, Iraq |
| Period | January 2024 – December 2024 |
| Hierarchy | Zone → City → National |
| Benchmark | `global_test_set.csv` |

### Column Descriptions

| Column | Description |
|---|---|
| `Invoice` | Unique transaction identifier |
| `Barcode` | Unique medication identifier |
| `Name` | Medication name |
| `Dosage_form` | Form of medication (tablet, syrup, capsule, etc.) |
| `Sheet` | Number of sheets per pack |
| `Sales_Sheet` | Quantity of sheets sold |
| `Sales_pack` | Quantity of packs sold |
| `Addedby` | Staff member who recorded the transaction |
| `Addeddate` | Date of transaction (MM-DD-YYYY) |
| `Time_` | Time of transaction |
| `Type` | Product classification (drug or supply) |

---

## Prediction Task 1 — Hierarchical Multi-Horizon Forecasting

### Goal
Predict future medication demand at multiple time horizons and evaluate across all hierarchical levels.

### Target Variables

| Horizon | Target | Formula |
|---|---|---|
| 1 day | `y_d1` | `sum(Sales_pack)` at `t+1` |
| 7 days | `y_d7` | `sum(Sales_pack)` from `t+1` to `t+7` |
| 30 days | `y_d30` | `sum(Sales_pack)` from `t+1` to `t+30` |

### Input Features

| Feature | Derived From |
|---|---|
| Day-of-week, week-of-year, month | `Addeddate` |
| Hour, time-of-day bucket | `Time_` |
| Lag sales: t-1, t-7, t-14, t-30 | `Sales_pack`, `Sales_Sheet` |
| Rolling mean/std (7, 14, 28 days) | `Sales_pack` |
| Product encoding | `Barcode`, `Name`, `Dosage_form`, `Type` |
| Transaction count per day | `Invoice` |

### FL Design
- Each pharmacy trains a local LSTM model on its own data.
- Only model weights/gradients are shared — no raw data leaves the pharmacy.
- Aggregation: FedAvg (or weighted FedAvg by sample size).
- Hierarchical aggregation: pharmacy → zone → city → national.

### Evaluation Metrics

| Metric | Purpose |
|---|---|
| MAE | Mean absolute error per horizon |
| RMSE | Root mean squared error |
| sMAPE | Symmetric mean absolute percentage error |

### Baselines
- **Local-only**: each pharmacy trains independently, no federation.
- **Centralized**: all data pooled (offline benchmark only).
- **Proposed HFL**: hierarchical federated aggregation.

### Expected Insight
FL outperforms local-only under data sparsity, and distinct patterns emerge at each hierarchical level.

---

## Prediction Task 2 — Early Outbreak Risk Prediction

### Goal
Predict the probability of an unusual health-event signal at zone/city level before it is clinically observed, using medication demand as a population-symptom proxy.

### Target Variable
Binary risk label per medicine group / zone / day:

```
risk = 1  if next-week demand > (rolling_mean_28 + 2 × rolling_std_28)
risk = 0  otherwise
```

Or as a continuous risk score (probability output from LSTM classifier).

### Input Features

| Feature | Derived From |
|---|---|
| Demand trajectory (last 7/14/28 days) | `Sales_pack` |
| Rate of change / acceleration | Derived from lag features |
| Product-mix proportions | `Type`, `Dosage_form` |
| Temporal context | `Addeddate` → month, season, week |
| Zone-level aggregated demand | City/Zone CSVs |

### FL Design
- Local pharmacies train risk classifiers on local time series.
- Zone/city aggregation builds stronger regional predictors.
- Global model captures broad seasonal dynamics via FedAvg.
- Output: risk score per zone/city/national level per week.

### Model
- LSTM sequence classifier with calibrated probability output.
- Optional: Temporal CNN or GRU as alternative backbone.

### Evaluation Metrics

| Metric | Purpose |
|---|---|
| AUC-ROC | Overall discrimination ability |
| PR-AUC | Performance under class imbalance |
| F1 / Recall | Alert quality (prioritize high recall) |
| Lead time | Days before peak that model flags risk |

### Expected Insight
Demonstrates actionable public-health early warning capability with clear translational value for policy timing and intervention planning.

---

## Prediction Task 3 — Federated Anomaly Detection

### Goal
Detect unusual transaction or demand patterns that deviate significantly from normal behavior, without requiring labeled outbreak data.

### What Can Be Anomalous

| Signal | Column(s) |
|---|---|
| Sudden spike / drop in sales | `Sales_pack`, `Sales_Sheet` |
| Unusual transaction time patterns | `Time_` |
| Rare shifts in product mix | `Barcode`, `Type`, `Dosage_form` |
| Pharmacy behavior vs own history | All transactional fields |

### Anomaly Score Definition
```
anomaly_score = |actual_sales - predicted_sales|     (forecast-residual method)
             OR reconstruction_error                 (autoencoder method)
```

Threshold: local quantile (e.g., top 1%) or robust statistics (median ± k×MAD).

### FL Design — Option A (Recommended): Federated Representation + Local Detector
1. Train shared feature extractor federatively across all pharmacies.
2. Fit anomaly thresholds/detectors **locally** per pharmacy (preserves local normality).
3. Aggregate anomaly rates upward to zone/city for surveillance.

### FL Design — Option B: Fully Federated Anomaly Model
1. Train one-class SVM or autoencoder parameters via federated rounds.
2. Keep calibration thresholds local.

### Models
- One-class SVM (as referenced in abstract)
- Autoencoder with reconstruction error
- Forecast-residual anomaly: `|actual − LSTM forecast|`

### Evaluation Metrics

| Metric | Condition |
|---|---|
| PR-AUC, AUC-ROC, F1 | If proxy labels available |
| Precision@k | On top-k flagged alerts |
| False alert rate | Stability over time |
| Expert validation | On flagged periods |

### Expected Insight
Identifies local irregular events while preserving full data privacy, and supports continuous surveillance as a complement to the forecasting pipeline.

---

## Combined Framework Architecture

```
Raw Pharmacy Data (stays local)
         │
         ▼
┌─────────────────────┐
│  Local Pharmacy      │  ← LSTM Forecasting
│  Model Training      │  ← Risk Classifier
│  (per pharmacy)      │  ← Anomaly Detector
└────────┬────────────┘
         │  weights only
         ▼
┌─────────────────────┐
│  Zone Aggregator     │  ← FedAvg across pharmacies in zone
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  City Aggregator     │  ← FedAvg across zones
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Global Aggregator   │  ← FedAvg across cities
│  (National Model)    │  ← Evaluated on global_test_set.csv
└─────────────────────┘
```

| Layer | Inputs | Task |
|---|---|---|
| Pharmacy | Local transactions | Local forecasting + anomaly scoring |
| Zone | Pharmacy model weights | Regional demand + risk aggregation |
| City | Zone model weights | City-level health trend prediction |
| National | City model weights | Pandemic/policy-level early warning |

---

## Baseline Comparison Matrix

| Approach | Data Sharing | Privacy | Expected Performance |
|---|---|---|---|
| Local-only | None | Full privacy | Lowest (limited data) |
| Centralized | All raw data | No privacy | Theoretical upper bound |
| **Proposed HFL** | Weights only | Full privacy | Close to centralized |

---

## Key Claims Supported by This Framework

1. FL achieves near-centralized accuracy without sharing raw data.
2. Hierarchical aggregation produces distinct insights at each level.
3. Medication demand serves as a reliable early health-trend proxy.
4. Anomaly detection under FL supports real-time surveillance at scale.
5. The framework is deployable in low-infrastructure healthcare systems.

---

## References

- Dataset: Pharmacy Transactional Dataset, Sulaimaniyah, Iraq (2024)
- Framework: Hierarchical Federated Learning (HFL)
- Methods: FedAvg, LSTM, One-Class SVM, Autoencoder
- Benchmark: `global_test_set.csv`

