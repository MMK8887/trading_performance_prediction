# 📘 Sentiment-Driven Trading Performance Prediction

### Machine Learning & Time-Series Forecasting using Market Sentiment + Trader Performance Data

---

## Project Overview
This project analyzes the relationship between **market sentiment** (Fear-Greed Index) and **daily trading performance** and predicts whether **the next day will be profitable**. It integrates sentiment data with live trading execution records and builds a full forecasting + classification pipeline using machine learning, cross-validation, SHAP explainability, and statistical time-series models.

---

##  Key Capabilities
- End-to-end ML workflow and real financial forecasting
- Sentiment + trade performance dataset merge (date-aligned)
- VIF multicollinearity detection and lag-based feature engineering
- Chronological split (no lookahead bias)
- SMOTE to balance target classes
- Multiple ML algorithms benchmarked & tuned
- Walk-forward rolling retrain evaluation for realistic simulation
- Statistical & ML forecasting comparison (Prophet vs ARIMA)
- SHAP model interpretability




##  Pipeline Summary
### **Data Preparation**
- Clean raw trade records
- Aggregate by account and trading day
- Merge with sentiment index
- Create engineered features:
  - Lagged sentiment (1, 3, 7 days)
  - Lagged PnL (1, 3, 7 days)
  - Win rate, trade count, avg trade size

### **Feature Processing**
- VIF multicollinearity analysis
- Scaling (StandardScaler)
- SMOTE oversampling on training data only
- Time-Series Split (TimeSeriesSplit) cross-validation

---

##  Machine Learning Models
| Model | Baseline | Tuned | Notes |
|--------|---------|---------|--------|
| Logistic Regression | ✔ | ✔ | Best interpretability |
| Random Forest | ✔ | ✔ | Strong benchmark model |
| XGBoost | ✔ | ✔ | Highest predictive power |
| LightGBM | ✔ | ✔ | Efficient & stable |
| CatBoost | ✔ | ✔ | Sparse & categorical-friendly |

### Evaluation Metrics
- Accuracy, Precision, Recall, F1
- ROC-AUC curves
- Bootstrapped 95% confidence intervals
- Rolling-window walk-forward performance

---

##  Forecasting Models (Future PnL Prediction)
| Model | Purpose |
|--------|----------|
| Prophet | Seasonality & future PnL forecasting |
| ARIMA | Statistical baseline for comparison |

---

## Explainability
| Technique | Deliverable |
|------------|---------------|
| SHAP | Feature impact visualization & interpretability |
| Feature importances & coefficient ranking | Understand drivers of performance |

---

## Visualizations Included
- PnL distribution & sentiment histograms
- Correlation heatmaps
- ROC curves for all models
- SHAP summary
- Prophet forecast & ARIMA fit
- Rolling-window accuracy graph

---

##  Key Insights
- Lagged PnL and sentiment score strongly influence next-day profitability
- Higher trade count and win-rate correlate with profitable outcomes
- Ensemble models provide the strongest predictive signal
- Walk-forward testing is essential for realistic evaluation



##Future Enhancements

Deep learning (LSTM / Temporal CNN)



## Author
Manish Kumar
Quantitative Trading | Machine Learning | Time-Series Forecasting
