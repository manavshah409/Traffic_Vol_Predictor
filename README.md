# 🚦 Metro Interstate Traffic Volume Prediction

A research-grade Machine Learning project to predict hourly traffic volume using advanced feature engineering, ensemble learning, and explainable AI.

---

## 📌 Project Overview

Traffic congestion is a major challenge in modern cities. This project builds a high-performance ML system to predict hourly traffic volume using historical weather and temporal data.

🔍 **Goal:** Predict traffic volume accurately and understand key influencing factors.

---

## 🚀 Key Highlights

- ✅ 7 Machine Learning Models implemented
- ✅ Best R² Score: **0.9853 (Tuned XGBoost)**
- ✅ Feature Engineering: **9 → 27 features**
- ✅ Time-series aware training (no data leakage)
- ✅ SHAP Explainability (global + local insights)
- ✅ 11-page interactive Streamlit dashboard

---

## 🧠 Models Used

| Model | Type |
|------|------|
| Linear Regression | Baseline |
| Random Forest | Bagging |
| Tuned Random Forest | Optimized |
| XGBoost | Boosting |
| Tuned XGBoost | Optimized |
| LightGBM | Boosting |
| Stacking Ensemble | Meta-learning |

---

## 📊 Dataset

- Source: UCI ML Repository
- Records: 48,204 → 38,926 after preprocessing
- Time Range: 2012–2018
- Target: `traffic_volume`

### Features Include:
- Weather data (temperature, rain, clouds)
- Time features (hour, weekday, season)
- Lag features (past traffic)
- Rolling statistics

---

## ⚙️ Data Pipeline

1. Data cleaning (missing values, duplicates)
2. Time parsing & sorting
3. Segment-aware lag feature creation
4. Rolling statistics computation
5. Feature encoding
6. Final dataset preparation

---

## 🔬 Feature Engineering

- Lag Features → Strongest predictors
- Time Features → Capture traffic patterns
- Rolling Stats → Short-term trends
- Interaction Features → Improve model learning

📈 Improved model performance significantly (R² boost ~0.016)

---

## 🧪 Training Strategy

- Train/Test Split:
  - Train: 2012–2016
  - Test: 2017–2018
- Cross-validation: TimeSeriesSplit

✔ Prevents data leakage
✔ Mimics real-world deployment

---

## 🏆 Results

| Model | R² Score |
|------|---------|
| Tuned XGBoost | ⭐ 0.9853 |
| Stacking Ensemble | 0.9852 |
| Random Forest | 0.9847 |
| Linear Regression | 0.9340 |

---

## 🔍 Explainability (SHAP)

Key Insights:
- 🚗 Lag features dominate predictions (~68%)
- ⏰ Time (hour) is highly influential
- 🌦 Weather impact is minimal (<5%)

Includes:
- SHAP summary plots
- Dependence plots
- Individual prediction explanations

---

## 📊 Interactive Dashboard (Streamlit)

Features:
- 📦 Dataset exploration
- 🔍 EDA visualizations
- ⚙️ Feature engineering insights
- 🏆 Model comparison
- 🔮 Live prediction tool
- 🧠 SHAP explainability
- 📈 Cross-validation analysis

---

## 🛠 Tech Stack

- Python
- Scikit-learn
- XGBoost
- LightGBM
- SHAP
- Plotly
- Streamlit
- Pandas, NumPy

---

## 📁 Project Structure

```
.
├── Metro_Interstate_Traffic_Volume.csv   # Dataset
├── traffic_dashboard_v3.py               # Streamlit dashboard (main app)
├── .gitignore                           # Git ignore rules
└── README.md                            # Project documentation
```

---

## ▶️ How to Run

```bash
git clone <repo-url>
cd project-folder
pip install -r requirements.txt
streamlit run app.py
```

---

## ⚠️ Limitations

- Large time gaps in dataset
- Limited weather impact
- Single highway dataset

---

## 🔮 Future Work

- LSTM / Transformer models
- Real-time weather API integration
- Multi-road traffic modeling
- API deployment (FastAPI)

---

## 📌 Conclusion

A high-performance, explainable ML system demonstrating that **feature engineering + time-aware modeling** are key to accurate traffic prediction.

---

## ⭐ If you like this project

Give it a star ⭐ and feel free to contribute!

