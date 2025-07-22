# 💳 Credit Card Fraud Detection - Machine Learning Project

This project identifies fraudulent credit card transactions using supervised machine learning techniques. The dataset is highly imbalanced, making it an ideal candidate to apply data balancing (SMOTE) and model tuning (Optuna).

---

## 📌 Objective

To build a robust classification model that accurately detects fraudulent transactions with high recall and precision, helping prevent fraud losses for banks and customers.

---

## 📊 Dataset Summary

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Records**: 284,807 transactions
- **Features**: 30 anonymized features (`V1` to `V28`) + `Time`, `Amount`, and `Class`
- **Target**: `Class` (0 = Normal, 1 = Fraud)

---

## 🔁 Project Workflow

| Phase                     | Description |
|--------------------------|-------------|
| **1. Data Collection**     | Fetched anonymized credit card transactions |
| **2. Data Cleaning**       | Removed duplicates (only from Class 0), no missing values |
| **3. EDA & Visualization** | Explored distributions and correlations using `Seaborn` and `Matplotlib` |
| **4. Feature Engineering** | Scaled `Amount` and `Time`; other features preprocessed via PCA |
| **5. Model Training**      | Random Forest with SMOTE applied on training data |
| **6. Hyperparameter Tuning** | Optuna used to optimize Random Forest parameters |
| **7. Evaluation**          | Precision, Recall, F1-score, ROC-AUC, PR-AUC |

---

## ⚙️ Technologies Used

- **Python**: Main language
- **SQL (PostgreSQL)**: Data storage and validation
- **Libraries**: 
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn`, `optuna`, `imbalanced-learn`, `xgboost`

---

## 📈 Evaluation Results

| Metric       | Baseline RF (SMOTE) | Tuned RF (Optuna) |
|--------------|---------------------|-------------------|
| Precision    | 90%                 | 87%               |
| Recall       | 66%                 | 75–78%            |
| F1-Score     | 76%                 | 78.6%             |
| ROC-AUC      | 0.95                | 0.96+             |
| PR-AUC       | 0.74                | 0.77+             |

---

## 🧠 Key Takeaways

- Data imbalance was a major challenge: only **0.17%** transactions were fraud.
- **SMOTE** successfully created a balanced training dataset.
- **Optuna** tuning significantly improved the model's **recall**, reducing missed fraud cases.
- All fraud records were retained, including duplicates, to preserve rare patterns.

---

## 📦 Files in This Repository

| File Name              | Description                              |
| ---------------------- | ---------------------------------------- |
| `creditcard.csv`       | Original dataset from Kaggle             |
| `CC_Project.sql`  | Original Dataset import into Postgresql   |
| `credit_card_MachineLearning.ipynb`   | Exploratory data analysis in Jupyter     |
| `README.md`            | GitHub project documentation (this file) |

## 🚀 Next Steps

- Try XGBoost + Optuna
- Add Stratified K-Fold CV
- Deploy the model via Streamlit or Flask

## Author
Malini Roy Choudhury - Data Analyst | Freelancer

