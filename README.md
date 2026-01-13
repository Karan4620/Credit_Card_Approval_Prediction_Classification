# 💳 Credit Card Approval Prediction System

## 📌 Project Overview
This project builds an **end-to-end machine learning pipeline** to predict **credit card approval decisions** based on applicant demographic and financial attributes. The solution covers the full ML lifecycle — from data ingestion and preprocessing to model training, evaluation, and deployment readiness.

The goal is to simulate a **real-world credit risk assessment system**, emphasizing clean architecture, reproducibility, and scalability.

---

## 🎯 Problem Statement
Financial institutions must efficiently assess whether a credit card application should be approved or rejected while minimizing risk and bias.

Given applicant information such as income, employment status, education, and financial indicators, this project aims to:
- Predict **approval status (Yes / No)**
- Handle **imbalanced class distribution**
- Compare multiple classification models
- Deliver interpretable performance metrics

---

## 📊 Dataset Description
- **Source:** Publicly available credit card application dataset  
- **Target Variable:** Credit Card Approval (Approved / Not Approved)  
- **Features include:**
  - Demographic attributes (gender, age, education, marital status)
  - Financial indicators (income type, housing status)
  - Employment and credit-related variables
- **Challenges addressed:**
  - Mixed data types (categorical + numerical)
  - Missing values
  - Class imbalance

---

## 🧰 Tools & Technologies
- **Language:** Python  
- **Data Analysis:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Machine Learning:** Scikit-learn, Imbalanced-learn (SMOTE)  
- **Model Persistence:** Joblib  
- **Deployment Ready:** Streamlit, AWS S3  

---

## 🏗️ Project Structure
```
├── notebooks/
│   └── Credit_Card_Code_rewritten.ipynb
├── saved_models/
│   └── <model_name>/
├── saved_models_final/
├── requirements.txt
├── README.md
```

---

## 🔄 Workflow & Methodology

### 1️⃣ Data Loading & Preparation
- Combined training and test datasets
- Randomized records to prevent ordering bias
- Separated features and target variable

### 2️⃣ Data Preprocessing
- Numerical features scaled using **MinMaxScaler**
- Categorical features encoded using **OneHotEncoder** and **OrdinalEncoder**
- Implemented preprocessing via **Scikit-learn Pipelines**
- Ensured reproducibility and clean transformations

### 3️⃣ Handling Class Imbalance
- Applied **SMOTE (Synthetic Minority Oversampling Technique)**
- Prevented model bias toward majority class
- Integrated SMOTE directly into the training pipeline

### 4️⃣ Model Training
Multiple classifiers were trained and evaluated, including:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

Each model was trained using **cross-validation** to ensure robust performance estimates.

### 5️⃣ Model Evaluation
Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Curve
- Confusion Matrix

Visual diagnostics were generated for each model to enable **comparative analysis**.

### 6️⃣ Model Persistence
- Trained models and predictions were serialized using **Joblib**
- Saved to disk with versioned directories
- Supports reproducibility and future inference

---

## 📈 Key Results & Insights
- Ensemble-based models (Random Forest, Gradient Boosting) consistently outperformed linear models
- SMOTE significantly improved recall for the minority class
- Pipeline-based preprocessing ensured zero data leakage
- SVM showed strong performance but required longer training time

---

## 🚀 Deployment Readiness
The project is structured for deployment with:
- Serialized models compatible with **Streamlit**
- Support for loading models from **AWS S3**
- Modular functions that separate training, evaluation, and inference logic

This makes the system suitable for:
- Web apps
- Internal decision-support tools
- API-based inference services

---

## ▶️ How to Run the Project

1. Clone the repository
```bash
git clone <repo-url>
cd credit-card-approval-prediction
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the notebook
```bash
jupyter notebook
```

4. (Optional) Launch Streamlit app
```bash
streamlit run app.py
```

---

## 🔮 Future Improvements
- Hyperparameter optimization using GridSearchCV / Optuna
- Fairness and bias analysis
- SHAP-based explainability
- Model monitoring and drift detection
- Full REST API deployment

---

## 👤 Author
**Karan Sangukrishnan**  
MS in Business Analytics  

---

## ⭐ Why This Project Stands Out
- End-to-end ML lifecycle implementation  
- Production-style pipelines and structure  
- Real-world challenges: imbalance, preprocessing, evaluation  
- Deployment-aware design  
