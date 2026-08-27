# 🏥 AI HealthGuard — AI Health Insurance Risk Manager

## 📌 Project Overview

**AI HealthGuard** is a Machine Learning-based Health Insurance Risk Management application built with **Python, Scikit-learn, and Streamlit**.

The application analyzes customer information to estimate insurance costs, classify risk levels, and provide meaningful insights that can support insurance risk assessment and decision-making.



## 💡 Problem Statement

Insurance companies need to understand customer risk and estimate potential insurance costs accurately.

AI HealthGuard uses Machine Learning to analyze customer attributes and provide an interactive risk assessment.

## 🚀 Solution

Users can enter customer information and receive:

- Predicted insurance cost
- Customer risk classification
- Risk assessment
- Interactive Streamlit interface

## 🔄 Project Workflow

```text
Health Insurance Dataset
          ↓
Data Cleaning
          ↓
Exploratory Data Analysis
          ↓
Feature Engineering
          ↓
Machine Learning Model
          ↓
Insurance Cost Prediction
          ↓
Risk Classification
          ↓
Streamlit Application
```

## 📊 Key Features

### 🔮 Insurance Cost Prediction

Predicts expected insurance cost based on customer information.

### 🚦 Risk Classification

- 🟢 Low Risk
- 🟡 Medium Risk
- 🔴 High Risk

### 🔍 Risk Factors

- Age
- Sex
- BMI
- Smoking status
- Number of children
- Region

## 🧠 Machine Learning

The project uses a **Random Forest Regressor** to predict insurance charges.

### Input Features

```text
Age
Sex
BMI
Children
Smoker
Region
```

### Target Variable

```text
Insurance Charges
```

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Model | Random Forest |
| Web Application | Streamlit |
| Model Management | Joblib |
| Version Control | Git, GitHub |

## 📁 Project Structure

```text
AI-HealthGuard/
│
├── app.py
├── health_insurance.csv
├── health_insurance_model.pkl
├── requirements.txt
└── README.md
```

## ⚙️ Installation

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AI-HealthGuard
pip install -r requirements.txt
```

## ▶️ Run the Streamlit Application

```bash
streamlit run app.py
```

## 📊 Business Value

AI HealthGuard can help insurance teams:

- Identify potentially high-risk customer profiles
- Estimate expected insurance costs
- Understand major risk factors
- Support data-driven risk assessment

## 👨‍💻 Author

**Ajay Vasudevan**

B.Sc. Data Science

**Interests:** Data Analytics | Machine Learning | Artificial Intelligence | Generative AI
