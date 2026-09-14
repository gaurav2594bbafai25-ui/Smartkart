# 🛒 SmartKart Customer Churn Prediction

An end-to-end **Machine Learning project** that predicts whether a SmartKart customer is likely to churn using **Logistic Regression**.

The project demonstrates a complete ML workflow — from cleaning messy customer data to generating a business-ready customer churn risk report.

---

## 📌 Project Overview

Customer churn is a major challenge for retail businesses. Identifying customers who are likely to leave allows companies to take preventive actions such as personalized offers, better customer support, and targeted retention campaigns.

In this project, we build a **binary classification model** to predict:

* `0` → Customer is **Not Likely to Churn**
* `1` → Customer is **Likely to Churn**

The model uses three key customer attributes:

* **Age**
* **Monthly Spend**
* **Complaints**

The final output ranks customers according to their **churn probability**, allowing a retention team to prioritize high-risk customers.

---

## 🎯 Objectives

* Clean and prepare real-world-style customer data
* Handle missing and invalid values
* Detect and treat outliers
* Select meaningful predictive features
* Split data into training and testing sets
* Standardize numerical features
* Build a Logistic Regression model
* Evaluate model performance
* Interpret model coefficients
* Generate customer churn probabilities
* Create a business-ready churn risk report

---

## 🔄 Machine Learning Pipeline

```text
Raw Customer Data
       ↓
Data Cleaning
       ↓
Missing Value Treatment
       ↓
Outlier Detection & Treatment
       ↓
Feature Selection
       ↓
Target Definition
       ↓
Train-Test Split
       ↓
Feature Standardisation
       ↓
Logistic Regression
       ↓
Prediction
       ↓
Model Evaluation
       ↓
Business Interpretation
       ↓
Customer Churn Risk Report
```

---

## 📊 Dataset

The project uses a **100-record retail customer dataset** containing customer information and churn status.

During data cleaning:

* Duplicate records were removed
* Invalid age values were identified
* Negative monthly spending was handled
* Missing values were filled using the median
* Extreme values were treated using the IQR method

After cleaning, the dataset contains **95 usable customer records**.

### Main Features

| Feature         | Description                               |
| --------------- | ----------------------------------------- |
| `Customer_ID`   | Unique customer identifier                |
| `Age`           | Customer age                              |
| `Monthly_Spend` | Customer's monthly spending               |
| `Complaints`    | Number of complaints made by the customer |
| `Churn`         | Target variable: 0 = No Churn, 1 = Churn  |

`Customer_ID` is retained for reporting but is **not used as a model feature**.

---

## 🤖 Machine Learning Model

### Logistic Regression

Logistic Regression was selected because this is a **binary classification problem**.

The model predicts the probability that a customer will churn.

### Input Features

```text
Age
Monthly_Spend
Complaints
```

### Target

```text
Churn
```

Where:

```text
0 = No Churn
1 = Churn
```

---

## 🧹 Data Cleaning

The notebook handles several real-world data quality issues.

### Invalid Age Values

Age values outside the realistic range of **15–90 years** are treated as invalid.

Examples include:

```text
-5
150
"thirty"
blank values
```

Invalid values are replaced with the median age.

### Monthly Spend

Negative spending values are considered invalid and converted to missing values before median imputation.

### Missing Values

Missing values in:

```text
Age
Monthly_Spend
Complaints
```

are filled using the respective column median.

---

## 📈 Outlier Treatment

Outliers are detected using the **Interquartile Range (IQR)** method.

Instead of deleting entire customer records, extreme values are **capped** within the calculated valid range.

This approach preserves the customer's other information while preventing extreme values from disproportionately influencing Logistic Regression.

---

## ⚙️ Feature Standardisation

`StandardScaler` is used to standardize the numerical features.

The scaler is:

1. Fit only on the training dataset
2. Used to transform the training dataset
3. Applied to the test dataset without refitting

This prevents **data leakage** from the test set.

---

## 🧪 Train-Test Split

The cleaned dataset is divided into:

```text
80% → Training Data
20% → Testing Data
```

The split uses:

```python
random_state=42
stratify=y
```

Stratification helps maintain a similar churn distribution in both training and testing datasets.

---

## 📊 Model Evaluation

The model is evaluated using:

* Confusion Matrix
* Accuracy
* Precision
* Recall
* F1-Score
* Classification Report

The notebook's expected results are approximately:

| Metric    | Expected Performance |
| --------- | -------------------: |
| Accuracy  |              ~89–95% |
| Recall    |                ~100% |
| Precision |              ~83–91% |
| F1-Score  |                 High |

> Exact values can vary if the dataset or execution environment changes.

### Why Recall Matters

For a churn prediction problem, **recall is particularly important**.

Missing a customer who is actually going to churn can result in losing that customer.

Therefore, the project prioritizes identifying as many actual churners as possible, even if that creates some false alarms.

---

## 🔍 Model Interpretation

One of the important goals of this project is not only predicting churn but also understanding **why customers may churn**.

The Logistic Regression coefficients provide insight into the relationship between the features and churn risk.

### Key Findings

#### 💰 Monthly Spend

`Monthly_Spend` has a strong **negative relationship** with churn.

This suggests that higher-spending customers tend to have lower churn risk in this dataset.

#### 📞 Complaints

`Complaints` has a strong **positive relationship** with churn.

More complaints are associated with higher churn risk.

This is an important actionable business insight because improving customer support and complaint resolution may help reduce churn.

#### 👤 Age

`Age` has a relatively smaller positive relationship with churn compared with the other features.

---

## 💡 Business Insights

The model provides several practical insights for SmartKart.

### 1. Focus on Customers With Multiple Complaints

Customers with more complaints should receive greater attention from the customer support and retention teams.

### 2. Protect High-Value Customers

High-spending customers appear less likely to churn, making them valuable customers whose relationships should be protected.

### 3. Use Churn Probability for Prioritisation

Instead of treating every customer equally, the business can prioritize customers according to their predicted churn probability.

---

## 📋 Business-Ready Output

The notebook generates:

```text
smartkart_churn_risk_report.csv
```

The report contains:

| Column              | Purpose                         |
| ------------------- | ------------------------------- |
| `Customer_ID`       | Identifies the customer         |
| `Age`               | Customer age                    |
| `Monthly_Spend`     | Monthly spending                |
| `Complaints`        | Number of complaints            |
| `Actual_Churn`      | Actual churn status             |
| `Predicted_Churn`   | Model prediction                |
| `Churn_Probability` | Probability of churn            |
| `Risk_Label`        | Business-friendly risk category |

Example:

```text
Customer_ID → SK001
Churn_Probability → 0.87
Risk_Label → Likely to Churn
```

The customers are sorted by **highest churn probability first**, making the report directly useful for retention teams.

---

## 🛠️ Technologies Used

* 🐍 Python
* 🐼 Pandas
* 🔢 NumPy
* 📊 Matplotlib
* 📈 Seaborn
* 🤖 Scikit-learn
* 📓 Jupyter Notebook / Google Colab
* 📄 CSV

---

## 📂 Project Structure

```text
SmartKart-Churn-Prediction/
│
├── SmartKart_Churn_Prediction_ML_Pipeline.ipynb
├── smartkart_churn_risk_report.csv
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Open the notebook

Open:

```text
SmartKart_Churn_Prediction_ML_Pipeline.ipynb
```

using:

* Google Colab
* Jupyter Notebook
* JupyterLab

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 4. Run the notebook

Run the cells from beginning to end.

The notebook will:

```text
Clean the data
↓
Prepare features
↓
Train the model
↓
Evaluate performance
↓
Predict churn
↓
Generate the risk report
```

---

## 📌 Project Highlights

✅ Complete end-to-end ML pipeline
✅ Real-world data cleaning
✅ Missing-value handling
✅ IQR-based outlier treatment
✅ Feature selection
✅ Train-test split with stratification
✅ Feature standardisation
✅ Logistic Regression
✅ Confusion Matrix
✅ Accuracy, Precision, Recall & F1
✅ Model coefficient interpretation
✅ Churn probability prediction
✅ Business-ready customer risk report

---

## 🎓 Learning Outcomes

This project demonstrates practical understanding of:

* Supervised Machine Learning
* Binary Classification
* Logistic Regression
* Data Preprocessing
* Feature Scaling
* Outlier Treatment
* Model Evaluation
* Business Interpretation of ML Models
* Customer Churn Analytics

---

## 💼 Resume Project Description

> **SmartKart Customer Churn Prediction:** Built an end-to-end customer churn prediction pipeline using Python and Logistic Regression, including data cleaning, missing-value treatment, IQR-based outlier handling, feature standardisation, model evaluation, and coefficient interpretation. Generated a business-ready churn risk report ranking customers by predicted churn probability.

---

## 🔮 Future Improvements

The project can be extended by:

* Adding more customer behavioural features
* Testing Random Forest and XGBoost
* Performing hyperparameter tuning
* Using cross-validation
* Creating an interactive Power BI/Tableau dashboard
* Deploying the model as a web application
* Adding automated retention recommendations
* Monitoring model performance over time
* Using a larger real-world customer dataset

---

## 👨‍💻 Author

**Gaurav Chawla**

BBA FinTech & AI

---

## ⭐ Project Purpose

This project demonstrates how Machine Learning can transform customer data into **actionable business insights**.

The ultimate goal is not simply to predict churn, but to help businesses answer:

> **Which customers are most likely to leave, and where should we focus our retention efforts?**

---

⭐ If you find this project useful, consider giving the repository a star!
# Smartkart
End-to-end customer churn prediction using Logistic Regression, including data cleaning, outlier treatment, feature scaling, model evaluation, interpretation, and a business-ready churn risk report.
