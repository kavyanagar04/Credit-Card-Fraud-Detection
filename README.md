# Credit Card Fraud Detection using Machine Learning

[![Open in nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.org/github/kavyanagar04/Credit-Card-Fraud-Detection/blob/main/Credit_Card_Fraud_Detection.ipynb)

> **Note:** If the Jupyter Notebook fails to load on GitHub, click the **nbviewer** badge above to view it!


This repository contains an end-to-end Machine Learning project to detect fraudulent credit card transactions. It handles a highly imbalanced dataset using advanced resampling techniques (SMOTE) and trains a state-of-the-art Gradient Boosting model (XGBoost) for superior predictive performance compared to a baseline Random Forest model.

## 🚀 Project Overview

The goal of this project is to accurately distinguish between normal and fraudulent credit card transactions. 
The main challenge is the **class imbalance**: fraud cases represent only a tiny fraction of total transactions. To solve this:
- We use **SMOTE** (Synthetic Minority Over-sampling Technique) to synthetically balance the training data.
- We utilize **XGBoost** to effectively model complex patterns in the data.
- We evaluate using metrics suitable for imbalanced datasets, such as **Precision, Recall, F1-Score**, and **MCC** (Matthews Correlation Coefficient), rather than relying solely on Accuracy.

## 📊 Dataset

The dataset used is the [Credit Card Fraud Detection dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
- **Transactions**: 284,807
- **Fraud Cases**: 492 (0.172% of all transactions)
- **Features**: 31 features, including `Time`, `Amount`, and 28 principal components (`V1` to `V28`) obtained via PCA for confidentiality.

## 🛠️ Technologies Used

- **Python**: Primary programming language.
- **Pandas & NumPy**: Data manipulation and numerical operations.
- **Scikit-learn**: Data preprocessing, splitting, and baseline modeling (Random Forest).
- **Imbalanced-learn**: SMOTE implementation for oversampling.
- **XGBoost**: Advanced Gradient Boosting classifier for the final model.
- **Matplotlib & Seaborn**: Data visualization.

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset:**
   Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the root directory of this project.

4. **Run the Notebook:**
   Start Jupyter Notebook or Jupyter Lab and open `credit_card_fraud_detection.ipynb`:
   ```bash
   jupyter notebook credit_card_fraud_detection.ipynb
   ```
   *(Alternatively, you can open the notebook directly in **Google Colab**.)*

## 📈 Results and Evaluation

Because the data is highly imbalanced, accuracy is not a reliable metric. Instead, the models were evaluated based on their ability to correctly identify fraud (Recall) while minimizing false alarms (Precision).

The **XGBoost model trained on SMOTE data** significantly outperformed the baseline Random Forest on the original imbalanced data, offering a much better balance between precision and recall, validated by a strong Matthews Correlation Coefficient (MCC).

---

