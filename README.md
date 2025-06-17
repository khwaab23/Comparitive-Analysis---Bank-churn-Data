# Bank Customer Churn Prediction

A comparative analysis of machine learning models to predict customer churn for a bank, enabling proactive retention strategies and minimizing revenue loss.

## Project Overview

Banks face significant financial challenges due to customer churn, as retaining existing customers is far more cost-effective than acquiring new ones. This project develops, compares, and evaluates various machine learning models to predict whether a customer will exit the bank, helping optimize marketing efforts and improve customer lifetime value.

Key highlights:
- Implements multiple models: Logistic Regression, SVM, Decision Tree, Random Forest, Boosting algorithms (XGBoost, LightGBM, CatBoost), and a simple Neural Network.
- Handles class imbalance using SMOTE and random undersampling.
- Focuses on maximizing recall to identify customers likely to churn.

## Dataset

- **Source:** Kaggle Playground Series - Season 4, Episode 1
- **Records:** Approximately 165,000 bank customer profiles
- **Features:**
  - Demographics: Age, Gender, Geography
  - Financial: Credit Score, Balance, Estimated Salary
  - Account Information: Tenure, Number of Products, Has Credit Card, Is Active Member
  - Target Variable: `Exited` (1 = Churned, 0 = Retained)

## Data Preprocessing

- One-Hot Encoding for categorical features (Geography, Gender)
- Standard scaling for numerical features
- Outlier detection and inspection
- Class imbalance handled using SMOTE and random undersampling

## Models Implemented

| Model | Description |
|-------|--------------|
| Logistic Regression | Baseline with L1 and L2 regularization; improved recall using SMOTE |
| Support Vector Machine (SVM) | Effective for linearly separable data |
| Decision Tree | Tuned for optimal depth and interpretability |
| Random Forest | Tuned with RandomizedSearchCV; robust and high recall |
| Boosting Algorithms (GBM, XGBoost, LightGBM, CatBoost) | Strong performance on structured data with minimal tuning |
| Neural Network | Explored with dropout and early stopping as a benchmark |

## Results and Key Findings

- **Best Performing Model: Random Forest (with SMOTE and hyperparameter tuning)**
  - Accuracy: ~80.6%
  - Recall for churned customers: ~79%
  - ROC-AUC: ~0.88
  - Balanced performance between high recall and interpretability
- **Boosting models (CatBoost, XGBoost, LightGBM):** Comparable results with slight variations, good alternatives to Random Forest.
- **Key insight:** In churn prediction, prioritizing recall is critical to catch potential churners, even at the expense of a slight drop in overall accuracy.

## How to Run

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd <repo-folder>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook or scripts**
   - Open and run `ml_project_bank_churn_dataset_nb.ipynb` for full preprocessing, training, and evaluation.
   - Ensure `train.csv` is available in the project directory.

## Future Work

- Advanced feature engineering (e.g., tenure bands, balance-to-salary ratios)
- Model ensembling (stacking or blending)
- Deployment as a REST API for real-time churn predictions

## Authors

Khwaab Thareja  
Abha Wadjikar

## License

This project is open-source and intended for educational purposes.
