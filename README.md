# ML System for Credit Default Prediction Using BERT, SHAP & Topic Modeling
Built a machine learning pipeline to predict loan defaults using a combination of structured financial attributes and unstructured borrower narratives. Integrated fine-tuned BERT embeddings, SHAP explainability, and BERTopic to improve prediction accuracy and model interpretability.

## Dataset: 
 - Source: <a href="https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv?select=loan.csv" target="_blank">LendingClub Loan Dataset (Kaggle)</a>
 - Contents: Financial metadata, borrower demographics, and loan purpose narratives

## Architecture
- Preprocessing: Cleaned structured loan data (categoricals, nulls, outliers)
- BERT Risk Score: Fine-tuned BERT to convert narratives into bert_risk_score
- Topic Modeling: Used BERTopic to extract borrower themes by risk pattern
- Modeling: Trained Logistic Regression, Random Forest, XGBoost on structured + BERT + Topic ID features
- Explainability: Integrated SHAP for global and local interpretation
- Evaluation: Measured performance using F1-score, ROC-AUC, accuracy, and calibrated thresholds

## Tools and Technologies
- Languages: Python
- Libraries: Scikit-learn, XGBoost, HuggingFace Transformers, BERTopic, SHAP
- Dev Tools: Jupyter Notebook, Pandas, NumPy, Matplotlib, Seaborn

## Research Questions
- Can borrower narratives meaningfully improve credit risk prediction?
- How do narrative themes correlate with loan default probability?
- Which features — structured or unstructured — are most predictive?
- How can we make our risk models more explainable?

## Results & Insights
- `bert_risk_score` was the most predictive feature across all models
- Logistic Regression achieved best performance (F1: 0.7153, ROC-AUC: 0.8647)
- SHAP visualizations enabled case-level transparency for stakeholders
- BERTopic identified high-risk clusters (e.g. medical expenses, small business loans)
- Topic IDs improved interpretability but didn’t significantly enhance model accuracy

## Future Enhancements
- Deploy model as a REST API (FastAPI or Flask) for real-time credit scoring
- Build interactive dashboards for model insights using Streamlit
- Integrate economic indicators or credit score datasets for richer feature space

## How to Reproduce

```bash
# 1. Clone the repository
git clone https://github.com/your-username/loan-default-prediction.git
cd loan-default-prediction

# 2. Install dependencies
pip install -r requirements.txt
