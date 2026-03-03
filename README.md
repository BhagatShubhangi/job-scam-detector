# Fake Recruiter & Job Scam Detector

A machine learning system that automatically flags suspicious job postings and fake recruiter profiles — built to protect job seekers from data harvesting, financial fraud, and malware distribution.

Live demo: Paste any job description and get an instant scam probability score with a human-readable breakdown of why the model is suspicious.



<img width="1919" height="970" alt="image" src="https://github.com/user-attachments/assets/2d73eddf-d7a5-4458-a9f6-0c5d1031bc48" />

<img width="1919" height="976" alt="image" src="https://github.com/user-attachments/assets/118cd25f-b409-44a0-9889-be8fa5e03966" />

<img width="1919" height="968" alt="image" src="https://github.com/user-attachments/assets/7527aa2b-e531-4748-972c-d1dcd368a83c" />

<img width="1909" height="965" alt="image" src="https://github.com/user-attachments/assets/d5445630-a899-4810-b3b6-fa5ae7d40c33" />


## The Problem

Every day, thousands of fake recruiters flood platforms like LinkedIn with fraudulent job listings. These scams harvest personal data, extract money, and distribute malware — and they're nearly impossible to detect manually at scale. This project builds an automated system to catch them before they reach victims.


## How It Works

The model extracts a behavioral fingerprint from each job posting using 14 engineered features:

| Feature | Why It Matters |
|---|---|
| has_company_profile | Scammers rarely build out full company profiles |
| has_logo | No logo = 16% fraud rate vs 2% with logo |
| scam_keyword_count | Phrases like "guaranteed income", "wire transfer" |
| benefit_to_req_ratio | Scams oversell benefits, list no real requirements |
| desc_length / req_length | Fake postings tend to be vague and short |
| has_salary | Missing salary is a weak but consistent signal |

These features feed into an XGBoost classifier trained on the EMSCAD dataset, catching 86% of all scam postings while keeping false positives manageable.



## Dataset

EMSCAD - Employment Scam Aegean Corpus and Dataset
Source: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

- ~18,000 real-world job postings
- 866 confirmed fraudulent (~4.8%)
- Class imbalance handled via SMOTE oversampling



## Results

| Model | Precision (fraud) | Recall (fraud) | F1 |
|---|---|---|---|
| Logistic Regression (baseline) | 0.15 | 0.60 | 0.24 |
| XGBoost (final) | 0.26 | 0.86 | 0.40 |

Confusion Matrix (XGBoost):
- True Positives (caught scams): 149
- False Negatives (missed scams): 24
- False Positives (real jobs flagged): 415
- True Negatives (real jobs cleared): 2,988

Design choice: In fraud detection, missing a scam is worse than a false alarm. The model is tuned to maximize recall rather than chasing raw accuracy.



## Top Features by Importance

```
has_company_profile    33.7%
has_logo               19.4%
has_benefits            8.1%
profile_length          6.6%
has_location            4.8%
```



## Project Structure

```
job-scam-detector/
├── data/
│   ├── raw/                  # Original EMSCAD dataset
│   └── processed/            # Engineered features
├── notebooks/
│   ├── _eda.ipynb            # Exploratory data analysis
│   ├── _features.ipynb       # Feature engineering & validation
│   ├── _modeling.ipynb       # Model training & evaluation
│   └── _explainability.ipynb # Feature importance
├── src/
│   ├── features.py           # Feature engineering logic
│   ├── train.py              # Model training pipeline
│   └── predict.py            # Inference + human-readable reasons
├── models/
│   └── xgb_model.pkl         # Saved XGBoost model
└── app/
    └── streamlit_app.py      # Live demo app
```



## Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/job-scam-detector
cd job-scam-detector

# Install dependencies
pip install pandas numpy scikit-learn xgboost imbalanced-learn streamlit joblib

# Train the model
cd src
python train.py

# Launch the app
cd ../app
python -m streamlit run streamlit_app.py
```



## Key Engineering Decisions

**Why XGBoost over Random Forest?**
XGBoost's regularization handles noise in text-derived features better, trains faster, and gives more stable feature importances across runs.

**Why SMOTE over class weights?**
SMOTE generates synthetic fraud examples in feature space rather than just reweighting — forcing the model to learn a richer decision boundary around the minority class.

**Why precision-recall over accuracy?**
With only 4.8% fraud, a model that calls everything "real" gets 95.2% accuracy — and catches zero scams. Recall on the fraud class is the metric that actually matters.

**What the model gets wrong:**
Sophisticated scams that include a real company name, plausible description, and correct formatting can slip through. The model's weakness is content quality, not structural signals.


## Tech Stack

- Python — pandas, scikit-learn, XGBoost, imbalanced-learn
- Streamlit — live interactive demo
- Jupyter — EDA and experimentation
