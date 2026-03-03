import joblib
import pandas as pd
import os

from features import FEATURE_COLS

# Load model once
model = joblib.load(os.path.join(os.path.dirname(__file__), '../models/xgb_model.pkl'))

SCAM_KEYWORDS = [
    'work from home', 'unlimited earning', 'no experience needed',
    'wire transfer', 'western union', 'make money fast', 'be your own boss',
    'weekly pay', 'guaranteed income', 'financial freedom', 'urgent hiring',
    'no degree required', 'earn from home', 'data entry', 'multi-level'
]

def build_features(description, has_logo, has_company_profile,
                   salary_range, requirements, benefits):
    
    desc = description or ''
    req  = requirements or ''
    ben  = benefits or ''

    data = {
        'has_company_profile': int(has_company_profile),
        'has_salary':          int(bool(salary_range)),
        'has_logo':            int(has_logo),
        'has_questions':       0,
        'has_location':        1,
        'has_requirements':    int(bool(req)),
        'has_benefits':        int(bool(ben)),
        'desc_length':         len(desc),
        'req_length':          len(req),
        'benefit_length':      len(ben),
        'profile_length':      0,
        'benefit_to_req_ratio': len(ben) / (len(req) + 1),
        'scam_keyword_count':  sum(kw in desc.lower() for kw in SCAM_KEYWORDS),
        'telecommuting':       0,
    }

    return pd.DataFrame([data])[FEATURE_COLS]


def predict(description, has_logo, has_company_profile,
            salary_range='', requirements='', benefits=''):

    features = build_features(description, has_logo, has_company_profile,
                               salary_range, requirements, benefits)
    
    prob = model.predict_proba(features)[0][1]

    # Generate human-readable reasons
    reasons = []
    if not has_company_profile:
        reasons.append("❌ No company profile provided")
    if not has_logo:
        reasons.append("❌ No company logo")
    if not requirements:
        reasons.append("⚠️ No requirements listed")
    if not salary_range:
        reasons.append("⚠️ No salary range given")
    
    row = features.iloc[0]
    if row['scam_keyword_count'] > 0:
        reasons.append(f"🚨 Contains {int(row['scam_keyword_count'])} scam keyword(s)")
    if row['benefit_to_req_ratio'] > 3:
        reasons.append("⚠️ Benefits far outweigh listed requirements")

    if not reasons:
        reasons.append("✅ No major red flags detected")

    return round(float(prob), 4), reasons