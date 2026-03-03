import pandas as pd
import re

SCAM_KEYWORDS = [
    'work from home', 'unlimited earning', 'no experience needed',
    'wire transfer', 'western union', 'make money fast', 'be your own boss',
    'weekly pay', 'guaranteed income', 'financial freedom', 'urgent hiring',
    'no degree required', 'earn from home', 'data entry', 'multi-level'
]

def engineer_features(df):
    df = df.copy()

    # --- Binary presence flags ---
    df['has_company_profile'] = df['company_profile'].notna().astype(int)
    df['has_salary']          = df['salary_range'].notna().astype(int)
    df['has_logo']            = df['has_company_logo'].fillna(0).astype(int)
    df['has_questions']       = df['has_questions'].fillna(0).astype(int)
    df['has_location']        = df['location'].notna().astype(int)
    df['has_requirements']    = df['requirements'].notna().astype(int)
    df['has_benefits']        = df['benefits'].notna().astype(int)

    # --- Text length features ---
    df['desc_length']    = df['description'].fillna('').apply(len)
    df['req_length']     = df['requirements'].fillna('').apply(len)
    df['benefit_length'] = df['benefits'].fillna('').apply(len)
    df['profile_length'] = df['company_profile'].fillna('').apply(len)

    # --- Key ratio: benefits vs requirements ---
    # Scammers oversell benefits but list no real requirements
    df['benefit_to_req_ratio'] = df['benefit_length'] / (df['req_length'] + 1)

    # --- Scam keyword count in description ---
    df['scam_keyword_count'] = df['description'].fillna('').apply(
        lambda x: sum(kw in x.lower() for kw in SCAM_KEYWORDS)
    )

    # --- Telecommuting flag ---
    df['telecommuting'] = df['telecommuting'].fillna(0).astype(int)

    # --- Employment type one-hot ---
    df['employment_type'] = df['employment_type'].fillna('Unknown')
    emp_dummies = pd.get_dummies(df['employment_type'], prefix='emp')
    df = pd.concat([df, emp_dummies], axis=1)

    return df


FEATURE_COLS = [
    'has_company_profile', 'has_salary', 'has_logo', 'has_questions',
    'has_location', 'has_requirements', 'has_benefits',
    'desc_length', 'req_length', 'benefit_length', 'profile_length',
    'benefit_to_req_ratio', 'scam_keyword_count', 'telecommuting'
]