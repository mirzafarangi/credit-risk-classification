"""
Feature Engineering for Credit Risk
Create additional features to improve model performance
"""

import pandas as pd
import numpy as np


def create_financial_ratios(df):
    """
    Create financial ratio features
    
    Args:
        df: DataFrame with raw features
        
    Returns:
        DataFrame with added ratio features
    """
    df = df.copy()
    
    # Credit amount per month ratio
    if 'credit_amount' in df.columns and 'duration' in df.columns:
        df['monthly_payment'] = df['credit_amount'] / df['duration']
        df['credit_to_duration_ratio'] = df['credit_amount'] / (df['duration'] + 1)
    
    # Age-based features
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 100], labels=[0, 1, 2, 3])
        df['is_young'] = (df['age'] < 25).astype(int)
        df['is_senior'] = (df['age'] > 60).astype(int)
    
    # Installment burden
    if 'installment_rate' in df.columns and 'duration' in df.columns:
        df['total_installment_burden'] = df['installment_rate'] * df['duration']
    
    return df


def create_risk_indicators(df):
    """
    Create risk indicator features based on domain knowledge
    
    Args:
        df: DataFrame with raw features
        
    Returns:
        DataFrame with added risk indicators
    """
    df = df.copy()
    
    # High credit amount flag
    if 'credit_amount' in df.columns:
        df['high_credit_amount'] = (df['credit_amount'] > df['credit_amount'].quantile(0.75)).astype(int)
    
    # Long duration flag
    if 'duration' in df.columns:
        df['long_duration'] = (df['duration'] > 36).astype(int)
    
    # Young borrower with high credit
    if 'age' in df.columns and 'credit_amount' in df.columns:
        df['young_high_credit'] = ((df['age'] < 25) & (df['credit_amount'] > df['credit_amount'].median())).astype(int)
    
    # Multiple existing credits
    if 'existing_credits' in df.columns:
        df['multiple_credits'] = (df['existing_credits'] > 1).astype(int)
    
    return df


def engineer_features(X):
    """
    Main feature engineering pipeline
    
    Args:
        X: Raw features DataFrame
        
    Returns:
        X_engineered: DataFrame with engineered features
        new_feature_names: List of new feature names
    """
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    original_features = X.shape[1]
    
    # Apply feature engineering
    X_engineered = create_financial_ratios(X)
    X_engineered = create_risk_indicators(X_engineered)
    
    # Convert age_group to numeric if it exists
    if 'age_group' in X_engineered.columns:
        X_engineered['age_group'] = X_engineered['age_group'].astype(float)
    
    new_features = X_engineered.shape[1] - original_features
    new_feature_names = [col for col in X_engineered.columns if col not in X.columns]
    
    print(f"\n📊 Feature Engineering Results:")
    print(f"   Original features: {original_features}")
    print(f"   New features: {new_features}")
    print(f"   Total features: {X_engineered.shape[1]}")
    
    if new_feature_names:
        print(f"\n✨ New Features Created:")
        for feat in new_feature_names:
            print(f"   - {feat}")
    
    print("\n✅ Feature engineering complete")
    
    return X_engineered, new_feature_names


if __name__ == "__main__":
    # Demo feature engineering
    from pathlib import Path
    
    # Load processed data
    data_dir = Path(__file__).parent.parent / "data"
    X = pd.read_csv(data_dir / "X_processed.csv")
    
    # Engineer features
    X_engineered, new_features = engineer_features(X)
    
    print(f"\n📝 Sample of engineered features:")
    print(X_engineered[new_features].head() if new_features else "No new features")
