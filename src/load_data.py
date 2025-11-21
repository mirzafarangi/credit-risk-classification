"""
Data Loading and Preprocessing
Load German Credit Data from UCI repository
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_german_credit_data():
    """
    Load German Credit Data from UCI ML Repository
    
    Returns:
        pd.DataFrame: Preprocessed credit data
    """
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    data_file = data_dir / "german_credit_data.csv"
    
    # Column names for German Credit Data
    column_names = [
        'checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
        'savings_account', 'employment', 'installment_rate', 'personal_status_sex',
        'other_debtors', 'residence_since', 'property', 'age', 'other_installments',
        'housing', 'existing_credits', 'job', 'num_dependents', 'telephone', 'foreign_worker',
        'credit_risk'  # 1 = good, 2 = bad
    ]
    
    # If file doesn't exist, create sample data for demonstration
    if not data_file.exists():
        print("Creating sample German Credit dataset...")
        
        # UCI German Credit Data URL
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
        
        try:
            # Try to download from UCI
            df = pd.read_csv(url, sep=' ', header=None, names=column_names)
            df.to_csv(data_file, index=False)
            print(f"✅ Downloaded data to {data_file}")
        except Exception as e:
            print(f"⚠️  Could not download from UCI: {e}")
            print("Creating synthetic dataset for demonstration...")
            df = create_synthetic_credit_data()
            df.to_csv(data_file, index=False)
    else:
        df = pd.read_csv(data_file)
        print(f"✅ Loaded data from {data_file}")
    
    return df


def create_synthetic_credit_data():
    """
    Create synthetic credit data for demonstration
    Mimics German Credit Data structure
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic features
    data = {
        'checking_account': np.random.choice(['A11', 'A12', 'A13', 'A14'], n_samples),
        'duration': np.random.randint(6, 72, n_samples),
        'credit_history': np.random.choice(['A30', 'A31', 'A32', 'A33', 'A34'], n_samples),
        'purpose': np.random.choice(['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A48', 'A49', 'A410'], n_samples),
        'credit_amount': np.random.randint(250, 18500, n_samples),
        'savings_account': np.random.choice(['A61', 'A62', 'A63', 'A64', 'A65'], n_samples),
        'employment': np.random.choice(['A71', 'A72', 'A73', 'A74', 'A75'], n_samples),
        'installment_rate': np.random.randint(1, 5, n_samples),
        'personal_status_sex': np.random.choice(['A91', 'A92', 'A93', 'A94'], n_samples),
        'other_debtors': np.random.choice(['A101', 'A102', 'A103'], n_samples),
        'residence_since': np.random.randint(1, 5, n_samples),
        'property': np.random.choice(['A121', 'A122', 'A123', 'A124'], n_samples),
        'age': np.random.randint(19, 75, n_samples),
        'other_installments': np.random.choice(['A141', 'A142', 'A143'], n_samples),
        'housing': np.random.choice(['A151', 'A152', 'A153'], n_samples),
        'existing_credits': np.random.randint(1, 5, n_samples),
        'job': np.random.choice(['A171', 'A172', 'A173', 'A174'], n_samples),
        'num_dependents': np.random.randint(1, 3, n_samples),
        'telephone': np.random.choice(['A191', 'A192'], n_samples),
        'foreign_worker': np.random.choice(['A201', 'A202'], n_samples),
    }
    
    # Create target with ~30% default rate (imbalanced)
    # Make it correlated with features for realistic patterns
    risk_score = (
        (data['duration'] > 36).astype(int) * 2 +
        (data['credit_amount'] > 10000).astype(int) * 2 +
        (data['age'] < 25).astype(int) * 1 +
        np.random.randn(n_samples)
    )
    
    # 1 = good credit (70%), 2 = bad credit (30%)
    data['credit_risk'] = np.where(risk_score > np.percentile(risk_score, 70), 2, 1)
    
    df = pd.DataFrame(data)
    return df


def preprocess_data(df):
    """
    Preprocess credit data for ML models
    
    Args:
        df: Raw credit data
        
    Returns:
        X: Features (encoded and scaled)
        y: Target (0 = good, 1 = bad/default)
        feature_names: List of feature names
    """
    print("\n" + "="*80)
    print("DATA PREPROCESSING")
    print("="*80)
    
    # Create copy
    df = df.copy()
    
    # Convert target to binary (1=good->0, 2=bad->1)
    y = (df['credit_risk'] == 2).astype(int)
    
    print(f"\n📊 Dataset Shape: {df.shape}")
    print(f"   Total samples: {len(df)}")
    print(f"   Features: {df.shape[1] - 1}")
    
    print(f"\n⚖️  Class Distribution:")
    print(f"   Good Credit (0): {(y == 0).sum()} ({(y == 0).mean():.1%})")
    print(f"   Bad Credit (1):  {(y == 1).sum()} ({(y == 1).mean():.1%})")
    print(f"   ⚠️  Imbalance Ratio: 1:{(y == 0).sum() / (y == 1).sum():.1f}")
    
    # Drop target column
    X = df.drop('credit_risk', axis=1)
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"\n🏷️  Feature Types:")
    print(f"   Categorical: {len(categorical_cols)}")
    print(f"   Numerical: {len(numerical_cols)}")
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Scale numerical features
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    feature_names = X.columns.tolist()
    
    print(f"\n✅ Preprocessing complete")
    print(f"   Features ready: {len(feature_names)}")
    
    return X, y, feature_names


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("GERMAN CREDIT RISK DATA - LOADING")
    print("="*80)
    
    # Load data
    df = load_german_credit_data()
    
    # Preprocess
    X, y, feature_names = preprocess_data(df)
    
    # Save processed data
    output_dir = Path(__file__).parent.parent / "data"
    
    # Save features and target
    X.to_csv(output_dir / "X_processed.csv", index=False)
    pd.DataFrame({'target': y.values}).to_csv(output_dir / "y_processed.csv", index=False)
    
    print(f"\n💾 Saved processed data:")
    print(f"   Features: {output_dir / 'X_processed.csv'}")
    print(f"   Target: {output_dir / 'y_processed.csv'}")
    
    print("\n" + "="*80)
    print("✅ DATA LOADING COMPLETE")
    print("="*80)
    print("\nNext step: Run 'python src/train_models.py' to train models")


if __name__ == "__main__":
    main()
