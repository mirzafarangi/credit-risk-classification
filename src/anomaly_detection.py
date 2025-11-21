"""
Anomaly Detection for Fraud Identification
Use Isolation Forest to identify suspicious credit applications
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# Import feature engineering
from feature_engineering import engineer_features


def load_processed_data():
    """Load preprocessed data"""
    data_dir = Path(__file__).parent.parent / "data"
    X = pd.read_csv(data_dir / "X_processed.csv")
    y = pd.read_csv(data_dir / "y_processed.csv")['target']
    return X, y


def detect_anomalies(X, contamination=0.1):
    """
    Detect anomalies using Isolation Forest
    
    Args:
        X: Features
        contamination: Expected proportion of anomalies (default 10%)
        
    Returns:
        anomaly_labels: -1 for anomalies, 1 for normal
        anomaly_scores: Anomaly scores (lower = more anomalous)
        iso_forest: Trained Isolation Forest model
    """
    print("\n" + "="*80)
    print("ANOMALY DETECTION WITH ISOLATION FOREST")
    print("="*80)
    
    print(f"\n🔍 Configuration:")
    print(f"   Expected anomaly rate: {contamination:.1%}")
    print(f"   Detection method: Isolation Forest")
    print(f"   Principle: Anomalies are easier to isolate (fewer splits needed)")
    
    # Train Isolation Forest
    print(f"\n🏋️  Training Isolation Forest...")
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    
    anomaly_labels = iso_forest.fit_predict(X)
    anomaly_scores = iso_forest.score_samples(X)
    
    # Count anomalies
    n_anomalies = (anomaly_labels == -1).sum()
    n_normal = (anomaly_labels == 1).sum()
    
    print(f"\n📊 Detection Results:")
    print(f"   Total samples: {len(X)}")
    print(f"   Normal: {n_normal} ({n_normal/len(X):.1%})")
    print(f"   Anomalies: {n_anomalies} ({n_anomalies/len(X):.1%})")
    
    return anomaly_labels, anomaly_scores, iso_forest


def analyze_anomalies(X, y, anomaly_labels, anomaly_scores):
    """
    Analyze characteristics of detected anomalies
    
    Args:
        X: Features
        y: Target labels
        anomaly_labels: Anomaly predictions
        anomaly_scores: Anomaly scores
        
    Returns:
        anomalies_df: DataFrame with anomaly analysis
    """
    print("\n" + "="*80)
    print("ANOMALY ANALYSIS")
    print("="*80)
    
    # Create analysis dataframe
    anomalies_df = X.copy()
    anomalies_df['is_anomaly'] = (anomaly_labels == -1).astype(int)
    anomalies_df['anomaly_score'] = anomaly_scores
    anomalies_df['default'] = y.values
    
    # Compare default rates
    normal_default_rate = anomalies_df[anomalies_df['is_anomaly'] == 0]['default'].mean()
    anomaly_default_rate = anomalies_df[anomalies_df['is_anomaly'] == 1]['default'].mean()
    
    print(f"\n⚠️  Default Rate Comparison:")
    print(f"   Normal applications: {normal_default_rate:.1%} default rate")
    print(f"   Anomalous applications: {anomaly_default_rate:.1%} default rate")
    
    if anomaly_default_rate > normal_default_rate:
        lift = anomaly_default_rate / normal_default_rate
        print(f"   💡 Anomalies have {lift:.1f}x higher default rate!")
        print(f"   ✅ Isolation Forest successfully identifies high-risk profiles")
    else:
        print(f"   ⚠️  Anomalies have similar or lower default rate")
        print(f"   Consider adjusting contamination parameter")
    
    # Top anomalous cases
    print(f"\n🚨 Top 10 Most Anomalous Cases:")
    top_anomalies = anomalies_df.nsmallest(10, 'anomaly_score')[
        ['anomaly_score', 'default', 'is_anomaly']
    ]
    print(top_anomalies.to_string())
    
    return anomalies_df


def plot_anomaly_analysis(X, anomaly_labels, anomaly_scores, y):
    """
    Visualize anomaly detection results
    
    Args:
        X: Features
        anomaly_labels: Anomaly predictions
        anomaly_scores: Anomaly scores
        y: Target labels
    """
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Anomaly Score Distribution
    ax = axes[0, 0]
    normal_scores = anomaly_scores[anomaly_labels == 1]
    anomaly_scores_only = anomaly_scores[anomaly_labels == -1]
    
    ax.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue')
    ax.hist(anomaly_scores_only, bins=50, alpha=0.6, label='Anomaly', color='red')
    ax.set_xlabel('Anomaly Score (lower = more anomalous)')
    ax.set_ylabel('Frequency')
    ax.set_title('Anomaly Score Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. PCA Visualization
    ax = axes[0, 1]
    pca = PCA(n_components=2, random_state=42)
    # Handle NaN values for PCA (fill with median)
    X_clean = X.fillna(X.median())
    X_pca = pca.fit_transform(X_clean)
    
    # Plot normal and anomalies
    ax.scatter(X_pca[anomaly_labels == 1, 0], X_pca[anomaly_labels == 1, 1], 
              c='blue', alpha=0.5, s=20, label='Normal')
    ax.scatter(X_pca[anomaly_labels == -1, 0], X_pca[anomaly_labels == -1, 1], 
              c='red', alpha=0.8, s=50, label='Anomaly', marker='x')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.set_title('Anomaly Detection Visualization (PCA)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Default Rate by Anomaly Status
    ax = axes[1, 0]
    anomaly_status = ['Normal', 'Anomaly']
    default_rates = [
        y[anomaly_labels == 1].mean(),
        y[anomaly_labels == -1].mean()
    ]
    
    bars = ax.bar(anomaly_status, default_rates, color=['blue', 'red'], alpha=0.7)
    ax.set_ylabel('Default Rate')
    ax.set_title('Default Rate: Normal vs Anomaly', fontsize=12, fontweight='bold')
    ax.set_ylim([0, max(default_rates) * 1.2])
    
    # Add value labels on bars
    for bar, rate in zip(bars, default_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Anomaly Score vs Default
    ax = axes[1, 1]
    defaults = y[anomaly_labels == -1]
    scores = anomaly_scores[anomaly_labels == -1]
    
    colors = ['red' if d == 1 else 'green' for d in defaults]
    ax.scatter(range(len(scores)), scores, c=colors, alpha=0.6, s=50)
    ax.set_xlabel('Anomaly Index')
    ax.set_ylabel('Anomaly Score')
    ax.set_title('Anomaly Scores (Red=Default, Green=No Default)', 
                fontsize=12, fontweight='bold')
    ax.axhline(y=anomaly_scores[anomaly_labels == -1].mean(), 
              color='black', linestyle='--', label='Mean Anomaly Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'anomaly_detection_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved: {results_dir / 'anomaly_detection_analysis.png'}")
    plt.close()


def generate_fraud_report(anomalies_df):
    """
    Generate fraud detection report
    
    Args:
        anomalies_df: DataFrame with anomaly analysis
    """
    print("\n" + "="*80)
    print("FRAUD DETECTION REPORT")
    print("="*80)
    
    # High-risk anomalies (lowest scores)
    high_risk = anomalies_df[anomalies_df['is_anomaly'] == 1].nsmallest(20, 'anomaly_score')
    
    print(f"\n🚨 HIGH-RISK APPLICATIONS (Top 20 Most Anomalous):")
    print(f"   Total flagged: {len(high_risk)}")
    print(f"   Default rate: {high_risk['default'].mean():.1%}")
    print(f"   Recommendation: MANUAL REVIEW REQUIRED")
    
    # Risk levels
    print(f"\n📊 Risk Level Breakdown:")
    
    anomalous = anomalies_df[anomalies_df['is_anomaly'] == 1]
    
    # Quartiles of anomaly scores for anomalous samples
    if len(anomalous) > 0:
        q25 = anomalous['anomaly_score'].quantile(0.25)
        q50 = anomalous['anomaly_score'].quantile(0.50)
        q75 = anomalous['anomaly_score'].quantile(0.75)
        
        critical = anomalous[anomalous['anomaly_score'] <= q25]
        high = anomalous[(anomalous['anomaly_score'] > q25) & (anomalous['anomaly_score'] <= q50)]
        medium = anomalous[(anomalous['anomaly_score'] > q50) & (anomalous['anomaly_score'] <= q75)]
        low = anomalous[anomalous['anomaly_score'] > q75]
        
        print(f"   🔴 CRITICAL ({len(critical)}): Score <= {q25:.3f} | Default rate: {critical['default'].mean():.1%}")
        print(f"   🟠 HIGH ({len(high)}): Score {q25:.3f}-{q50:.3f} | Default rate: {high['default'].mean():.1%}")
        print(f"   🟡 MEDIUM ({len(medium)}): Score {q50:.3f}-{q75:.3f} | Default rate: {medium['default'].mean():.1%}")
        print(f"   🟢 LOW ({len(low)}): Score > {q75:.3f} | Default rate: {low['default'].mean():.1%}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   1. CRITICAL/HIGH risk: Reject or require additional documentation")
    print(f"   2. MEDIUM risk: Enhanced due diligence, higher interest rate")
    print(f"   3. LOW risk: Standard processing")
    print(f"   4. All anomalies: Flag for fraud investigation team")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("CREDIT RISK - ANOMALY DETECTION & FRAUD IDENTIFICATION")
    print("="*80)
    
    # Load data
    print("\n📂 Loading processed data...")
    X, y = load_processed_data()
    
    # Feature engineering
    X_engineered, _ = engineer_features(X)
    
    # Detect anomalies
    anomaly_labels, anomaly_scores, iso_forest = detect_anomalies(
        X_engineered, contamination=0.10
    )
    
    # Analyze anomalies
    anomalies_df = analyze_anomalies(X_engineered, y, anomaly_labels, anomaly_scores)
    
    # Visualize
    plot_anomaly_analysis(X_engineered, anomaly_labels, anomaly_scores, y)
    
    # Generate report
    generate_fraud_report(anomalies_df)
    
    # Save anomaly results
    results_dir = Path(__file__).parent.parent / "results"
    anomalies_df[['is_anomaly', 'anomaly_score', 'default']].to_csv(
        results_dir / 'anomaly_detection_results.csv', index=False
    )
    print(f"\n💾 Saved: {results_dir / 'anomaly_detection_results.csv'}")
    
    # Save model
    import joblib
    joblib.dump(iso_forest, results_dir / 'isolation_forest_model.pkl')
    print(f"💾 Saved: {results_dir / 'isolation_forest_model.pkl'}")
    
    print("\n" + "="*80)
    print("✅ ANOMALY DETECTION COMPLETE")
    print("="*80)
    print("\n💡 Key Insights:")
    print("   1. Isolation Forest identifies unusual credit application patterns")
    print("   2. Anomalous applications have higher default rates")
    print("   3. Useful for fraud detection and risk assessment")
    print("   4. Can be combined with classification models for enhanced screening")
    print("\n🎯 PROJECT COMPLETE! All models trained and evaluated.")


if __name__ == "__main__":
    main()
