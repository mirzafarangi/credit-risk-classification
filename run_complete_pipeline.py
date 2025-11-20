"""
Complete Credit Risk Classification Pipeline
Run all steps: data loading, model training, and anomaly detection
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.load_data import main as load_data_main
from src.train_models import main as train_models_main
from src.anomaly_detection import main as anomaly_detection_main


def main():
    """Run complete pipeline"""
    print("\n" + "="*80)
    print("CREDIT RISK CLASSIFICATION - COMPLETE PIPELINE")
    print("="*80)
    print("\nThis will run:")
    print("1. Data loading and preprocessing")
    print("2. Model training with class imbalance handling")
    print("3. Anomaly detection for fraud identification")
    print("\n" + "="*80)
    
    try:
        # Step 1: Load and preprocess data
        print("\n\n🔹 STEP 1: DATA LOADING")
        print("="*80)
        load_data_main()
        
        # Step 2: Train models
        print("\n\n🔹 STEP 2: MODEL TRAINING")
        print("="*80)
        train_models_main()
        
        # Step 3: Anomaly detection
        print("\n\n🔹 STEP 3: ANOMALY DETECTION")
        print("="*80)
        anomaly_detection_main()
        
        # Success
        print("\n\n" + "="*80)
        print("🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
        print("="*80)
        print("\n📁 Check the 'results/' folder for:")
        print("   - Model comparison plots")
        print("   - ROC and Precision-Recall curves")
        print("   - Anomaly detection analysis")
        print("   - Saved models (best classifier + isolation forest)")
        
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        print("Please check the error message and try again.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
