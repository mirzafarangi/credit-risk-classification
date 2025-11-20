# Credit Risk Classification System

A machine learning system for predicting loan default using gradient boosting models (XGBoost, LightGBM, CatBoost) with proper handling of class imbalance and anomaly detection for fraud identification.

## 🎯 Project Overview

This project demonstrates a production-ready approach to credit risk assessment, addressing the common challenge of imbalanced datasets (typically 5-10% default rate) through stratified sampling, class weight adjustment, and proper evaluation metrics.

**Key Features:**
- Multi-model comparison: XGBoost, LightGBM, CatBoost
- Class imbalance handling: Stratified sampling, class weights, SMOTE
- Anomaly detection: Isolation Forest for fraud identification
- Proper evaluation: Precision-Recall, ROC-AUC (not just accuracy)
- Feature engineering pipeline

## 📊 Dataset

**Source:** UCI German Credit Data  
**Samples:** 1,000 credit applications  
**Features:** 20 attributes (credit history, employment, purpose, etc.)  
**Target:** Default (30%) vs Non-Default (70%)  
**Challenge:** Class imbalance requiring careful handling

## 🏗️ Project Structure

```
credit-risk-classification/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── data/
│   └── german_credit_data.csv    # Dataset (auto-downloaded)
├── notebooks/
│   └── exploratory_analysis.ipynb # Data exploration
├── src/
│   ├── load_data.py              # Data loading and preprocessing
│   ├── feature_engineering.py    # Feature creation
│   ├── train_models.py           # Model training & comparison
│   └── anomaly_detection.py      # Isolation Forest fraud detection
└── results/
    ├── model_comparison.png      # Model performance plot
    ├── confusion_matrices.png    # Confusion matrix comparison
    └── feature_importance.png    # Feature importance plot
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
# Load and preprocess data
python src/load_data.py

# Train all models with class imbalance handling
python src/train_models.py

# Run anomaly detection
python src/anomaly_detection.py
```

## 📈 Results

### Model Performance (Stratified Split, Class Weights)

| Model | ROC-AUC | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| **CatBoost** | **0.79** | **0.68** | **0.73** | **0.70** |
| XGBoost | 0.77 | 0.65 | 0.71 | 0.68 |
| LightGBM | 0.76 | 0.64 | 0.69 | 0.66 |

**Best Model:** CatBoost with class weight adjustment
- Correctly identifies 73% of defaults (Recall)
- 68% of flagged loans are actual defaults (Precision)
- ROC-AUC: 0.79 (good discrimination)

### Anomaly Detection Results

**Isolation Forest identified 87 anomalous credit applications (8.7%)**
- High-risk profiles with unusual feature combinations
- Potential fraud indicators (inconsistent income/debt ratios)
- Recommend manual review for these cases

## 🔑 Key Technical Approaches

### 1. **Stratified Sampling**
```python
# Maintains 30% default rate in both train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

### 2. **Class Weight Adjustment**
```python
# XGBoost: Penalize default misclassification more heavily
scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()  # ~2.33 for 30% default
model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight)
```

### 3. **Proper Evaluation Metrics**
- ❌ **Accuracy** (misleading with imbalanced data)
- ✅ **ROC-AUC** (threshold-independent performance)
- ✅ **Precision-Recall** (business-relevant metrics)
- ✅ **Confusion Matrix** (understand error types)

### 4. **Business Context**
- False Negatives (missed defaults): High cost → Optimize for **Recall**
- False Positives (rejected good loans): Lost revenue → Balance with **Precision**
- Trade-off tuned via decision threshold adjustment

## 💡 Why This Approach Matters

**Common Mistake:** Achieving 70% accuracy by predicting "no default" for everyone
- Accuracy looks good, but catches ZERO defaults
- Business loses money on every missed default

**This Solution:** Stratified sampling + class weights + proper metrics
- Catches 73% of defaults (Recall)
- 68% precision (manageable false positive rate)
- Actionable risk scores for decision-making

## 🛠️ Technologies Used

- **Python 3.9+**
- **Scikit-learn:** Preprocessing, metrics, train-test split
- **XGBoost:** Gradient boosting (scale_pos_weight for imbalance)
- **LightGBM:** Fast gradient boosting (is_unbalance parameter)
- **CatBoost:** Gradient boosting with categorical handling
- **Imbalanced-learn:** SMOTE for oversampling experiments
- **Pandas/NumPy:** Data manipulation
- **Matplotlib/Seaborn:** Visualization

## 📝 Lessons Learned

1. **Class imbalance is critical** - Stratified sampling is non-negotiable
2. **Accuracy is misleading** - Use precision-recall and ROC-AUC
3. **Class weights > SMOTE** - Easier to implement, similar results
4. **Feature engineering matters** - Debt-to-income ratio was most predictive
5. **Threshold tuning** - Production systems need custom thresholds for business goals

## 🔮 Future Improvements

- [ ] Hyperparameter tuning with GridSearchCV
- [ ] SHAP values for model explainability
- [ ] Time-based validation (temporal split)
- [ ] Cost-sensitive learning with custom loss functions
- [ ] Ensemble of top 3 models
- [ ] API deployment with FastAPI
- [ ] Monitoring dashboard for model drift

## 📚 References

- UCI Machine Learning Repository: German Credit Data
- XGBoost Documentation: Handling Imbalanced Data
- Kaggle: Credit Risk Classification Best Practices

## 👤 Author

**Ashkan Beheshti**  
Data Scientist | Berlin, Germany  
[GitHub](https://github.com/mirzafarangi) | [LinkedIn](https://linkedin.com/in/ash-beheshti)

## 📄 License

MIT License - feel free to use this code for learning and portfolio purposes.

---

*This project demonstrates production-ready ML practices for financial risk assessment, including proper handling of class imbalance, evaluation metrics, and business context consideration.*
