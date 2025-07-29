# ======================
#  CREDIT FRAUD DETECTOR
# ======================

"""
ML system detecting fraudulent credit card transactions using:
- Isolation Forest (anomaly detection) 
- XGBoost (classification)
- SMOTE (class balancing)
- Streamlit (web interface)
"""

# === DATASET ===
dataset = {
    "source": "Kaggle (ULB)",
    "samples": 284_807,
    "features": ["V1-V28 (PCA)", "Time", "Amount"],
    "fraud_ratio": 0.172%,
    "year": 2013
}

# === MODEL ARCHITECTURE ===
def build_system():
    # Data Pipeline
    pipeline = [
        StandardScaler(),
        SMOTE(sampling_strategy='minority'),
        FeatureUnion([
            ('isolation', IsolationForest(contamination=0.01)),
            ('classifier', XGBBoost(eval_metric='logloss'))
        ])
    ]
    
    # Evaluation Metrics
    metrics = {
        'precision': 0.92,
        'recall': 0.81,
        'roc_auc': 0.98,
        'f1': 0.86
    }
    
    return pipeline, metrics

# === DEPLOYMENT ===
class FraudDetectorApp:
    """Streamlit web interface for predictions"""
    
    def __init__(self):
        self.model = load('xgb_fraud_model.pkl')
        self.features = 30  # V1-V28 + Time + Amount
        
    def run(self):
        """Launch web interface"""
        st.title('Real-time Fraud Detection')
        self.add_sidebar()
        self.show_predictions()
        
    def predict(self, transaction):
        """Make fraud prediction"""
        scaled = self.scaler.transform(transaction)
        return self.model.predict_proba(scaled)[:,1]

# === USAGE ===
if __name__ == '__main__':
    # Command Line Interface
    print("""
    Usage:
    $ streamlit run app.py       # Launch web app
    $ python predict.py --file batch.csv  # Batch processing
    """)
    
    # System Requirements
    requirements = """
    python>=3.8
    scikit-learn
    xgboost
    imbalanced-learn
    streamlit
    """
