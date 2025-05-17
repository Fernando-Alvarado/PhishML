import pandas as pd
import joblib
import os
import sys

# Add parent directory to path to import from preprocess.py
sys.path.append('../..')
from preprocess import load_and_preprocess_data

def predict_with_pca_model(data_path, pca_model_path='../pca_results/pca_transformer_plotly.joblib', 
                          lr_model_path='../pca_results/logistic_regression_model_plotly.joblib'):
    """
    Make predictions using the PCA-transformed logistic regression model.
    
    Parameters:
    - data_path: Path to the input data CSV
    - pca_model_path: Path to the saved PCA transformer
    - lr_model_path: Path to the saved logistic regression model
    
    Returns:
    - DataFrame with URLs and predictions
    """
    # Load original data
    original_data = pd.read_csv(data_path)
    
    # Use the same preprocessing function used during training
    X_scaled, _ = load_and_preprocess_data(data_path)
    
    # Load PCA transformer and logistic regression model
    pca = joblib.load(pca_model_path)
    model = joblib.load(lr_model_path)
    
    # Transform data using PCA
    X_pca = pca.transform(X_scaled)
    
    # Make predictions
    predictions = model.predict(X_pca)
    prediction_probas = model.predict_proba(X_pca)[:, 1]
    
    # Create results DataFrame
    results = pd.DataFrame({
        'URL': original_data['URL'],
        'Prediction': predictions,
        'Phishing_Probability': prediction_probas
    })
    
    # Map predictions to labels
    results['Prediction'] = results['Prediction'].map({0: 'Legitimate', 1: 'Phishing'})
    
    return results 