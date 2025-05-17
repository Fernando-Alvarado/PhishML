import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

def load_original_data(file_path):
    """Load the original data with labels"""
    df = pd.read_csv(file_path)
    return df

def check_labels():
    """Check for label inversion issues"""
    # Load original data
    data_path = 'Data/Raw/PhiUSIIL_Phishing_URL_Dataset.csv'
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        # Try alternate path
        data_path = 'Data/PhiUSIIL_Phishing_URL_Dataset.csv'
        if not os.path.exists(data_path):
            print(f"Data file not found: {data_path}")
            return
    
    original_data = load_original_data(data_path)
    
    # Print label distribution in original data
    print("Original label distribution:")
    print(original_data['label'].value_counts())
    print("\nSample of original data:")
    print(original_data[['URL', 'label']].head())
    
    # Try to load saved model
    try:
        model_path = 'pca_results/logistic_regression_model.joblib'
        if not os.path.exists(model_path):
            model_path = 'Notebooks/pca_results/logistic_regression_model_plotly.joblib'
        model = joblib.load(model_path)
        print(f"\nLoaded model from {model_path}")
        print(f"Model classes: {model.classes_}")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    # Compare prediction with true labels
    try:
        # Load model and PCA transformer
        pca_path = 'pca_results/pca_transformer.joblib'
        if not os.path.exists(pca_path):
            pca_path = 'Notebooks/pca_results/pca_transformer_plotly.joblib'
        pca = joblib.load(pca_path)
        
        # Preprocess data similar to training
        from preprocess import load_and_preprocess_data
        X_scaled, y = load_and_preprocess_data(data_path)
        
        # Apply PCA transformation
        X_pca = pca.transform(X_scaled)
        
        # Make predictions on the transformed data
        y_pred = model.predict(X_pca)
        
        # Compare with original labels
        print("\nPrediction vs original labels (first 10 samples):")
        comparison = pd.DataFrame({
            'Original Label': y.iloc[:10].values,
            'Predicted': y_pred[:10],
            'Match': y.iloc[:10].values == y_pred[:10]
        })
        print(comparison)
        
        # Print classification report
        print("\nClassification Report on entire dataset:")
        print(classification_report(y, y_pred))
        
        # Check for inverse mapping
        print("\nTesting inverse mapping hypothesis:")
        inverse_match = y.iloc[:10].values == (1 - y_pred[:10])
        print(pd.DataFrame({
            'Original Label': y.iloc[:10].values,
            'Inverse of Prediction': 1 - y_pred[:10],
            'Match with Inverse': inverse_match
        }))
        
        # Check overall accuracy with inverted predictions
        inverted_accuracy = (y == (1 - y_pred)).mean()
        print(f"\nAccuracy with inverted predictions: {inverted_accuracy:.4f}")
        
    except Exception as e:
        print(f"Error comparing predictions: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_labels() 