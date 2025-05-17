import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import classification_report

def check_custom_prediction():
    """Check if there's a custom prediction function causing label inversion"""
    # Load original data
    data_path = 'Data/Raw/PhiUSIIL_Phishing_URL_Dataset.csv'
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        # Try alternate path
        data_path = 'Data/PhiUSIIL_Phishing_URL_Dataset.csv'
        if not os.path.exists(data_path):
            print(f"Data file not found: {data_path}")
            return
    
    original_data = pd.read_csv(data_path)
    
    # Extract true labels
    true_labels = original_data['label']
    print("Original label distribution:")
    print(true_labels.value_counts())
    
    # Load prediction function from existing modules
    try:
        from preprocess import load_preprocessing_artifacts
        
        # Check if predict_with_pca_model function exists
        try:
            from pca import predict_with_pca_model
            print("\nUsing predict_with_pca_model from pca module")
            
            # Run the prediction function
            results = predict_with_pca_model(data_path)
            
            # Check the prediction mapping
            print("\nPrediction results mapping:")
            print(results['Prediction'].value_counts())
            
            # Get numeric prediction values
            if 'Phishing' in results['Prediction'].values:
                # Convert string predictions back to numeric
                numeric_preds = results['Prediction'].map({'Legitimate': 0, 'Phishing': 1})
            else:
                numeric_preds = results['Prediction']  # Already numeric
                
            # Compare with original labels (first 10 samples)
            print("\nComparison with original labels (first 10 samples):")
            comparison = pd.DataFrame({
                'URL': original_data['URL'].iloc[:10],
                'True Label': true_labels.iloc[:10],
                'Predicted': numeric_preds.iloc[:10],
                'Match': true_labels.iloc[:10].values == numeric_preds.iloc[:10].values
            })
            print(comparison)
            
            # Check overall accuracy 
            accuracy = (true_labels == numeric_preds).mean()
            print(f"\nAccuracy with original labels: {accuracy:.4f}")
            
            # Check inverted accuracy
            inverted_accuracy = (true_labels == (1 - numeric_preds)).mean()
            print(f"Accuracy with inverted labels: {inverted_accuracy:.4f}")
            
        except ImportError:
            print("predict_with_pca_model function not found in pca module")
        
    except Exception as e:
        print(f"Error in prediction check: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_custom_prediction() 