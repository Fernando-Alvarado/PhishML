import pandas as pd
import os
import sys

def test_predictions():
    """Test predictions from both pca.py and notebook version"""
    # Define data path
    data_path = 'Data/Raw/PhiUSIIL_Phishing_URL_Dataset.csv'
    if not os.path.exists(data_path):
        data_path = 'Data/PhiUSIIL_Phishing_URL_Dataset.csv'
        if not os.path.exists(data_path):
            print("Data file not found")
            return
    
    # Load true labels
    print("Loading original data...")
    original_data = pd.read_csv(data_path)
    true_labels = original_data['label']
    
    # Test pca.py predictions
    print("\n--- Testing pca.py predictions ---")
    try:
        from pca import predict_with_pca_model as predict_pca
        
        print("Running predictions with pca.py...")
        results_pca = predict_pca(data_path)
        
        # Convert string predictions to numeric
        if 'Phishing' in results_pca['Prediction'].values:
            numeric_preds_pca = results_pca['Prediction'].map({'Legitimate': 0, 'Phishing': 1})
        else:
            numeric_preds_pca = results_pca['Prediction']
        
        # Calculate accuracy
        accuracy_pca = (true_labels == numeric_preds_pca).mean()
        print(f"pca.py Prediction distribution: {results_pca['Prediction'].value_counts().to_dict()}")
        print(f"pca.py Accuracy: {accuracy_pca:.4f}")
        
    except Exception as e:
        print(f"Error testing pca.py: {e}")
        import traceback
        traceback.print_exc()
    
    # Test notebook predictions
    print("\n--- Testing notebook prediction function ---")
    try:
        # Add notebook path to system path
        notebook_path = os.path.abspath("Notebooks/Python")
        if notebook_path not in sys.path:
            sys.path.append(notebook_path)
        
        # Try to import the standalone prediction function
        try:
            from pca_prediction import predict_with_pca_model as predict_notebook
            
            # Set notebook model paths
            pca_model_path = 'pca_results/pca_transformer_plotly.joblib'
            lr_model_path = 'pca_results/logistic_regression_model_plotly.joblib'
            if not os.path.exists(pca_model_path):
                pca_model_path = 'Notebooks/pca_results/pca_transformer_plotly.joblib'
            if not os.path.exists(lr_model_path):
                lr_model_path = 'Notebooks/pca_results/logistic_regression_model_plotly.joblib'
            
            print("Running predictions with notebook function...")
            results_notebook = predict_notebook(data_path, 
                                               pca_model_path=pca_model_path,
                                               lr_model_path=lr_model_path)
            
            # Convert string predictions to numeric
            if 'Phishing' in results_notebook['Prediction'].values:
                numeric_preds_notebook = results_notebook['Prediction'].map({'Legitimate': 0, 'Phishing': 1})
            else:
                numeric_preds_notebook = results_notebook['Prediction']
            
            # Calculate accuracy
            accuracy_notebook = (true_labels == numeric_preds_notebook).mean()
            print(f"Notebook Prediction distribution: {results_notebook['Prediction'].value_counts().to_dict()}")
            print(f"Notebook Accuracy: {accuracy_notebook:.4f}")
            
        except ImportError as e:
            print(f"Failed to import from notebook: {e}")
            print("The notebook code has been updated, but we can't test it directly here.")
        
    except Exception as e:
        print(f"Error testing notebook code: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_predictions() 