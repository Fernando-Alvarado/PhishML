import pandas as pd
import joblib
from preprocess import prepare_inference_data, load_preprocessing_artifacts

def load_model(model_path='models/best_model.joblib'):
    """
    Load the trained model.
    """
    return joblib.load(model_path)

def predict_urls(data_path, model_path='models/best_model.joblib'):
    """
    Make predictions for new URLs.
    """
    # Load model
    model = load_model(model_path)
    
    # Load preprocessing artifacts
    scaler, label_encoders = load_preprocessing_artifacts()
    
    # Prepare data for inference
    X_inference, _, _ = prepare_inference_data(
        data_path,
        scaler=scaler,
        label_encoders=label_encoders
    )
    
    # Make predictions
    predictions = model.predict(X_inference)
    prediction_probas = model.predict_proba(X_inference)[:, 1]
    
    # Load original data to get URLs
    original_data = pd.read_csv(data_path)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'URL': original_data['URL'],
        'Prediction': predictions,
        'Phishing_Probability': prediction_probas
    })
    
    # Map predictions to labels
    results['Prediction'] = results['Prediction'].map({0: 'Legitimate', 1: 'Phishing'})
    
    return results

def save_predictions(results, output_path='predictions.csv'):
    """
    Save predictions to a CSV file.
    """
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    data_path = 'Data/PhiUSIIL_Phishing_URL_Dataset.csv'
    
    # Make predictions
    results = predict_urls(data_path)
    
    # Print summary
    print("\nPrediction Summary:")
    print(results['Prediction'].value_counts())
    print("\nTop 5 Most Suspicious URLs:")
    print(results.sort_values('Phishing_Probability', ascending=False).head())
    
    # Save predictions
    save_predictions(results) 