import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the phishing dataset for machine learning.
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Separate features and target
    X = df.drop(['label', 'FILENAME', 'URL', 'Domain', 'Title'], axis=1)
    y = df['label']
    
    # Convert boolean columns to int
    boolean_columns = X.select_dtypes(include=['bool']).columns
    X[boolean_columns] = X[boolean_columns].astype(int)
    
    # Handle categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert back to DataFrame to maintain column names
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y

def prepare_inference_data(file_path, scaler=None, label_encoders=None):
    """
    Prepare new data for inference using the same preprocessing steps.
    """
    # Load the data
    df = pd.read_csv(file_path)
    
    # Separate features
    X = df.drop(['FILENAME', 'URL', 'Domain', 'Title'], axis=1, errors='ignore')
    
    # Convert boolean columns to int
    boolean_columns = X.select_dtypes(include=['bool']).columns
    X[boolean_columns] = X[boolean_columns].astype(int)
    
    # Handle categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if label_encoders and col in label_encoders:
            X[col] = label_encoders[col].transform(X[col].astype(str))
        else:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            if label_encoders is not None:
                label_encoders[col] = le
    
    # Scale numerical features
    if scaler:
        X_scaled = scaler.transform(X)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    
    # Convert back to DataFrame
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, scaler, label_encoders

def save_preprocessing_artifacts(scaler, label_encoders, output_dir='artifacts'):
    """
    Save preprocessing artifacts for later use in inference.
    """
    import joblib
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save scaler
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    
    # Save label encoders
    for col, le in label_encoders.items():
        joblib.dump(le, os.path.join(output_dir, f'label_encoder_{col}.joblib'))

def load_preprocessing_artifacts(input_dir='artifacts'):
    """
    Load preprocessing artifacts for inference.
    """
    import joblib
    import os
    
    # Load scaler
    scaler = joblib.load(os.path.join(input_dir, 'scaler.joblib'))
    
    # Load label encoders
    label_encoders = {}
    for file in os.listdir(input_dir):
        if file.startswith('label_encoder_'):
            col = file.replace('label_encoder_', '').replace('.joblib', '')
            label_encoders[col] = joblib.load(os.path.join(input_dir, file))
    
    return scaler, label_encoders

if __name__ == "__main__":
    # Example usage
    file_path = 'Data/PhiUSIIL_Phishing_URL_Dataset.csv'
    
    # Prepare training data
    X_scaled, y = load_and_preprocess_data(file_path)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print("Training data shape:", X_train.shape)
    print("Test data shape:", X_test.shape)
    
    # Example of preparing new data for inference
    X_new, scaler, label_encoders = prepare_inference_data(file_path)
    
    # Save preprocessing artifacts
    save_preprocessing_artifacts(scaler, label_encoders)
    
    # Example of loading artifacts and using them for inference
    loaded_scaler, loaded_label_encoders = load_preprocessing_artifacts()
    X_inference, _, _ = prepare_inference_data(
        file_path, 
        scaler=loaded_scaler, 
        label_encoders=loaded_label_encoders
    ) 