import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from preprocess import load_and_preprocess_data, prepare_inference_data, load_preprocessing_artifacts, save_preprocessing_artifacts
from sklearn.preprocessing import StandardScaler, LabelEncoder

def apply_pca(X_scaled, n_components=None, variance_threshold=0.95):
    """
    Apply PCA to the scaled features.
    
    Parameters:
    - X_scaled: Scaled feature matrix
    - n_components: Number of components to keep (if None, use variance_threshold)
    - variance_threshold: Threshold of explained variance ratio to determine components
    
    Returns:
    - X_pca: Transformed features
    - pca: Fitted PCA object
    """
    if n_components is None:
        # Initialize PCA without specifying number of components
        pca = PCA()
        pca.fit(X_scaled)
        
        # Determine number of components based on explained variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        print(f"Selected {n_components} components to explain {variance_threshold*100:.1f}% of variance")
        
        # Refit PCA with the determined number of components
        pca = PCA(n_components=n_components)
    else:
        pca = PCA(n_components=n_components)
    
    # Transform the data
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca, pca

def plot_explained_variance(pca, output_dir='plots'):
    """
    Plot the explained variance ratio of PCA components.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot explained variance ratio
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             np.cumsum(pca.explained_variance_ratio_), 'r-')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Component')
    plt.savefig(os.path.join(output_dir, 'pca_explained_variance.png'))
    plt.close()

def plot_feature_contributions(pca, feature_names, n_top_features=10, output_dir='plots'):
    """
    Plot the feature contributions to principal components.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot top feature contributions to first two components
    n_components = min(2, pca.n_components_)
    
    for i in range(n_components):
        component = pca.components_[i]
        # Get indices of top contributing features (both positive and negative)
        top_positive_idx = np.argsort(component)[-n_top_features:]
        top_negative_idx = np.argsort(component)[:n_top_features]
        
        plt.figure(figsize=(12, 8))
        # Plot positive contributions
        plt.barh(range(n_top_features), component[top_positive_idx], color='blue')
        plt.yticks(range(n_top_features), [feature_names[idx] for idx in top_positive_idx])
        
        # Plot negative contributions
        plt.barh(range(n_top_features, 2*n_top_features), component[top_negative_idx], color='red')
        plt.yticks(range(n_top_features, 2*n_top_features), [feature_names[idx] for idx in top_negative_idx])
        
        plt.axvline(x=0, color='black', linestyle='-')
        plt.title(f'Top Features Contributing to PC{i+1}')
        plt.xlabel('Component Coefficient')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pca_component_{i+1}_features.png'))
        plt.close()

def plot_pca_2d(X_pca, y, output_dir='plots'):
    """
    Plot the first two PCA components with class labels.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if X_pca.shape[1] < 2:
        print("Not enough components for 2D visualization")
        return
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, alpha=0.5, 
               cmap='coolwarm', edgecolors='k', s=40)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA: First Two Principal Components')
    plt.colorbar(scatter, label='Class')
    plt.savefig(os.path.join(output_dir, 'pca_2d_visualization.png'))
    plt.close()

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Train a logistic regression model on PCA-transformed data.
    
    Returns:
    - model: Trained logistic regression model
    - performance: Dictionary with performance metrics
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    performance = {
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return model, performance

def plot_roc_curve(y_test, y_pred_proba, output_dir='plots'):
    """
    Plot the ROC curve for the logistic regression model.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')  # Random classifier line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Logistic Regression on PCA Features')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, 'logistic_regression_roc.png'))
    plt.close()

def plot_confusion_matrix(confusion_mat, output_dir='plots'):
    """
    Plot the confusion matrix.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Legitimate', 'Phishing'],
               yticklabels=['Legitimate', 'Phishing'])
    plt.title('Confusion Matrix - Logistic Regression on PCA Features')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'logistic_regression_confusion_matrix.png'))
    plt.close()

def save_results(model, pca, performance, output_dir='pca_results'):
    """
    Save model, PCA transformer, and performance metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and PCA transformer
    joblib.dump(model, os.path.join(output_dir, 'logistic_regression_model.joblib'))
    joblib.dump(pca, os.path.join(output_dir, 'pca_transformer.joblib'))
    
    # Save performance metrics to text file
    with open(os.path.join(output_dir, 'performance_metrics.txt'), 'w') as f:
        f.write("# Logistic Regression on PCA Features - Performance Metrics\n\n")
        f.write("## Classification Report\n")
        f.write(performance['classification_report'])
        f.write("\n\n## ROC AUC Score\n")
        f.write(f"ROC AUC: {performance['roc_auc']:.3f}\n")

def predict_with_pca_model(data_path, pca_model_path='pca_results/pca_transformer.joblib', 
                          lr_model_path='pca_results/logistic_regression_model.joblib'):
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
    
    # Preprocess the data in the same way as training data
    X = original_data.drop(['label', 'FILENAME', 'URL', 'Domain', 'Title'], axis=1, errors='ignore')
    
    # Convert boolean columns to int
    boolean_columns = X.select_dtypes(include=['bool']).columns
    X[boolean_columns] = X[boolean_columns].astype(int)
    
    # Load preprocessing artifacts
    scaler, label_encoders = load_preprocessing_artifacts()
    
    # Handle categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col in label_encoders:
            X[col] = label_encoders[col].transform(X[col].astype(str))
        else:
            # If a column wasn't seen during training, use a new encoder
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Scale the data
    X_scaled = scaler.transform(X)
    
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

if __name__ == "__main__":
    # Create output directories
    os.makedirs('plots', exist_ok=True)
    os.makedirs('pca_results', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)
    
    # Load and preprocess data
    file_path = 'Data/Raw/PhiUSIIL_Phishing_URL_Dataset.csv'
    X_scaled, y = load_and_preprocess_data(file_path)
    
    # Save preprocessing artifacts for later use
    # Create a scaler and fit it to the data
    scaler = StandardScaler()
    scaler.fit(X_scaled)
    
    # Create empty label encoders dictionary
    label_encoders = {}
    
    # Read the original data to get categorical columns
    df = pd.read_csv(file_path)
    X = df.drop(['label', 'FILENAME', 'URL', 'Domain', 'Title'], axis=1)
    categorical_columns = X.select_dtypes(include=['object']).columns
    
    # Create label encoders for each categorical column
    for col in categorical_columns:
        le = LabelEncoder()
        le.fit(X[col].astype(str))
        label_encoders[col] = le
    
    # Save the preprocessing artifacts
    save_preprocessing_artifacts(scaler, label_encoders)
    
    print("Original data shape:", X_scaled.shape)
    
    # Apply PCA (automatically determine number of components)
    X_pca, pca = apply_pca(X_scaled, variance_threshold=0.75)
    
    print("PCA transformed data shape:", X_pca.shape)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Plot PCA results
    plot_explained_variance(pca)
    plot_feature_contributions(pca, X_scaled.columns)
    plot_pca_2d(X_pca, y)
    
    # Write top feature contributions to file
    with open('pca_results/top_features.txt', 'w') as f:
        f.write("# Top Features Contributing to Principal Components\n\n")
        for i in range(min(3, pca.n_components_)):
            component = pca.components_[i]
            # Get indices of top contributing features
            top_indices = np.argsort(np.abs(component))[::-1][:10]
            
            f.write(f"## Principal Component {i+1}\n")
            for idx in top_indices:
                feature = X_scaled.columns[idx]
                contribution = component[idx]
                f.write(f"{feature}: {contribution:.4f}\n")
            f.write("\n")
    
    # Split data for training
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train logistic regression model
    model, performance = train_logistic_regression(X_train, X_test, y_train, y_test)
    
    # Plot model performance
    plot_roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plot_confusion_matrix(performance['confusion_matrix'])
    
    # Save results
    save_results(model, pca, performance)
    
    print("\nPCA analysis and Logistic Regression completed!")
    print(f"Explained variance with {pca.n_components_} components: {np.sum(pca.explained_variance_ratio_):.3f}")
    print(f"Logistic Regression ROC AUC: {performance['roc_auc']:.3f}")
    print("\nResults saved to pca_results/ directory")
    
    # Example of making predictions with the model
    print("\nExample prediction with the model:")
    test_file = 'Data/Raw/PhiUSIIL_Phishing_URL_Dataset.csv'  # For demonstration, using same file
    results = predict_with_pca_model(test_file)
    print(f"Predictions summary: {results['Prediction'].value_counts().to_dict()}")
    print("Sample predictions:")
    print(results.head(5)) 