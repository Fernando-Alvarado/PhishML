import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    """
    Load the phishing dataset
    """
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Remove non-feature columns
    features = df.drop(['label', 'FILENAME', 'URL', 'Domain', 'Title'], axis=1)
    
    # Convert boolean columns to int
    boolean_columns = features.select_dtypes(include=['bool']).columns
    features[boolean_columns] = features[boolean_columns].astype(int)
    
    # Convert all object columns to string
    object_columns = features.select_dtypes(include=['object']).columns
    for col in object_columns:
        features[col] = features[col].astype(str)
    
    # Add the label column back
    features['label'] = df['label']
    
    return features

def train_autogluon(data, label_column='label', time_limit=600):
    """
    Train models using AutoGluon
    """
    logger.info("Splitting data into train and test sets")
    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data[label_column]
    )
    
    logger.info(f"Training data shape: {train_data.shape}")
    logger.info(f"Test data shape: {test_data.shape}")
    
    # Create save path
    save_path = 'autogluon_models'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    logger.info(f"Training AutoGluon models with time limit: {time_limit} seconds")
    start_time = time.time()
    
    # Initialize and train the predictor
    predictor = TabularPredictor(
        label=label_column,
        path=save_path,
        eval_metric='roc_auc'
    ).fit(
        train_data,
        time_limit=time_limit,
        presets='best_quality'
    )
    
    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.2f} seconds")
    
    # Evaluate on test data
    logger.info("Evaluating models on test data")
    test_performance = predictor.evaluate(test_data)
    
    # Leaderboard
    leaderboard = predictor.leaderboard(test_data)
    
    # Feature importance
    feature_importance = None
    try:
        model_to_use = predictor.get_model_best()
        feature_importance = predictor.feature_importance(model=model_to_use)
    except:
        logger.warning("Could not calculate feature importance")
    
    return {
        'predictor': predictor,
        'leaderboard': leaderboard,
        'test_performance': test_performance,
        'feature_importance': feature_importance,
        'train_time': train_time
    }

def save_results(results, output_file='results.txt'):
    """
    Save the results to a text file
    """
    with open(output_file, 'w') as f:
        f.write("# AutoGluon Model Training Results\n\n")
        
        f.write(f"## Training Time\n")
        f.write(f"Total training time: {results['train_time']:.2f} seconds\n\n")
        
        f.write(f"## Test Performance\n")
        for metric, value in results['test_performance'].items():
            f.write(f"{metric}: {value}\n")
        f.write("\n")
        
        f.write("## Model Leaderboard\n")
        f.write(results['leaderboard'].to_string() + "\n\n")
        
        if results['feature_importance'] is not None:
            f.write("## Feature Importance (Top 20)\n")
            f.write(results['feature_importance'].head(20).to_string() + "\n")
    
    logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    # Path to dataset
    file_path = 'Data/Raw/PhiUSIIL_Phishing_URL_Dataset.csv'
    
    # Load data
    data = load_data(file_path)
    
    # Train models with AutoGluon
    # Set time_limit to control how long AutoGluon trains (in seconds)
    # For a large dataset, you might want to increase this
    results = train_autogluon(data, time_limit=1200)  # 20 minutes
    
    # Save results to file
    save_results(results, 'results.txt')
    
    logger.info("Process completed successfully") 