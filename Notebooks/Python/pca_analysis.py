"""
# PCA Analysis of Phishing Dataset with Plotly Visualizations

This script performs Principal Component Analysis (PCA) on the phishing dataset 
and trains a logistic regression model on the reduced features. It includes 
Plotly visualizations to explore the components and model performance.
"""

# %% [markdown]
# ## Import Libraries

# %%
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import joblib
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys

# Add parent directory to path to import from preprocess.py
sys.path.append('..')
try:
    from preprocess import load_and_preprocess_data, prepare_inference_data, load_preprocessing_artifacts, save_preprocessing_artifacts
except ImportError:
    print("Could not import from preprocess.py. Will define necessary functions here.")

# %% [markdown]
# ## Define PCA Analysis Functions

# %%
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

# %%
def plot_explained_variance_plotly(pca):
    """
    Plot the explained variance ratio of PCA components using Plotly.
    """
    # Create data for the plot
    components = list(range(1, len(pca.explained_variance_ratio_) + 1))
    variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_ratio)
    
    # Create a figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar chart for individual explained variance
    fig.add_trace(
        go.Bar(x=components, y=variance_ratio, name="Explained Variance"),
        secondary_y=False,
    )
    
    # Add line chart for cumulative explained variance
    fig.add_trace(
        go.Scatter(x=components, y=cumulative_variance, name="Cumulative Variance", line=dict(color='red')),
        secondary_y=True,
    )
    
    # Set titles and labels
    fig.update_layout(
        title_text="Explained Variance by Principal Component",
        xaxis_title="Principal Component",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Explained Variance Ratio", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Explained Variance", secondary_y=True)
    
    # Add horizontal line at common variance thresholds
    for threshold in [0.7, 0.8, 0.9, 0.95]:
        fig.add_hline(y=threshold, line=dict(color="green", width=1, dash="dash"), 
                     annotation_text=f"{threshold*100}%", annotation_position="right",
                     secondary_y=True)
    
    return fig

# %%
def plot_feature_contributions_plotly(pca, feature_names, n_top_features=10):
    """
    Plot the feature contributions to principal components using Plotly.
    """
    # Create figures for top components
    n_components = min(3, pca.n_components_)
    figures = []
    
    for i in range(n_components):
        component = pca.components_[i]
        
        # Get indices of top contributing features (by absolute value)
        top_indices = np.argsort(np.abs(component))[::-1][:n_top_features]
        
        # Extract feature names and contribution values
        features = [feature_names[idx] for idx in top_indices]
        contributions = component[top_indices]
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        # Add bars colored by contribution direction (positive/negative)
        colors = ['blue' if x >= 0 else 'red' for x in contributions]
        
        fig.add_trace(go.Bar(
            y=features,
            x=contributions,
            orientation='h',
            marker_color=colors
        ))
        
        # Set titles and labels
        fig.update_layout(
            title=f"Top Features Contributing to Principal Component {i+1}",
            xaxis_title="Component Coefficient",
            yaxis_title="Feature",
            yaxis=dict(autorange="reversed"),  # Reverse y-axis to show highest values at top
            height=500
        )
        
        figures.append(fig)
    
    return figures

# %%
def plot_pca_2d_plotly(X_pca, y):
    """
    Plot the first two PCA components with class labels using Plotly.
    """
    if X_pca.shape[1] < 2:
        print("Not enough components for 2D visualization")
        return None
    
    # Create a DataFrame for plotting
    df_plot = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Class': y
    })
    
    # Map class labels to meaningful names
    df_plot['Class_Label'] = df_plot['Class'].map({0: 'Legitimate', 1: 'Phishing'})
    
    # Create scatter plot
    fig = px.scatter(
        df_plot, 
        x='PC1', 
        y='PC2', 
        color='Class_Label',
        color_discrete_map={'Legitimate': 'blue', 'Phishing': 'red'},
        opacity=0.7,
        title="PCA: First Two Principal Components",
        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
        hover_data=['Class_Label']
    )
    
    # Update layout
    fig.update_layout(
        legend_title="Class",
        height=600, 
        width=800
    )
    
    return fig

# %%
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

# %%
def plot_roc_curve_plotly(y_test, y_pred_proba):
    """
    Plot the ROC curve for the logistic regression model using Plotly.
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Create figure
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, 
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {auc:.3f})',
        line=dict(color='blue', width=2)
    ))
    
    # Add random classifier line
    fig.add_trace(go.Scatter(
        x=[0, 1], 
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title='ROC Curve - Logistic Regression on PCA Features',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(x=0.7, y=0.1),
        width=700,
        height=500,
        xaxis=dict(range=[0, 1], constrain='domain'),
        yaxis=dict(range=[0, 1], constrain='domain'),
        showlegend=True
    )
    
    return fig

# %%
def plot_confusion_matrix_plotly(confusion_mat):
    """
    Plot the confusion matrix using Plotly.
    """
    # Define labels
    labels = ['Legitimate', 'Phishing']
    
    # Create annotation text
    annotations = []
    for i in range(len(confusion_mat)):
        for j in range(len(confusion_mat[i])):
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=str(confusion_mat[i, j]),
                    font=dict(color='white' if confusion_mat[i, j] > confusion_mat.max()/2 else 'black', size=14),
                    showarrow=False
                )
            )
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=confusion_mat,
        x=labels,
        y=labels,
        colorscale='Blues',
    ))
    
    # Add annotations
    fig.update_layout(
        title='Confusion Matrix - Logistic Regression on PCA Features',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        xaxis=dict(side='bottom'),
        annotations=annotations,
        width=600,
        height=500
    )
    
    return fig 

# %% [markdown]
# ## Load and Preprocess Data

# %%
# Create output directories
os.makedirs('../plots', exist_ok=True)
os.makedirs('../pca_results', exist_ok=True)
os.makedirs('../artifacts', exist_ok=True)

# Load and preprocess data
file_path = '../Data/Raw/PhiUSIIL_Phishing_URL_Dataset.csv'
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

# %% [markdown]
# ## Apply PCA and Visualize Components 

# %%
# Apply PCA (automatically determine number of components)
variance_threshold = 0.75  # Explain 75% of variance
X_pca, pca = apply_pca(X_scaled, variance_threshold=variance_threshold)

print("PCA transformed data shape:", X_pca.shape)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# %%
# Plot explained variance
fig1 = plot_explained_variance_plotly(pca)
fig1.show()

# %%
# Plot feature contributions
figs_contributions = plot_feature_contributions_plotly(pca, X_scaled.columns)
for i, fig in enumerate(figs_contributions):
    print(f"Top features contributing to PC{i+1}:")
    fig.show()

# %%
# Plot PCA 2D visualization
fig_2d = plot_pca_2d_plotly(X_pca, y)
fig_2d.show()

# %% [markdown]
# ## Train Logistic Regression Model

# %%
# Split data for training
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)

# Train logistic regression model
model, performance = train_logistic_regression(X_train, X_test, y_train, y_test)

print("Classification Report:")
print(performance['classification_report'])
print(f"ROC AUC Score: {performance['roc_auc']:.3f}")

# %%
# Plot ROC curve
fig_roc = plot_roc_curve_plotly(y_test, model.predict_proba(X_test)[:, 1])
fig_roc.show()

# %%
# Plot confusion matrix
fig_cm = plot_confusion_matrix_plotly(performance['confusion_matrix'])
fig_cm.show()

# %% [markdown]
# ## Write Top Features to File

# %%
# Write top feature contributions to file
with open('../pca_results/top_features_plotly.txt', 'w') as f:
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

# %% [markdown]
# ## Save Model and Results

# %%
# Save results
os.makedirs('../pca_results', exist_ok=True)

# Save model and PCA transformer
joblib.dump(model, '../pca_results/logistic_regression_model_plotly.joblib')
joblib.dump(pca, '../pca_results/pca_transformer_plotly.joblib')

# Save performance metrics to text file
with open('../pca_results/performance_metrics_plotly.txt', 'w') as f:
    f.write("# Logistic Regression on PCA Features - Performance Metrics\n\n")
    f.write("## Classification Report\n")
    f.write(performance['classification_report'])
    f.write("\n\n## ROC AUC Score\n")
    f.write(f"ROC AUC: {performance['roc_auc']:.3f}\n")

print("\nPCA analysis and Logistic Regression completed!")
print(f"Explained variance with {pca.n_components_} components: {np.sum(pca.explained_variance_ratio_):.3f}")
print(f"Logistic Regression ROC AUC: {performance['roc_auc']:.3f}")
print("\nResults saved to ../pca_results/ directory")

# %% [markdown]
# ## Make Predictions with the Model

# %%
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

# %%
# Example of making predictions with the model
test_file = '../Data/Raw/PhiUSIIL_Phishing_URL_Dataset.csv'  # For demonstration, using same file
results = predict_with_pca_model(test_file)
print(f"Predictions summary: {results['Prediction'].value_counts().to_dict()}")
print("Sample predictions:")
display(results.head(5))  # Using display for better table formatting in Jupyter 