import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import joblib
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from preprocess import load_and_preprocess_data

def create_and_export_plotly_visuals():
    """
    Recreate the PCA analysis visualizations using Plotly and export them as interactive HTML
    """
    # Set up output directory
    output_dir = "Notebooks/HTML/interactive_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    file_path = 'Data/Raw/PhiUSIIL_Phishing_URL_Dataset.csv'
    if not os.path.exists(file_path):
        file_path = 'Data/PhiUSIIL_Phishing_URL_Dataset.csv'
        if not os.path.exists(file_path):
            print("Data file not found")
            return
    
    X_scaled, y = load_and_preprocess_data(file_path)
    
    # Load PCA transformer and model from files if they exist
    pca_path = 'pca_results/pca_transformer.joblib'
    if not os.path.exists(pca_path):
        pca_path = 'Notebooks/pca_results/pca_transformer_plotly.joblib'
    
    if os.path.exists(pca_path):
        print(f"Loading PCA transformer from {pca_path}")
        pca = joblib.load(pca_path)
        X_pca = pca.transform(X_scaled)
    else:
        print("PCA transformer not found, creating new one...")
        # Apply PCA with 75% variance threshold
        pca = PCA()
        pca.fit(X_scaled)
        
        # Determine number of components based on explained variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= 0.75) + 1
        print(f"Selected {n_components} components to explain 75.0% of variance")
        
        # Refit PCA with the determined number of components
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
    
    print(f"PCA components: {pca.n_components_}")
    
    # Create and export plots
    html_files = []
    
    # 1. Explained Variance Plot
    print("Creating Explained Variance plot...")
    fig_variance = create_explained_variance_plot(pca)
    variance_html = os.path.join(output_dir, "explained_variance.html")
    fig_variance.write_html(variance_html, full_html=True, include_plotlyjs='cdn')
    html_files.append(variance_html)
    
    # 2. Feature Contributions Plot
    print("Creating Feature Contributions plot...")
    figs_contributions = create_feature_contributions_plots(pca, X_scaled.columns)
    contribution_htmls = []
    for i, fig in enumerate(figs_contributions):
        contribution_html = os.path.join(output_dir, f"feature_contributions_pc{i+1}.html")
        fig.write_html(contribution_html, full_html=True, include_plotlyjs='cdn')
        contribution_htmls.append(contribution_html)
        html_files.append(contribution_html)
    
    # 3. PCA 2D Scatter Plot
    print("Creating PCA 2D Scatter plot...")
    fig_2d = create_pca_2d_plot(X_pca, y, pca, X_scaled.columns)
    scatter_html = os.path.join(output_dir, "pca_2d_scatter.html")
    fig_2d.write_html(scatter_html, full_html=True, include_plotlyjs='cdn')
    html_files.append(scatter_html)
    
    # 4. ROC Curve (if model exists)
    model_path = 'pca_results/logistic_regression_model.joblib'
    if not os.path.exists(model_path):
        model_path = 'Notebooks/pca_results/logistic_regression_model_plotly.joblib'
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path} for ROC curve...")
        model = joblib.load(model_path)
        
        # Split data for fair evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_pca, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Make predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Create ROC curve
        fig_roc = create_roc_curve_plot(y_test, y_pred_proba)
        roc_html = os.path.join(output_dir, "roc_curve.html")
        fig_roc.write_html(roc_html, full_html=True, include_plotlyjs='cdn')
        html_files.append(roc_html)
    
    # Create index HTML that links to all plots
    create_index_html(html_files, output_dir)
    
    print(f"\nInteractive Plotly visualizations have been created in: {output_dir}")
    print(f"Open {os.path.join(output_dir, 'index.html')} in your web browser to view them")

def create_explained_variance_plot(pca):
    """Create interactive explained variance plot with Plotly"""
    # Create data for the plot
    components = list(range(1, len(pca.explained_variance_ratio_) + 1))
    variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_ratio)
    
    # Create a figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar chart for individual explained variance
    fig.add_trace(
        go.Bar(
            x=components, 
            y=variance_ratio, 
            name="Explained Variance",
            marker_color='blue'
        ),
        secondary_y=False,
    )
    
    # Add line chart for cumulative explained variance
    fig.add_trace(
        go.Scatter(
            x=components, 
            y=cumulative_variance, 
            name="Cumulative Variance", 
            line=dict(color='red', width=2)
        ),
        secondary_y=True,
    )
    
    # Set titles and labels
    fig.update_layout(
        title="Explained Variance by Principal Component",
        title_font_size=20,
        xaxis_title="Principal Component",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        width=900,
        height=600
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Explained Variance Ratio", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Explained Variance", secondary_y=True)
    
    # Add horizontal line at common variance thresholds
    for threshold in [0.7, 0.8, 0.9, 0.95]:
        fig.add_hline(
            y=threshold, 
            line=dict(color="green", width=1, dash="dash"), 
            annotation_text=f"{threshold*100}%", 
            annotation_position="right",
            secondary_y=True
        )
    
    return fig

def create_feature_contributions_plots(pca, feature_names, n_top_features=10):
    """Create interactive feature contributions plots with Plotly"""
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
            marker_color=colors,
            text=[f"{x:.4f}" for x in contributions],
            textposition='outside'
        ))
        
        # Set titles and labels
        fig.update_layout(
            title=f"Top Features Contributing to Principal Component {i+1}",
            title_font_size=18,
            xaxis_title="Component Coefficient",
            yaxis_title="Feature",
            yaxis=dict(autorange="reversed"),  # Reverse y-axis to show highest values at top
            width=900,
            height=600,
            margin=dict(l=200)  # Add more margin for feature names
        )
        
        figures.append(fig)
    
    return figures

def create_pca_2d_plot(X_pca, y, pca, feature_names):
    """Create interactive 2D PCA plot with Plotly"""
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
    
    # If there are enough components, add feature vectors
    if len(pca.components_) >= 2:
        # Get feature loadings for PC1 and PC2
        loadings = pca.components_.T[:, 0:2]
        
        # Scale the feature vectors for visibility
        scale_factor = 3  # Adjust to make vectors visible
        
        # Add each feature vector as an annotation and arrow
        for i, feature in enumerate(feature_names):
            # Skip features with small contributions to avoid clutter
            if np.sqrt(loadings[i, 0]**2 + loadings[i, 1]**2) < 0.1:
                continue
                
            # Add arrow
            fig.add_annotation(
                x=0, y=0,
                ax=loadings[i, 0] * scale_factor,
                ay=loadings[i, 1] * scale_factor,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="green"
            )
            
            # Add text label
            fig.add_annotation(
                x=loadings[i, 0] * scale_factor * 1.1,
                y=loadings[i, 1] * scale_factor * 1.1,
                text=feature,
                showarrow=False,
                font=dict(size=8)
            )
    
    # Update layout
    fig.update_layout(
        legend_title="Class",
        height=700, 
        width=900,
        title_font_size=18
    )
    
    return fig

def create_roc_curve_plot(y_test, y_pred_proba):
    """Create interactive ROC curve plot with Plotly"""
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
        title_font_size=18,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(x=0.7, y=0.1),
        width=800,
        height=600,
        xaxis=dict(range=[0, 1], constrain='domain'),
        yaxis=dict(range=[0, 1], constrain='domain'),
        showlegend=True
    )
    
    return fig

def create_index_html(html_files, output_dir):
    """Create an index HTML file linking to all plots"""
    index_path = os.path.join(output_dir, "index.html")
    
    # Extract just the filenames from the paths
    relative_files = [os.path.basename(file) for file in html_files]
    
    # Create HTML content
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PCA Analysis Visualizations</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 40px;
                line-height: 1.6;
            }
            h1, h2 {
                color: #2c3e50;
            }
            .container {
                max-width: 900px;
                margin: 0 auto;
            }
            ul {
                padding-left: 20px;
            }
            li {
                margin-bottom: 10px;
            }
            a {
                color: #3498db;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
            .plot-container {
                margin-top: 30px;
                margin-bottom: 50px;
                border-bottom: 1px solid #eee;
                padding-bottom: 20px;
            }
            .description {
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>PCA Analysis Visualizations for Phishing Detection</h1>
            <p>This page contains interactive Plotly visualizations from the PCA analysis of the phishing dataset.</p>
            
            <h2>Available Visualizations:</h2>
            <ul>
    """
    
    # Add links to all plots
    for filename in relative_files:
        name = filename.replace('.html', '').replace('_', ' ').title()
        html_content += f'        <li><a href="{filename}" target="_blank">{name}</a></li>\n'
    
    html_content += """
            </ul>
            
            <div class="plot-container">
                <h2>Explained Variance</h2>
                <div class="description">
                    <p>This plot shows how much variance is explained by each principal component. 
                    The bar chart represents individual contribution, while the red line shows cumulative explained variance.</p>
                </div>
                <iframe src="explained_variance.html" width="100%" height="600px" frameborder="0"></iframe>
            </div>
    """
    
    # Add feature contribution sections
    for i, html_file in enumerate([f for f in relative_files if 'feature_contributions' in f]):
        html_content += f"""
            <div class="plot-container">
                <h2>Feature Contributions to PC{i+1}</h2>
                <div class="description">
                    <p>This plot shows which features contribute most to Principal Component {i+1}. 
                    Positive values (blue) and negative values (red) indicate the direction and strength of contribution.</p>
                </div>
                <iframe src="{html_file}" width="100%" height="600px" frameborder="0"></iframe>
            </div>
        """
    
    # Add 2D scatter plot
    if 'pca_2d_scatter.html' in relative_files:
        html_content += """
            <div class="plot-container">
                <h2>PCA 2D Scatter Plot</h2>
                <div class="description">
                    <p>This scatter plot shows the first two principal components with class labels. 
                    Blue points represent legitimate websites, while red points represent phishing websites.</p>
                    <p>The green arrows represent feature vectors, showing how features contribute to the principal components.</p>
                </div>
                <iframe src="pca_2d_scatter.html" width="100%" height="700px" frameborder="0"></iframe>
            </div>
        """
    
    # Add ROC curve if available
    if 'roc_curve.html' in relative_files:
        html_content += """
            <div class="plot-container">
                <h2>ROC Curve</h2>
                <div class="description">
                    <p>The ROC curve shows the performance of the logistic regression model trained on the PCA-transformed features.
                    The AUC (Area Under Curve) is a measure of the model's ability to distinguish between classes.</p>
                </div>
                <iframe src="roc_curve.html" width="100%" height="600px" frameborder="0"></iframe>
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    print(f"Created index file: {index_path}")

if __name__ == "__main__":
    create_and_export_plotly_visuals() 