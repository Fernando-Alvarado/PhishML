import os
import subprocess
import sys
import shutil

def convert_notebook_to_interactive_html():
    """
    Convert the notebook to HTML with fully interactive Plotly visualizations using a custom template
    """
    notebook_path = "Notebooks/Python/pca_analysis.ipynb"
    output_dir = "Notebooks/HTML"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # First, ensure the notebook is using the Plotly renderer that outputs interactive HTML
    setup_cmd = [
        "python", "-c", 
        "import plotly.io as pio; pio.renderers.default = 'notebook'; print('Plotly renderer set to notebook')"
    ]
    
    try:
        subprocess.run(setup_cmd, check=True)
        print("Plotly renderer configured for HTML output")
    except subprocess.CalledProcessError:
        print("Warning: Failed to configure Plotly renderer")
    
    # Command to convert notebook to HTML with embedded outputs
    # The --template=lab option provides better Plotly support
    cmd = [
        "jupyter", "nbconvert", 
        "--to", "html", 
        "--execute",  # Execute the notebook before converting
        "--template=lab",  # Use the lab template for better JavaScript support
        "--ExecutePreprocessor.timeout=600",  # 10-minute timeout for execution
        "--output-dir", output_dir,
        notebook_path
    ]
    
    print(f"Converting notebook: {notebook_path}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the conversion command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            output_file = os.path.join(output_dir, os.path.basename(notebook_path).replace('.ipynb', '.html'))
            print(f"Conversion successful! Output file: {output_file}")
            
            # Print instructions for viewing the HTML file
            print("\nTo view the HTML file with interactive Plotly visualizations:")
            print(f"1. Open {output_file} in your web browser")
            print("2. The Plotly visualizations should be embedded and interactive")
            return output_file
        else:
            print(f"Error converting notebook: {result.stderr}")
            
            # Try a different template if the first one fails
            print("\nTrying with classic template...")
            classic_cmd = [
                "jupyter", "nbconvert", 
                "--to", "html", 
                "--execute",
                "--template=classic",
                "--ExecutePreprocessor.timeout=600",
                "--output-dir", output_dir,
                notebook_path
            ]
            classic_result = subprocess.run(classic_cmd, capture_output=True, text=True)
            
            if classic_result.returncode == 0:
                output_file = os.path.join(output_dir, os.path.basename(notebook_path).replace('.ipynb', '.html'))
                print(f"Classic template conversion successful! Output file: {output_file}")
                print("\nThe Plotly visualizations should be interactive.")
                return output_file
            else:
                print(f"Error in classic template conversion: {classic_result.stderr}")
                
                # Finally, try a simple conversion without execution
                print("\nTrying simple conversion without execution...")
                simple_cmd = ["jupyter", "nbconvert", "--to", "html", "--output-dir", output_dir, notebook_path]
                simple_result = subprocess.run(simple_cmd, capture_output=True, text=True)
                
                if simple_result.returncode == 0:
                    output_file = os.path.join(output_dir, os.path.basename(notebook_path).replace('.ipynb', '.html'))
                    print(f"Simple conversion successful! Output file: {output_file}")
                    print("\nNote: Plotly visualizations may not be interactive in this version.")
                    return output_file
                else:
                    print(f"Error in simple conversion: {simple_result.stderr}")
    
    except Exception as e:
        print(f"Exception occurred: {e}")
    
    return None

if __name__ == "__main__":
    html_file = convert_notebook_to_interactive_html() 