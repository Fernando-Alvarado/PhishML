import os
import subprocess
import sys

def convert_notebook_to_html():
    """
    Convert the notebook to HTML with embedded plotly visualizations
    """
    notebook_path = "Notebooks/Python/pca_analysis.ipynb"
    output_dir = "Notebooks/HTML"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Command to convert notebook to HTML with embedded outputs
    cmd = [
        "jupyter", "nbconvert", 
        "--to", "html", 
        "--execute",  # Execute the notebook before converting
        "--ExecutePreprocessor.timeout=300",  # 5-minute timeout for execution
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
            print("\nTo view the HTML file:")
            print(f"1. Open {output_file} in your web browser")
            print("2. All Plotly visualizations should be embedded and interactive")
        else:
            print(f"Error converting notebook: {result.stderr}")
            
            # Try a simpler conversion without execution if the first one fails
            print("\nTrying simpler conversion without execution...")
            simple_cmd = ["jupyter", "nbconvert", "--to", "html", "--output-dir", output_dir, notebook_path]
            simple_result = subprocess.run(simple_cmd, capture_output=True, text=True)
            
            if simple_result.returncode == 0:
                output_file = os.path.join(output_dir, os.path.basename(notebook_path).replace('.ipynb', '.html'))
                print(f"Simple conversion successful! Output file: {output_file}")
                print("\nNote: Plotly visualizations may not be interactive in this version.")
            else:
                print(f"Error in simple conversion: {simple_result.stderr}")
    
    except Exception as e:
        print(f"Exception occurred: {e}")

if __name__ == "__main__":
    convert_notebook_to_html() 