from ydata_profiling import ProfileReport

import duckdb
import polars as pl

# Read the CSV file using DuckDB
con = duckdb.connect(database=':memory:')
con.execute("""
    CREATE TABLE phishing_data AS 
    SELECT * FROM read_csv_auto('Data/PhiUSIIL_Phishing_URL_Dataset.csv')
""")

# Convert to Polars DataFrame
df = pl.from_arrow(con.execute("SELECT * FROM phishing_data").arrow())
# Convert Polars DataFrame to Pandas DataFrame
df_pandas = df.to_pandas()

# Generate comprehensive summary statistics
profile = ProfileReport(df_pandas, title="Phishing Dataset Analysis", explorative=True)

# Save the report
profile.to_file("phishing_analysis_report.html")

# Basic summary statistics
print("\nBasic Summary Statistics:")
print(df.describe())

# Column information
print("\nColumn Information:")
print(df.schema)

# Value counts for categorical columns
categorical_cols = df.select(pl.col("^.*$").filter(pl.col.dtype == pl.Categorical))
for col in categorical_cols.columns:
    print(f"\nValue counts for {col}:")
    print(df[col].value_counts())

# Create visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn')

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Distribution of target variable
sns.countplot(data=df_pandas, x='target', ax=axes[0,0])
axes[0,0].set_title('Distribution of Target Variable')

# Plot 2: Correlation heatmap
numeric_cols = df.select(pl.col("^.*$").filter(pl.col.dtype.is_numeric()))
correlation = numeric_cols.to_pandas().corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=axes[0,1])
axes[0,1].set_title('Correlation Heatmap')

# Plot 3: Box plots for numeric features
numeric_cols.to_pandas().boxplot(ax=axes[1,0])
axes[1,0].set_title('Box Plots of Numeric Features')
plt.xticks(rotation=45)

# Plot 4: Pair plot for selected numeric features
sns.pairplot(numeric_cols.to_pandas().sample(n=1000), ax=axes[1,1])
axes[1,1].set_title('Pair Plot of Numeric Features')

plt.tight_layout()
plt.savefig('phishing_analysis_plots.png')
plt.close()

# Print column descriptions
print("\nColumn Descriptions:")
for col in df.columns:
    print(f"\n{col}:")
    print(f"Type: {df[col].dtype}")
    print(f"Missing values: {df[col].null_count()}")
    print(f"Unique values: {df[col].n_unique()}")

