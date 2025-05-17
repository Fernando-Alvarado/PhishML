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

# Get summary statistics for numeric columns
numeric_summary = df.describe()
print("\nNumeric Columns Summary Statistics:")
print(numeric_summary)

# Get value counts for categorical columns
categorical_columns = ['IsDomainIP', 'IsHTTPS', 'HasTitle', 'HasFavicon', 'Robots', 
                      'IsResponsive', 'HasDescription', 'HasSocialNet', 'HasSubmitButton',
                      'HasHiddenFields', 'HasPasswordField', 'Bank', 'Pay', 'Crypto',
                      'HasCopyrightInfo', 'label']

print("\nCategorical Columns Value Counts:")
for col in categorical_columns:
    print(f"\n{col} value counts:")
    print(df[col].value_counts())

# Close the DuckDB connection
con.close()

