import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Connect to the SQLite database
conn = sqlite3.connect('heartDB.db')

# Load the data from the database into a pandas DataFrame
df = pd.read_sql_query("SELECT * FROM heart_data", conn)

# Close the connection
conn.close()
# Convert necessary columns to numeric types
df = df.apply(pd.to_numeric, errors='ignore')

# List of numeric variables
numeric_vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Plot the distribution of numeric variables based on the target variable
plt.figure(figsize=(18, 12))

for i, var in enumerate(numeric_vars, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data=df, x=var, hue='target', kde=True, element='step')
    plt.title(f'Distribution of {var} by Target')
    plt.xlabel(var)
    plt.ylabel('Density')

plt.tight_layout()
plt.show()
