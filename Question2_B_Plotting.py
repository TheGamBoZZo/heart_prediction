import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Connect to the SQLite database
conn = sqlite3.connect('heartDB.db')

# Load the data from the database into a pandas DataFrame
df = pd.read_sql_query("SELECT * FROM heart_data", conn)

# Close the connection
conn.close()

# Convert necessary columns to numeric types
df = df.apply(pd.to_numeric, errors='ignore')

# List of categorical variables
categorical_vars = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Plot the distribution of classes for categorical variables based on the target variable
plt.figure(figsize=(16, 12))

for i, var in enumerate(categorical_vars, 1):
    plt.subplot(3, 3, i)
    sns.countplot(x=var, hue='target', data=df)
    plt.title(f'Distribution of {var} by Target')
    plt.xlabel(var)
    plt.ylabel('Count')

plt.tight_layout()
plt.show()
