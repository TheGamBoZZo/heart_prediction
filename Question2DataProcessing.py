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

# Display the first few rows of the DataFrame
print(df.head())

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values:\n", missing_values)

# Handle missing values if any (example: fill with mean or drop)
df.fillna(df.mean(), inplace=True)

# Check for duplicate records
duplicates = df.duplicated().sum()
print("Duplicates:\n", duplicates)

# Remove duplicates if any
df.drop_duplicates(inplace=True)

# Convert categorical columns to the 'category' data type if needed
df['sex'] = df['sex'].astype('category')

# Normalize or standardize numerical features if necessary
scaler = StandardScaler()
df[['age']] = scaler.fit_transform(df[['age']])

# Display basic statistics
print(df.describe())

# Visualize the distribution of age
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], bins=30, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Visualize the relationship between age and cholesterol
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='chol', data=df)
plt.title('Age vs Cholesterol')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.show()

# Visualize the count of different target values
plt.figure(figsize=(10, 6))
sns.countplot(x='target', data=df)
plt.title('Count of Target Values')
plt.xlabel('Target')
plt.ylabel('Count')
plt.show()
