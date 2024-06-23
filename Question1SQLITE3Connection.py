import sqlite3
import pandas as pd

# Path to the uploaded CSV file
csv_file_path = (r'TheGamBoZZo/heart_prediction/heart.csv')

# Connect to the SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('heartDB.db')

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path, delimiter=';')

# Write the DataFrame to a new table in the SQLite database
table_name = 'heart_data'
df.to_sql(table_name, conn, if_exists='replace', index=False)

# Example query to select all rows from the table
cursor = conn.cursor()
cursor.execute(f'SELECT * FROM {table_name}')
rows = cursor.fetchall()

# Check for missing values
df.isnull().sum()

# If there are missing values, you can fill them or drop rows/columns
df = df.dropna()  # Example of dropping rows with missing values


# Print the fetched rows
for row in rows:
    print(row)

# Close the cursor and connection
cursor.close()
conn.close()

