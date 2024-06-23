import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import sqlite3

csv_file_path = (r'/mount/src/heart_prediction/heart.csv')
# Load the dataset
conn = sqlite3.connect('heartDB.db')

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path, delimiter=';')

# Handle missing values if any
df = df.dropna()

# Convert categorical variables to numeric using one-hot encoding
df = pd.get_dummies(df)

# Split the dataset into features and target
X = df.drop('target', axis=1)
y = df['target']

# Normalize the numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True)  # Enable probability estimates
}

# Train and evaluate models
best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy}')
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# Save the best model and the scaler
joblib.dump(best_model, r'/mount/src/heart_prediction/best_model.pkl')
joblib.dump(scaler, r'/mount/src/heart_prediction/scaler.pkl')