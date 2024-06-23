import streamlit as st
import pandas as pd
import joblib

# Load the model and scaler
model = joblib.load(r'/mount/src/heart_prediction/best_model.pkl')
scaler = joblib.load(r'/mount/src/heart_prediction/scaler.pkl')

# Define the input fields
def user_input_features():
    age = st.number_input("Age", min_value=0, max_value=120, value=35)
    
    sex = st.selectbox("Sex", ['Female', 'Male'])
    if sex == 'Female':
        sex = 0
    else:
        sex = 1
    
    cp = st.selectbox("Chest Pain Type", ['Typical Angina (Tightness or Squeezing feeling in the chest)', 'Atypical Angina (Weakness , Nausea or sweating sensation)', 'Non-anginal Pain (Chest Pain associated with no heart disease)', 'Asymptomatic (No sign of any symptoms)'])
    if cp == 'Typical Angina (Tightness or Squeezing feeling in the chest)':
        cp = 0
    elif cp == 'Atypical Angina (Weakness , Nausea or sweating sensation)':
        cp = 1
    elif cp == 'Non-anginal Pain (Chest Pain associated with no heart disease)':
        cp = 2
    else:
        cp = 3

    trestbps = st.number_input("Resting Blood Pressure ( /mmHg)", min_value=0, max_value=400, value=120)
    chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=350, value=180)
    
    fbs = st.selectbox(" Is your Fasting Blood Sugar > 120 mg/dl ", ['False', 'True'])
    if fbs == 'False':
        fbs = 0
    else:
        fbs = 1
    
    restecg = st.selectbox("Resting ECG", ['Normal', 'Having ST-T wave abnormality', 'Showing probable or definite left ventricular hypertrophy'])
    if restecg == 'Normal':
        restecg = 0
    elif restecg == 'Having ST-T wave abnormality':
        restecg = 1
    else:
        restecg = 2

    thalach = st.number_input("Max Heart Rate Achieved", min_value=0, max_value=500, value=200)
    
    exang = st.selectbox("Exercise Induced Angina", ['No', 'Yes'])
    if exang == 'No':
        exang = 0
    else:
        exang = 1
    
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
    
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", ['Upsloping', 'Flat', 'Downsloping'])
    if slope == 'Upsloping':
        slope = 0
    elif slope == 'Flat':
        slope = 1
    else:
        slope = 2
    
    ca = st.number_input("Number of Major Vessels Colored by Flourosopy", min_value=0, max_value=3, value=0)
    
    thal = st.selectbox("Thalassemia", ['Normal', 'Fixed Defect', 'Reversible Defect'])
    if thal == 'Normal':
        thal = 0
    elif thal == 'Fixed Defect':
        thal = 1
    else:
        thal = 2
    
    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features


def main():
    st.title("Heart Disease Prediction")
    st.write("Enter the details of the patient to predict the possibility of heart disease.")

    # Get user input
    input_df = user_input_features()

    # Preprocess the input data
    input_df_scaled = scaler.transform(input_df)  # Use the loaded scaler to transform the input data

    # Make prediction
    prediction = model.predict(input_df_scaled)
    prediction_proba = model.predict_proba(input_df_scaled)

    # Convert prediction probabilities to percentages
    prob_no_disease = prediction_proba[0][0] * 100
    prob_disease = prediction_proba[0][1] * 100

    # Display prediction
    if prediction[0] == 1:
        st.error(f'The patient is likely to have heart disease with a probability of {prob_disease:.2f}%.')
    else:
        st.success(f'The patient is unlikely to have heart disease with a probability of {prob_no_disease:.2f}%.')

# Run the app
if __name__ == '__main__':
    main()
