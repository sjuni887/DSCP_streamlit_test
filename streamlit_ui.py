import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title='ICU Prediction', page_icon='chart_with_upwards_trend')

# Load the saved logistic regression model
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define encoding mappings
Anemia_category_mapping = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}
GradeofKidneydisease_mapping = {"g1": 1, "G2": 2, "G3a": 3.1, "G3b": 3.2, "G4": 4, "G5": 5}
SurgRiskCategory_mapping = {"Low": 1, "Moderate": 2, "High": 3}
ASAcategorybinned_mapping = {"I": 1, "II": 2, "III": 3, "IV-VI": 4}
RaceCategory_mapping = {"Chinese": 1, "Others": 2, "Indian": 3, "Malay": 4}
GENDER_mapping = {'MALE': 1, 'FEMALE': 0}
AnaestypeCategory_mapping = {'GA': 0, 'RA': 1}
PriorityCategory_mapping = {'Elective': 0, 'Emergency': 1}
RDW15_7_mapping = {'<= 15.7': 0, '>15.7': 1}

# Define function to preprocess input features
def preprocess_features(features):
    features['Anemia category'] = Anemia_category_mapping.get(features['Anemia category'].lower(), 0)
    features['GradeofKidneydisease'] = GradeofKidneydisease_mapping.get(features['GradeofKidneydisease'].lower(), 1)
    features['SurgRiskCategory'] = SurgRiskCategory_mapping.get(features['SurgRiskCategory'].lower(), 1)
    features['ASAcategorybinned'] = ASAcategorybinned_mapping.get(features['ASAcategorybinned'].lower(), 1)
    features['GENDER'] = GENDER_mapping.get(features['GENDER'].upper(), 0)
    features['AnaestypeCategory'] = AnaestypeCategory_mapping.get(features['AnaestypeCategory'].upper(), 0)
    features['PriorityCategory'] = PriorityCategory_mapping.get(features['PriorityCategory'].upper(), 0)
    features['RDW15.7'] = RDW15_7_mapping.get(features['RDW15.7'].lower(), 0)
    race_category = features['RaceCategory'].lower()
    features['RaceCategory'] = RaceCategory_mapping.get(race_category, RaceCategory_mapping['Others'])
    return features

# Define function to make predictions
def predict_icu(input_features):
    input_features_processed = preprocess_features(input_features)
    input_features_processed = pd.DataFrame(input_features_processed, index=[0])
    prediction = model.predict(input_features_processed)
    probability = model.predict_proba(input_features_processed)[0][1]
    return prediction, probability

# Streamlit app layout
st.title('ICU Prediction')
st.write('Enter the patient features:')

# Feature names and default values
feature_defaults = {
    'AGE': 50,  # Example default value, replace with appropriate default values
    'GENDER': 'FEMALE',  # Example default value, replace with appropriate default values
    'RCRI score': 0.5,  # Example default value, replace with appropriate default values
    'Anemia category': 'none',  # Example default value, replace with appropriate default values
    'PreopEGFRMDRD': 60,  # Example default value, replace with appropriate default values
    'GradeofKidneydisease': 'g1',  # Example default value, replace with appropriate default values
    'AnaestypeCategory': 'GA',  # Example default value, replace with appropriate default values
    'PriorityCategory': 'Elective',  # Example default value, replace with appropriate default values
    'SurgRiskCategory': 'Low',  # Example default value, replace with appropriate default values
    'RaceCategory': 'Others',  # Example default value, replace with appropriate default values
    'RDW15.7': '<= 15.7',  # Example default value, replace with appropriate default values
    'ASAcategorybinned': 'I'  # Example default value, replace with appropriate default values
}

# User input for features
input_features = {}
for feature_name, default_value in feature_defaults.items():
    input_features[feature_name] = st.text_input(f'{feature_name}', default_value)

# Predict button
if st.button('Predict'):
    prediction, probability = predict_icu(input_features)
    st.write(f'Prediction: {"Goes to ICU" if prediction else "Does not go to ICU"}')
    st.write(f'Probability: {probability:.2f}')
