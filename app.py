import streamlit as st
import pandas as pd
import pickle

# Load the cleaned dataset
df = pd.read_csv('cleaned_dataset1.csv')

features = ['Goal', 'Interest']

# Load the best model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

# Streamlit app
st.title('Course Recommendation System')

goals = df['Goal'].unique()
interests = df['Interest'].unique()
selected_goal = st.selectbox('Select your goal:', goals)
selected_interest = st.selectbox('Select your interest:', interests)

if st.button('Get Recommendations'):
    X_new = pd.DataFrame([[selected_goal, selected_interest]], columns=features)
    recommended_course_encoded = model.predict(X_new)
    recommended_course = le.inverse_transform(recommended_course_encoded)
    st.write(f"Recommended Course: {recommended_course[0]}")
