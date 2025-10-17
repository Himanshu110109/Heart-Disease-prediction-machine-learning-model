import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Hearty",
    page_icon="❤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit's default menu and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

model = joblib.load("knn_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("""#Heart Stroke prediction model❤
by Himanshu""")
st.markdown("Provide the following details")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("SEX",["M","F"])
chest_pain = st.selectbox("Chest pain type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting blood pressure (mm HG)", 80, 200, 120)
cholesterol = st.number_input("cholesterol (mg/dl)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting blood sugar > 120mg/dl", [0,1])
resting_ecg = st.selectbox("Resting ecg", ["Normal","ST", "LVH"])
max_hr = st.slider("Max Heart Rate",60, 220, 150)
exercise_angina = st.selectbox("Exercise includer Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST depression)",0, 6 , 1)
st_slope = st.selectbox("ST slope", ["UP", "Flat", "Down"])

if st.button("Predict"):
    raw_input = {
        "Age": age,
        "RestingBP":resting_bp,
        "Cholesterol":cholesterol,
        "FastingBS":fasting_bs,
        "MaxHR":max_hr,
        "Oldpeak":oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_slope_" + st_slope: 1
    }
    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠ High risk of Heart Disease")
    else:

        st.success("✅ Low Risk of Heart Disease")


