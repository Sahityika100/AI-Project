import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt

st.title("Medical AI Disease Predictor")
st.write("Predict *Diabetes* from patient data.")



with open("diabetes_model.pkl", "rb") as f:
    diabetes_model = pickle.load(f)

    st.subheader("Enter Patient Details for Diabetes Prediction")
with st.form("Diabetes Form"):
    pregnancies = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose Level", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)
    insulin = st.number_input("Insulin", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=1)

    if st.form_submit_button("Predict Diabetes"):
        user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                               insulin, bmi, dpf, age]])
        prediction = diabetes_model.predict(user_data)
        probability = diabetes_model.predict_proba(user_data)[0][1]*100
        
        if prediction[0]==1 :
            confidence = probability
            uncertainty = 100 - probability
            st.error("Result : Positive for Diabetes")
            st.write(f"Prediction Confidence : {probability:.2f}%")
            st.warning("Recommendation : Please consult a doctor for professional advice.")
            st.info("Tips : Maintain healthy diet, exercise regularly , monitor sugar levels.")
        else:
            confidence = 100 - probability
            uncertainty = probability
            st.success("Result : Negative for diabetes.")   
            st.write(f"Prediction Confidence : {100-probability:.2f}%")
            st.info("Keep up the healthy lifestyle.")

        fig , ax = plt.subplots()
        ax.pie([confidence,uncertainty],labels=['Confidence','Uncertainty'],colors=['green','red'],autopct='%1.1f%%',startangle=90,wedgeprops={'edgecolor':'black'},radius=0.5)
        ax.set_title("Prediction Confidence",fontsize=10)     
        st.pyplot(fig)       