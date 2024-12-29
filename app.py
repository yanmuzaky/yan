import streamlit as st
import pandas as pd
import base64
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load the model

model_svm = pickle.load(open("SVM/fruit_svm.sav", 'rb'))
scaler_svm = pickle.load(open("SVM/fruit_svmScaler.sav", 'rb'))
model_rf = pickle.load(open("Random Forest/fruit_rf.sav", 'rb'))
scaler_rf = pickle.load(open("Random Forest/fruit_rfScaler.sav", 'rb'))


# Streamlit app
st.set_page_config(page_title="Klasifikasi Buah", layout="wide")
st.title("Klafisikasi Buah")

# Add background image
def get_base64_image(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

image_path = "images.jpg"
base64_image = get_base64_image(image_path)
st.markdown(
    f"""
    <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{base64_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Input form
st.header("Masukkan Data Buah")

col1, col2 = st.columns(2)

with col1:
    diameter = st.number_input("Diameter ")
    weight = st.number_input("Weight ")
    red = st.number_input("Red ")

with col2:
    green = st.number_input("Green ")
    blue = st.number_input("Blue ")

    st.write(" ")
    st.write(" ")
    st.write(" ")

    col1, col2 = st.columns(2)
    with col1:

        # Predict button SVM
        if st.button("Klasifikasi SVM"):
            # Prepare the input data for prediction
            input_data = pd.DataFrame({
                'diameter': [diameter],
                'weight': [weight],
                'red': [red],
                'green': [green],
                'blue': [blue],
            })

            # Make prediction
            try:
                input_data_scaled = scaler_svm.transform(input_data)
                predicted_class = model_svm.predict(input_data_scaled)

                # Display result
                st.subheader("Hasil Klasifikasi")
                st.snow()
                st.write(f"**Klasifikasi diprediksi: {predicted_class}**")

            except Exception as e:
                st.error("Error in prediction: " + str(e))

    with col2:
        # Predict button Random Forest
        if st.button("Klasifikasi Random Forest"):
            # Prepare the input data for prediction
            input_data = pd.DataFrame({
                'diameter': [diameter],
                'weight': [weight],
                'red': [red],
                'green': [green],
                'blue': [blue],
            })

            # Make prediction
            try:
                input_data_scaled = scaler_rf.transform(input_data)
                predicted_class = model_rf.predict(input_data_scaled)

                # Display result
                st.subheader("Hasil Klasifikasi")
                st.balloons()
                st.write(f"**Klasifikasi diprediksi: {predicted_class}**")

            except Exception as e:
                st.error("Error in prediction: " + str(e))

st.markdown("---")
