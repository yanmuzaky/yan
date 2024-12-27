import streamlit as st
import pandas as pd
import base64
import pickle

# Load the model
model_filename = "fruit_svm.sav"
scaler_filename = "fruit_svmScaler.sav"

model = pickle.load(open(model_filename, 'rb'))
scaler = pickle.load(open(scaler_filename, 'rb'))


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

    # Predict button
    if st.button("Prediksi Kualitas"):
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
            input_data_scaled = scaler.transform(input_data)
            predicted_class = model.predict(input_data_scaled)

            # Display result
            st.subheader("Hasil Prediksi")
            st.snow()
            st.write(f"**Klasifikasi diprediksi: {predicted_class}**")

        except Exception as e:
            st.error("Error in prediction: " + str(e))

st.markdown("---")
