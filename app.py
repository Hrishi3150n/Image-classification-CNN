import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
from skimage.transform import resize
import base64


# Load the trained model
def load_my_model():
    return load_model("image_classification_cnn.h5")


model = load_my_model()

# Class labels
categories = ["Buildings", "Forest", "Glacier", "Mountain", "Sea", "Street"]
confidence_threshold = 0.6  # Adjust based on testing


# Function to set background image
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Prediction function
def predict_image(img_array):
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    if confidence < confidence_threshold:
        return "Unknown", confidence
    else:
        return categories[predicted_class], confidence


# Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📷 Upload Image", "🎥 Live Camera", "ℹ️ About"])

# 🏠 **Home Page**
if page == "🏠 Home":
    set_background("home.jpeg")
    st.title("🏞️ Image Classification App")
    st.write("""
        This app uses a **CNN model** to classify images into six categories:
        - 🏢 Buildings  
        - 🌳 Forest  
        - ❄️ Glacier  
        - ⛰️ Mountain  
        - 🌊 Sea  
        - 🏙️ Street  

        Upload an image or take a picture to get predictions!  
    """)
#    st.image("home.jpeg", use_container_width=True)

# 📷 **Upload Image Page**
elif page == "📷 Upload Image":
    set_background("upload.jpeg")
    st.title("📸 Upload Image for Prediction")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Process uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Convert to grayscale and resize
        img = img.convert("L").resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 150, 150, 1)

        # Make prediction
        predicted_label, confidence = predict_image(img_array)

        # Display result
        if predicted_label == "Unknown":
            st.error("🚨 The image does not belong to any known category! Please try another image.")
        else:
            st.success(f"Prediction: **{predicted_label}** ✅")
            st.write(f"Confidence: **{confidence:.2f}**")

# 🎥 **Live Camera Page**
elif page == "🎥 Live Camera":
    set_background("cam.jpeg")
    st.title("📷 Live Camera Prediction")

    # Open webcam
    cap = cv2.VideoCapture(0)

    if st.button("Capture Image"):
        ret, frame = cap.read()
        if ret:
            cap.release()
            # Convert to grayscale and process
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (150, 150))
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, 150, 150, 1)

            # Make prediction
            predicted_label, confidence = predict_image(img_array)

            # Display the captured image
            st.image(frame, caption="Captured Image", use_container_width=True, channels="BGR")

            # Show results
            if predicted_label == "Unknown":
                st.error("🚨 The image does not belong to any known category! Please try another image.")
            else:
                st.success(f"Prediction: **{predicted_label}** ✅")
                st.write(f"Confidence: **{confidence:.2f}**")

    cap.release()

# ℹ️ **About Page**
elif page == "ℹ️ About":
    set_background("about.jpg")
    st.title("ℹ️ About This App")
    st.write("""
        This application is built using **Streamlit** and a **Convolutional Neural Network (CNN)** model.

        - **Model:** CNN trained on six categories of images.  
        - **Framework:** TensorFlow & Keras  
        - **Frontend:** Streamlit with custom UI enhancements.  

        **How it works:**  
        1. Upload an image or take a picture.  
        2. The model processes the image and predicts its category.  
        3. If the image does not match any of the six categories, the app will notify you.  

        *Developed for educational and practical use cases in Image Classification.*  
    """)
