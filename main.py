import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import time

# Load the custom model
model = YOLO('best.pt')

# Streamlit app title
st.title("Deteksi Tobrut Dengan YOLO Model")

# File uploader for a single image
uploaded_file = st.file_uploader("Unggah Sebuah Gambar", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # Create a progress bar
    progress_bar = st.progress(0)
    
    # Simulate the loading process with increments
    for percent_complete in range(0, 100, 10):
        time.sleep(0.1)  # Simulating processing time
        progress_bar.progress(percent_complete + 10)
    
    # Read image as a PIL object and convert it to an OpenCV image (NumPy array)
    img_pil = Image.open(uploaded_file)
    img_rgb = np.array(img_pil)

    # Predict using the YOLO model
    results = model(img_rgb)
    progress_bar.progress(100)  # Ensure the progress bar completes

    # Get the predicted names and probabilities
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()

    # Get the predicted label
    predicted_label = names_dict[np.argmax(probs)]

    # If the predicted label is "tobrut", apply Gaussian blur to the image
    if predicted_label == "tobrut":
        kernel_size = (25, 25)  # Adjust the kernel size for blurring
        sigmaX = 10  # Adjust sigmaX for the smoothness of the blur
        img_rgb = cv2.GaussianBlur(img_rgb, kernel_size, sigmaX)

    # Display the image with the predicted label and probability
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(img_rgb)
    ax.axis('off')  # Hide the axis
    ax.set_title(f"{predicted_label}\n({max(probs) * 100:.2f}%)")

    # Show the plot using Streamlit
    st.pyplot(fig)

    # Print the names and probabilities for the image
    st.write("### Prediksi Gambar dari Gambar yang telah diunggah")
    st.write(f"File: {uploaded_file.name}")

    # Change the font color and size based on the predicted label
    if predicted_label == "tobrut":
        st.markdown(
            f"<p style='color:red; font-size:24px;'>Terprediksi: {predicted_label}</p>",
            unsafe_allow_html=True
        )
        # Display a warning if blur was applied
        st.warning(
            f"Menerapkan blur pada gambar pada file {uploaded_file.name} dikarenakan gambar tersebut terdapat gambar tobrut."
        )
    else:
        st.markdown(
            f"<p style='color:green; font-size:24px;'>Terprediksi: {predicted_label}</p>",
            unsafe_allow_html=True
        )

else:
    st.write("Mohon upload sebuah gambar untuk melakukan prediksi.")