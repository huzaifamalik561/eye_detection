import cv2
import streamlit as st
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt

# Load the pre-trained face and eye detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Define the reference distance in millimeters
reference_distance_mm = 600  # Change this to your actual reference distance

# Function to calculate the eye distance
def calculate_eye_distance(image1, image2):
    image1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
    image2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)

    # Your existing code for eye distance calculation
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    faces1 = face_cascade.detectMultiScale(gray_image1, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    faces2 = face_cascade.detectMultiScale(gray_image2, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    (x1, y1, w1, h1) = faces1[0]
    (x2, y2, w2, h2) = faces2[0]

    face1 = image1[y1:y1+h1, x1:x1+w1]
    face2 = image2[y2:y2+h2, x2:x2+w2]

    resized_face1 = cv2.resize(face1, (620, 620))
    resized_face2 = cv2.resize(face2, (620, 620))

    gray_resized_face1 = cv2.cvtColor(resized_face1, cv2.COLOR_BGR2GRAY)
    gray_resized_face2 = cv2.cvtColor(resized_face2, cv2.COLOR_BGR2GRAY)

    eyes1 = eye_cascade.detectMultiScale(gray_resized_face1, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    eyes2 = eye_cascade.detectMultiScale(gray_resized_face2, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

    sorted_eyes1 = sorted(eyes1, key=lambda eye: eye[2] * eye[3], reverse=True)
    sorted_eyes2 = sorted(eyes2, key=lambda eye: eye[2] * eye[3], reverse=True)

    top_eyes1 = sorted_eyes1[:2]
    top_eyes2 = sorted_eyes2[:2]

    if len(top_eyes1) == 2 and len(top_eyes2) == 2:
        eye_center_1_x1, eye_center_1_y1 = top_eyes1[0][0] + top_eyes1[0][2] // 2, top_eyes1[0][1] + top_eyes1[0][3] // 2
        eye_center_2_x1, eye_center_2_y1 = top_eyes1[1][0] + top_eyes1[1][2] // 2, top_eyes1[1][1] + top_eyes1[1][3] // 2

        eye_center_1_x2, eye_center_1_y2 = top_eyes2[0][0] + top_eyes2[0][2] // 2, top_eyes2[0][1] + top_eyes2[0][3] // 2
        eye_center_2_x2, eye_center_2_y2 = top_eyes2[1][0] + top_eyes2[1][2] // 2, top_eyes2[1][1] + top_eyes2[1][3] // 2

        distance_pixels1 = math.sqrt((eye_center_2_x1 - eye_center_1_x1)**2 + (eye_center_2_y1 - eye_center_1_y1)**2)
        distance_mm1 = (distance_pixels1 / reference_distance_mm)

        distance_pixels2 = math.sqrt((eye_center_2_x2 - eye_center_1_x2)**2 + (eye_center_2_y2 - eye_center_1_y2)**2)
        distance_mm2 = (distance_pixels2 / reference_distance_mm)

        return distance_mm1, distance_mm2
    else:
        return None, None
        

# Streamlit UI
st.title("Eye Distance Calculator")
st.write("Upload two high-resolution pictures to calculate the eye distance.")

uploaded_image1 = st.file_uploader("Upload Image 1", type=["jpg", "png", "jpeg"])
uploaded_image2 = st.file_uploader("Upload Image 2", type=["jpg", "png", "jpeg"])

if uploaded_image1 and uploaded_image2:
    image1 = Image.open(uploaded_image1)
    image2 = Image.open(uploaded_image2)

    # Display the uploaded images
    st.image([image1, image2], caption=["Image 1", "Image 2"], width=250)

    # Calculate the eye distance
    distance_mm1, distance_mm2 = calculate_eye_distance(image1, image2)
    if distance_mm1 is not None and distance_mm2 is not None:
        st.write(f"Distance between eyes in Image 1: {distance_mm1:.2f} mm")
        st.write(f"Distance between eyes in Image 2: {distance_mm2:.2f} mm")

        # Display a warning for high-resolution images
        st.warning("Make sure to upload high-resolution pictures for accurate results.")

        # Create a scatter plot
        fig, ax = plt.subplots()
        ax.scatter([distance_mm1, distance_mm2], [1, 2])
        ax.set_xlabel("Eye Distance (mm)")
        ax.set_ylabel("Image")
        ax.set_yticks([1, 2])
        ax.set_yticklabels(["Image 1", "Image 2"])
        plt.title("Eye Distance Scatter Plot")

        # Display the scatter plot using st.pyplot()
        st.pyplot(fig)
    else:
        st.error("Could not detect both eyes in both images.")