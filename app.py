import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Page configuration
st.set_page_config(page_title="Shape & Contour Analyzer", layout="wide")

# App title
st.title("🔷 Shape & Contour Analyzer")

st.write(
    "Upload an image containing geometric shapes. "
    "The app will detect shapes, count objects, and display area and perimeter."
)

# File uploader
uploaded_file = st.file_uploader(
    "Upload an image (jpg / png)", type=["jpg", "png", "jpeg"]
)

# Function to detect shape based on number of vertices
def detect_shape(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    vertices = len(approx)

    if vertices == 3:
        return "Triangle"
    elif vertices == 4:
        return "Quadrilateral"
    elif vertices == 5:
        return "Pentagon"
    else:
        return "Circle"

# If image is uploaded
if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    img = np.array(image)

    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    results = []
    object_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        # Ignore small noisy contours
        if area > 500:
            object_count += 1
            perimeter = cv2.arcLength(contour, True)
            shape = detect_shape(contour)

            # Draw contour
            cv2.drawContours(img, [contour], -1, (255, 0, 0), 2)

            # Find center to place text
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(
                    img,
                    shape,
                    (cx - 40, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            results.append(
                [object_count, shape, round(area, 2), round(perimeter, 2)]
            )

    # Convert results to DataFrame
    df = pd.DataFrame(
        results,
        columns=["Object ID", "Shape", "Area (pixels)", "Perimeter (pixels)"],
    )

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Detected Shapes")
        st.image(img, use_container_width=True)

    with col2:
        st.subheader("Measurements")
        st.dataframe(df)

    st.success(f"Total Objects Detected: {object_count}")
