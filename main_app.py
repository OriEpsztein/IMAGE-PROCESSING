import streamlit as st
import cv2
import numpy as np
from PIL import Image

def main():
    st.set_page_config(page_title="Image Processing App", layout="wide")
    
    st.title("🖼️ Image Processing & Segmentation App")
    st.markdown("Upload an image to apply filters and segmentation algorithms using OpenCV.")

    # --- Sidebar: Upload & Settings ---
    st.sidebar.header("Settings")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Convert BGR to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display Original Image
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image_rgb, use_container_width=True)

        # Select Mode
        mode = st.sidebar.selectbox("Select Mode", ["Basic Filters", "Segmentation"])

        processed_image = None

        # --- MODE: BASIC FILTERS ---
        if mode == "Basic Filters":
            filter_type = st.sidebar.radio("Choose Filter", ["Grayscale", "Gaussian Blur", "Canny Edge Detection"])
            
            if filter_type == "Grayscale":
                processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Convert back to RGB format for Streamlit (even though it looks gray)
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)

            elif filter_type == "Gaussian Blur":
                # Slider for kernel size (must be odd)
                k_size = st.sidebar.slider("Kernel Size", 3, 21, 5, step=2)
                processed_image = cv2.GaussianBlur(image_rgb, (k_size, k_size), 0)

            elif filter_type == "Canny Edge Detection":
                t_lower = st.sidebar.slider("Lower Threshold", 0, 100, 50)
                t_upper = st.sidebar.slider("Upper Threshold", 100, 255, 150)
                edges = cv2.Canny(image, t_lower, t_upper)
                # Convert edges to RGB for display consistency
                processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        # --- MODE: SEGMENTATION ---
        elif mode == "Segmentation":
            seg_type = st.sidebar.radio("Choose Method", ["Thresholding (Otsu)", "K-Means Clustering"])

            if seg_type == "Thresholding (Otsu)":
                st.sidebar.info("Otsu's method automatically calculates the optimal threshold value.")
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Apply Otsu's thresholding
                ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                processed_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
                st.sidebar.write(f"Otsu Threshold Value: {ret}")

            elif seg_type == "K-Means Clustering":
                k = st.sidebar.slider("Number of Clusters (K)", 2, 10, 3)
                
                # Reshape image to a 2D array of pixels
                pixel_values = image_rgb.reshape((-1, 3))
                pixel_values = np.float32(pixel_values)

                # Define criteria and apply kmeans()
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

                # Convert back to 8 bit values
                centers = np.uint8(centers)
                segmented_data = centers[labels.flatten()]
                processed_image = segmented_data.reshape(image_rgb.shape)

        # Display Processed Image
        with col2:
            if processed_image is not None:
                st.subheader("Processed Image")
                st.image(processed_image, use_container_width=True)
            else:
                st.write("Adjust settings to see the result.")

    else:
        st.info("Please upload an image to get started.")

if __name__ == "__main__":
    main()
