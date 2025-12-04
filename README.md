# IMAGE-PROCESSING
# 🖼️ Image Processing & Segmentation App

A dynamic web application built with **Streamlit** and **OpenCV** that allows users to upload images and apply various computer vision filters and segmentation algorithms in real-time.

## 🚀 Features

The application supports two main modes:

### 1. Basic Filters
* **Grayscale Conversion:** Converts the image to black and white.
* **Gaussian Blur:** Reduces image noise and detail using a customizable kernel size.
* **Canny Edge Detection:** Identifies structural edges in the image with adjustable hysteresis thresholds.

### 2. Segmentation
* **Otsu's Thresholding:** Automatically determines the optimal threshold value to separate the foreground from the background (Binarization).
* **K-Means Clustering:** Segments the image into *K* distinct colors/regions based on pixel color similarity. Includes a slider to adjust the number of clusters.

## 🛠️ Installation

1.  **Clone or Download** this repository.
2.  Ensure you have Python installed (version 3.8 or higher is recommended).
3.  Install the required dependencies using the provided requirements file:

```bash
pip install -r requirements.txt
