# Fashion MNIST Clustering and Deployment Project

## Project Overview

*applying dimensionality reduction and clustering to the Fashion MNIST dataset and deploying the results.*

## Members

*   **Raghad Aldosari** 
  
## Requirements

To run this project locally, you will need:

*   **Python:** Version 3.7 or higher is recommended.
*   **Pip:** Python package installer.
*   **Required Python Libraries:**
    *   `streamlit`
    *   `scikit-learn`
    *   `pandas`
    *   `numpy`
    *   `Pillow`
    *   `plotly`
    *   `joblib`

    You can install these libraries using pip:
    ```bash
    pip install streamlit scikit-learn pandas numpy Pillow plotly joblib
    ```

*   **Project Files:** The following files must be in the same directory:
    *   `fashion_cluster_app.py` 
    *   `pca_model.pkl` 
    *   `kmeans_model.pkl` 
    *   `train_images_sample_flat.npy` 
    *   `train_labels_sample.npy`
    *   `random_indices.npy` 
    *   `train_images.npy` 
    *   `train_labels.npy` 

## Instructions to Run

Follow these steps to get the Streamlit application running on your local machine:

1.  **Clone or Download the Project Files:** 
2.  **Navigate to the Project Directory:** 

3.  **Install Dependencies:**

4.  **Run the Streamlit Application:**
5.  **Access the Application:** 
6.  **Using the App:**
    *   You can upload your own 28x28 grayscale image using the file uploader.
    *   Alternatively, use the sidebar to select a fashion category and click the button to test with a random sample image from that category.
    *   The app will display the predicted K-Means cluster, the most similar items from the training data (in the PCA space), and the image's location on the PCA projection plot.


