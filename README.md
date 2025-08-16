# Fashion MNIST Clustering and Deployment Project

## Project Overview

*Applying dimensionality reduction and clustering to the Fashion MNIST dataset and deploying the results in a web application.*

## Team Members

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
    *   `fashion_cluster_app.py` *(Your Streamlit application script)*
    *   `pca_model.pkl` *(The saved PCA model)*
    *   `kmeans_model.pkl` *(The saved K-Means model)*
    *   `train_images_sample_flat.npy` *(Sampled training images - flattened)*
    *   `train_labels_sample.npy` *(Labels for the sampled training images)*
    *   `random_indices.npy` *(Indices used for sampling)*
    *   `train_images.npy` *(Original full training images - needed for displaying similar items and sample selection)*
    *   `train_labels.npy` *(Original full training labels - needed for similar items and sample selection)*

## Instructions to Run

Follow these steps to get the Streamlit application running on your local machine:

1.  **Clone or Download the Project Files:** Obtain all the project files listed in the "Requirements" section and place them in a single directory on your computer. Ensure you have downloaded all the saved `.pkl` and `.npy` files from your Colab notebook and have your `fashion_cluster_app.py` script.

2.  **Navigate to the Project Directory:** Open your terminal or command prompt and change your current directory to the location where you saved the project files.
    ```bash
    cd /path/to/your/project/directory
    ```
    *(Replace `/path/to/your/project/directory` with the actual path)*

3.  **Install Dependencies:** If you haven't already, install the required Python libraries using pip. It's recommended to use a virtual environment.
    ```bash
    pip install streamlit scikit-learn pandas numpy Pillow plotly joblib
    ```
    *(You can optionally create a `requirements.txt` file by running `pip freeze > requirements.txt` after installing the packages in your environment, and then use `pip install -r requirements.txt`)*

4.  **Run the Streamlit Application:** Execute the following command in your terminal:
    ```bash
    streamlit run fashion_cluster_app.py
    ```

5.  **Access the Application:** Your web browser should automatically open a new tab with the Streamlit application running (usually at `http://localhost:8501`). If not, open your browser and go to that address.

6.  **Using the App:**
    *   Use the sidebar on the left to select a fashion category from the dropdown list.
    *   A random sample image from the selected category will be displayed in the sidebar.
    *   The app will then automatically process this sample image, display its predicted K-Means cluster (including the dominant category name), show the most similar items from the sampled training data (in the PCA space), and highlight the image's location on the PCA projection plot.
