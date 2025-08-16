import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from PIL import Image
import plotly.express as px
import random # Import random for sample image selection
import os # Import os to check for files

# --- Define fashion categories ---
# This dictionary is needed to map numerical labels to names
fashion_categories = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# --- Define dominant category mapping for K-Means clusters ---
# This dictionary is derived from the analysis in the notebook (cell 7173a4d6 output)
# Replace these values with the actual dominant categories observed for your K=10 clusters
# based on the output of the K-Means cluster composition analysis.
# Example (replace with your actual findings):
kmeans_dominant_categories = {
    0: 'Trouser (Dominant)',
    1: 'Sandal (Dominant)',
    2: 'Ankle boot (Dominant)',
    3: 'Bag (Pure)',
    4: 'Sneaker (Dominant)',
    5: 'Mixed Garments', # T-shirt/top, Dress, Shirt, Pullover, Trouser, Coat, Bag, Sandal, Ankle boot
    6: 'Mixed Garments', # Bag, Shirt, Pullover, Coat, T-shirt/top, Dress, Trouser, Ankle boot, Sandal
    7: 'Ankle boot (Dominant)',
    8: 'Mixed Garments', # Coat, Pullover, Shirt, Bag, T-shirt/top, Trouser, Dress
    9: 'Mixed Garments' # T-shirt/top, Dress, Shirt, Coat, Trouser, Pullover, Bag
}
# I've updated these based on the cluster composition analysis outputs.
# You can refine these names further based on the percentages you observed.


# --- Load models and data ---
@st.cache_resource # Cache the models and data loading
def load_resources():
    try:
        pca_model = joblib.load("pca_model.pkl")
        kmeans_model = joblib.load("kmeans_model.pkl")
        # Load the sampled data and original labels used for training the models
        train_images_sample_flat = np.load("train_images_sample_flat.npy")
        train_labels_sample = np.load("train_labels_sample.npy")
        random_indices = np.load("random_indices.npy") # Indices in the original dataset

        # Load the original full training images and labels (needed for displaying similar items and sampling)
        original_train_images_path = "train_images.npy" # Or "train_images_full.npy" if you saved the full set
        original_train_labels_path = "train_labels.npy"

        original_train_images = None
        original_train_labels = None

        if os.path.exists(original_train_images_path) and os.path.exists(original_train_labels_path):
             original_train_images = np.load(original_train_images_path)
             original_train_labels = np.load(original_train_labels_path)
        else:
             st.warning(f"Original training images and labels not found at {original_train_images_path} and {original_train_labels_path}. Similar images and sample selection might not work correctly.")


        # Transform the sampled data using the loaded PCA model
        train_pca_transformed = pca_model.transform(train_images_sample_flat)

        # Create a DataFrame for the scatter plot using the 2D PCA projection
        df_train_pca_2d = pd.DataFrame(train_pca_transformed[:, :2], columns=["PC1", "PC2"])
        df_train_pca_2d['KMeans_Label'] = kmeans_model.labels_ # Add K-Means cluster labels

        # Add original fashion categories for coloring and hover info for the SAMPLED data
        if original_train_labels is not None and len(original_train_labels) >= len(random_indices):
             # Map sampled indices (random_indices) to original labels
             sampled_original_labels = original_train_labels[random_indices]
             df_train_pca_2d['Original_Category'] = pd.Series(sampled_original_labels).map(fashion_categories)
             df_train_pca_2d['Original_Label'] = sampled_original_labels # Keep numerical label
        else:
             st.warning("Original training labels are not fully available or do not match sampled data size. Cannot color plot by category.")
             df_train_pca_2d['Original_Category'] = 'Unknown'
             df_train_pca_2d['Original_Label'] = 'Unknown'


        return pca_model, kmeans_model, train_images_sample_flat, train_labels_sample, train_pca_transformed, df_train_pca_2d, random_indices, original_train_images, original_train_labels

    except FileNotFoundError:
        st.error("Model or data files not found. Please ensure 'pca_model.pkl', 'kmeans_model.pkl', 'train_images_sample_flat.npy', 'train_labels_sample.npy', 'random_indices.npy' and optionally 'train_images.npy', 'train_labels.npy' are in the same directory.")
        st.stop() # Stop the app if essential files are not found
    except Exception as e:
        st.error(f"An error occurred while loading resources: {e}")
        st.stop() # Stop the app on other loading errors


# --- Load resources and initialize Streamlit app ---
pca, kmeans, train_images_sample_flat, train_labels_sample, train_pca_transformed, df_train_pca_2d, random_indices_sample, original_train_images, original_train_labels = load_resources()


st.title("Fashion MNIST Image Clustering with PCA and K-Means")
st.write("Select a sample 28x28 grayscale fashion item image from the sidebar to see its predicted cluster and location in the PCA space.")

# --- Select a Sample Image from sidebar ---
st.sidebar.header("Explore Sample Images")
st.sidebar.markdown("Choose a category below to load a random sample image and see its clustering results.")

# Only show sample selection if original images and labels were loaded successfully
img_array = None
display_image = None
image_source = None
original_image_label = None # To store original label of the sample image
sample_image_original = None # Define here to make it available outside the if block


if original_train_images is not None and original_train_labels is not None:
    sample_options = {name: label for label, name in fashion_categories.items()}
    selected_category_name = st.sidebar.selectbox("Select Fashion Category:", list(sample_options.keys()), key='category_select')
    selected_category_label = sample_options[selected_category_name]

    # Add a button to randomly pick an image from the selected category
    if st.sidebar.button("Pick Random Image", key='random_pick_button'):
        # Find indices of images from the selected category in the *original* full dataset
        original_indices_for_category = np.where(original_train_labels == selected_category_label)[0]
        if len(original_indices_for_category) > 0:
            random_sample_original_index = random.choice(original_indices_for_category)
            sample_image_original = original_train_images[random_sample_original_index]

            # Store the selected image index and label in session state to persist it
            st.session_state['selected_sample_original_index'] = random_sample_original_index
            st.session_state['selected_category_label'] = selected_category_label
            # Set a flag to indicate results should be displayed
            st.session_state['display_results'] = True


        else:
            st.sidebar.write("No images found for this category in the original training data.")
            st.session_state['selected_sample_original_index'] = None
            st.session_state['selected_category_label'] = None
            st.session_state['display_results'] = False # Don't display results if no image


    # Check if an image index is stored in session state and load it for display
    if 'selected_sample_original_index' in st.session_state and st.session_state['selected_sample_original_index'] is not None:
        random_sample_original_index = st.session_state['selected_sample_original_index']
        selected_category_label = st.session_state['selected_category_label']
        selected_category_name = fashion_categories.get(selected_category_label, f"Label {selected_category_label}")

        sample_image_original = original_train_images[random_sample_original_index]

        # Display the selected sample image with a caption
        st.sidebar.image(sample_image_original, caption=f"Showing sample from: {selected_category_name}", width=150)

        # Prepare the image array for processing
        img_array = sample_image_original.reshape(1, -1).astype(np.float32) / 255.0 # flatten + normalize
        display_image = Image.fromarray(sample_image_original)
        image_source = f"Sample Image ({selected_category_name})"
        original_image_label = selected_category_label # Store the original label of the sample

        # Set a flag to display results if an image is loaded (only if not already set by button)
        if 'display_results' not in st.session_state:
            st.session_state['display_results'] = True


    st.sidebar.markdown("---")


else:
    st.sidebar.warning("Original training images and labels not available to select samples.")
    st.session_state['display_results'] = False # Don't display results if original data is missing


# --- Process the image and display results (only if an image is loaded and display_results is True) ---
if img_array is not None and st.session_state.get('display_results', False):
    st.subheader(f"{image_source}")
    st.image(display_image, caption=image_source, width=150)

    # --- PCA transform ---
    img_pca = pca.transform(img_array)

    # --- Predict cluster ---
    cluster = kmeans.predict(img_pca)[0]

    # Get the dominant category name for the predicted cluster
    predicted_cluster_name = kmeans_dominant_categories.get(cluster, f"Cluster {cluster} (Unknown Composition)")

    st.write(f"Predicted K-Means Cluster: **{cluster}** ({predicted_cluster_name})")

    # If it's a sample image, show its original category
    if original_image_label is not None:
        original_cat_name = fashion_categories.get(original_image_label, f"Label {original_image_label}")
        st.write(f"Original Category: **{original_cat_name}**")


    # --- Find most similar items in the SAMPLED training data ---
    st.subheader(f"Most Similar Items from the Sampled Training Data (in PCA space) - Top {5}")

    if original_train_images is not None and original_train_labels is not None:
        # Calculate distances to the sampled data points that the models were trained on
        distances = np.linalg.norm(train_pca_transformed - img_pca, axis=1)

        # Get indices of the nearest neighbors in the *sampled* data array
        # Get top N including potentially the image itself if it's in the sample
        items_to_find = 6 # Find N+1 to potentially exclude the sample itself
        nearest_sampled_indices = np.argsort(distances)[:items_to_find]

        # Filter out the selected sample image itself from the "most similar" list
        # Find the index of the current sample image within the sampled training data
        current_sample_idx_in_sampled = None
        if 'selected_sample_original_index' in st.session_state and st.session_state['selected_sample_original_index'] is not None:
             try:
                # Find the index in train_images_sample_flat corresponding to the original index
                # Check if the original index is actually in the random_indices_sample array
                if st.session_state['selected_sample_original_index'] in random_indices_sample:
                    current_sample_idx_in_sampled = int(np.where(random_indices_sample == st.session_state['selected_sample_original_index'])[0][0])
                else:
                    # This case should ideally not happen if sample comes from loaded data
                    current_sample_idx_in_sampled = None
             except Exception:
                  current_sample_idx_in_sampled = None


        displayed_count = 0
        cols = st.columns(5) # Adjust number of columns as needed, e.g., 5
        items_to_display = 5 # Display only the top N (excluding the sample itself)

        for sampled_idx in nearest_sampled_indices:
            # Skip the current selected sample image itself
            if current_sample_idx_in_sampled is not None and sampled_idx == current_sample_idx_in_sampled:
                 continue

            # Ensure we don't go out of bounds if nearest_sampled_indices is smaller than items_to_find
            if sampled_idx >= len(random_indices_sample):
                continue

            # Map sampled index back to original index
            original_idx = random_indices_sample[sampled_idx]
            dist = distances[sampled_idx]

            # Get the image and its original category
            similar_img = original_train_images[original_idx]
            similar_label = original_train_labels[original_idx]
            original_cat = fashion_categories.get(similar_label, f"Label {similar_label}")

            # Display the image and info in a column
            if displayed_count < items_to_display:
                with cols[displayed_count]: # Use 'with' to put elements in the column
                    st.image(similar_img, caption=f"Dist: {dist:.2f}\nOrig: {original_cat}", width=70)
                displayed_count += 1
            else:
                break # Stop once we have displayed N

        if displayed_count == 0:
             st.write("Could not find similar items in the sampled data (excluding the selected sample itself).")

    else:
        st.write("Original training images and labels not available to display similar items.")


    # --- PCA projection plot ---
    st.subheader("PCA Projection with Image Location")

    # Create the scatter plot using the 2D PCA of the sampled training data
    # Color by the original category of the sampled data
    fig = px.scatter(
        df_train_pca_2d,
        x="PC1",
        y="PC2",
        color='Original_Category', # Color by original category of sampled data
        hover_data=['Original_Category', 'KMeans_Label'], # Show original category and cluster on hover
        title=f"PCA Projection of Sampled Training Images (Colored by Original Category)",
        opacity=0.6
    )

    # Highlight user image location
    fig.add_scatter(
        x=[img_pca[0, 0]],
        y=[img_pca[0, 1]],
        mode="markers",
        marker=dict(color="red", size=12, symbol='star', line=dict(width=2, color='DarkSlateGrey')), # Use a star marker and add outline
        name=image_source # Use the source (Uploaded/Sample) as the name
    )

    # Update layout for better appearance and dark mode (if supported by Streamlit theme)
    fig.update_layout(
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        legend_title='Original Category',
        hovermode='closest'
    )


    st.plotly_chart(fig, use_container_width=True)

else:
    # Message to show if original images/labels are not loaded, preventing sample selection
    if original_train_images is None or original_train_labels is None:
        st.error("Cannot run the application because the original training images and labels were not loaded.")
        st.warning("Please ensure 'train_images.npy' and 'train_labels.npy' are in the same directory as the script.")
    else:
         # Only show this message if no image is currently selected or display_results is False
         if not st.session_state.get('display_results', False):
             st.write("Select a category from the sidebar and click 'Pick Random Image' to display a sample image and its clustering results.")


st.markdown("---")
st.write("Clustering performed using PCA (50 components) and K-Means (K=10) on a sample of 10,000 Fashion MNIST training images.")