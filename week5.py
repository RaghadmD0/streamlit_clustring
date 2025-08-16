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
    0: 'Shirt (Mixed)',
    1: 'Sandal (Mixed)',
    2: 'Ankle boot (Pure)',
    3: 'Bag (Pure)',
    4: 'Sneaker (Mixed)',
    5: 'Trouser (Mixed)',
    6: 'Pullover (Mixed)',
    7: 'Coat (Mixed)',
    8: 'T-shirt/top (Mixed)',
    9: 'Bag (Pure)'
}
# You can refine these names based on the purity you observed.
# For example, if cluster 2 was 85.27% Ankle boot, you might call it 'Ankle boot (Dominant)'.


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
        original_train_images = np.load("train_images.npy")
        original_train_labels = np.load("train_labels.npy")

        # Transform the sampled data using the loaded PCA model
        train_pca_transformed = pca_model.transform(train_images_sample_flat)

        # Create a DataFrame for the scatter plot using the 2D PCA projection
        df_train_pca_2d = pd.DataFrame(train_pca_transformed[:, :2], columns=["PC1", "PC2"])
        df_train_pca_2d['KMeans_Label'] = kmeans_model.labels_ # Add K-Means cluster labels

        # Add original fashion categories for coloring and hover info for the SAMPLED data
        # Map sampled indices (random_indices) to original labels
        sampled_original_labels = original_train_labels[random_indices]
        df_train_pca_2d['Original_Category'] = pd.Series(sampled_original_labels).map(fashion_categories)
        df_train_pca_2d['Original_Label'] = sampled_original_labels # Keep numerical label


        return pca_model, kmeans_model, train_images_sample_flat, train_labels_sample, train_pca_transformed, df_train_pca_2d, random_indices, original_train_images, original_train_labels

    except FileNotFoundError:
        st.error("Model or data files not found. Please ensure 'pca_model.pkl', 'kmeans_model.pkl', 'train_images_sample_flat.npy', 'train_labels_sample.npy', 'random_indices.npy', 'train_images.npy', and 'train_labels.npy' are in the same directory as the script.")
        st.stop() # Stop the app if essential files are not found
    except Exception as e:
        st.error(f"An error occurred while loading resources: {e}")
        st.stop() # Stop the app on other loading errors


pca, kmeans, train_images_sample_flat, train_labels_sample, train_pca_transformed, df_train_pca_2d, random_indices_sample, original_train_images, original_train_labels = load_resources()


st.title("Fashion MNIST Image Clustering with PCA and K-Means")
st.write("Upload a 28x28 grayscale fashion item image to see its predicted cluster and location in the PCA space.")

# --- Upload user image ---
uploaded_file = st.file_uploader("Upload a 28x28 image", type=["png", "jpg", "jpeg"])

# --- Or select a sample image ---
st.sidebar.title("Select a Sample Image")
sample_options = {name: label for label, name in fashion_categories.items()}
selected_category_name = st.sidebar.selectbox("Choose a category:", list(sample_options.keys()))
selected_category_label = sample_options[selected_category_name]

# Find indices of images from the selected category in the *original* full dataset
if original_train_images is not None and original_train_labels is not None:
    original_indices_for_category = np.where(original_train_labels == selected_category_label)[0] # Use loaded original_train_labels
    if len(original_indices_for_category) > 0:
        random_sample_original_index = random.choice(original_indices_for_category)
        sample_image_original = original_train_images[random_sample_original_index] # Use loaded original_train_images

        # Display the selected sample image
        st.sidebar.image(sample_image_original, caption=f"Selected Sample: {selected_category_name}", width=100)

        # Button to use this sample image
        use_sample_button = st.sidebar.button(f"Use this {selected_category_name} sample")
    else:
        st.sidebar.write("No images found for this category in the original training data.")
        use_sample_button = False # Disable button
        sample_image_original = None

else:
    st.sidebar.warning("Original training images and labels not available to select samples.")
    use_sample_button = False
    sample_image_original = None


# --- Process the image (uploaded or sample) ---
img_array = None
display_image = None
image_source = None
original_image_label = None # To store original label of the sample image

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")  # grayscale
    img = img.resize((28, 28))
    img_array = np.array(img).reshape(1, -1).astype(np.float32) / 255.0  # flatten + normalize
    display_image = img
    image_source = "Uploaded Image"
    original_image_label = None # No original label for uploaded image
elif use_sample_button and sample_image_original is not None:
     img_array = sample_image_original.reshape(1, -1).astype(np.float32) / 255.0 # flatten + normalize
     display_image = Image.fromarray(sample_image_original)
     image_source = f"Sample Image ({selected_category_name})"
     original_image_label = selected_category_label # Store the original label of the sample


if img_array is not None:
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
    if image_source.startswith("Sample Image") and original_image_label is not None:
        original_cat_name = fashion_categories.get(original_image_label, f"Label {original_image_label}")
        st.write(f"Original Category: **{original_cat_name}**")


    # --- Find most similar items in the SAMPLED training data ---
    # Calculate distances to the sampled data points that the models were trained on
    distances = np.linalg.norm(train_pca_transformed - img_pca, axis=1)

    # Get indices of the nearest neighbors in the *sampled* data array
    # Get top 6 including potentially the image itself if it's in the sample
    nearest_sampled_indices = np.argsort(distances)[:] # Get all for filtering

    # Filter out the uploaded image itself if it's a sample from the training set
    # This is a heuristic; for robust filtering, you'd need image comparison.
    # For simplicity, we'll just take the top N after sorting.
    # If the uploaded image is identical to a training image, it will be the first result with distance 0.
    # We'll take the top N *distinct* indices that are not the uploaded image itself (if uploaded).
    # A simple approach is to just take the top N, and if distance is very close to 0, assume it's the same image and skip it.

    st.subheader(f"Most Similar Items from the Sampled Training Data (in PCA space) - Top {5}") # Display Top N

    # Display the most similar items as images
    if original_train_images is not None and original_train_labels is not None:
        displayed_count = 0
        cols = st.columns(5) # Adjust number of columns as needed, e.g., 5

        for sampled_idx in nearest_sampled_indices:
            # Map sampled index back to original index
            original_idx = random_indices_sample[sampled_idx]
            dist = distances[sampled_idx]

            # Skip if the distance is very close to 0 (likely the uploaded image itself) and it's an uploaded image
            # and we haven't displayed all similar images yet.
            if uploaded_file is not None and dist < 1e-6 and displayed_count < 5: # Use a small tolerance
                 continue

            # Get the image and its original category
            similar_img = original_train_images[original_idx]
            similar_label = original_train_labels[original_idx]
            original_cat = fashion_categories.get(similar_label, f"Label {similar_label}")

            # Display the image and info in a column
            if displayed_count < 5: # Display only the top 5
                cols[displayed_count].image(similar_img, caption=f"Dist: {dist:.2f}\nOrig: {original_cat}", width=70)
                displayed_count += 1
            else:
                break # Stop once we have displayed 5

        if displayed_count == 0:
             st.write("Could not find similar items in the sampled data.")

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
    st.write("Please upload an image or select a sample from the sidebar to begin.")

st.markdown("---")
st.write("Clustering performed using PCA (50 components) and K-Means (K=10) on a sample of 10,000 Fashion MNIST training images.")