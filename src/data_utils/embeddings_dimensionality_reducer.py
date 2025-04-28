from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
import pandas as pd

def return_info_for(embeddings_array):
    """
    Plot the explained variance ratio for each principal component.
    """
    # Standardize the embeddings
    scaler = StandardScaler()
    embeddings_array = scaler.fit_transform(embeddings_array)

    pca = PCA()
    pca.fit(embeddings_array)

    return pca.explained_variance_ratio_

def reduce_dimensionality(embeddings_array, n_components=0.9):
    """
    Reduce the dimensionality of the embeddings using PCA.
    """
    # Standardize the embeddings
    scaler = StandardScaler()
    embeddings_array = scaler.fit_transform(embeddings_array)

    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings_array)

    return pca, reduced_embeddings

# function to transform new data using an existing PCA model
def transform_new_data(pca, new_data):
    """
    Transform new data using an existing PCA model.
    """
    # Standardize the new data
    scaler = StandardScaler()
    new_data = scaler.fit_transform(new_data)

    transformed_data = pca.transform(new_data)

    return transformed_data
