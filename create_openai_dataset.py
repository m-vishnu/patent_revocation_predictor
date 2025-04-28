import os.path
import pickle
import pandas as pd

from src.data_utils.data_preprocessor import merge_embedded_data
import src.data_utils.embeddings_dimensionality_reducer as embeddings_dimensionality_reducer
import warnings
warnings.filterwarnings('ignore')


def pickle_object(obj_to_pickle, filepath):
    if os.path.exists(filepath):
        print("File exists. Not saving.")

    else:
        # Save the object to a pickle file
        with open(filepath, 'wb') as f:
            pickle.dump(obj_to_pickle, f)
        print(f"Object saved to {filepath}")

def get_pca_embeddings(df, pca_percent=0.9):
    if type(df.iloc[0]['title_summary_embedding']) == str:
        df['title_summary_embedding'] = df['title_summary_embedding'].apply(lambda x: eval(x))

    pca, embeddings_reduced_dims = embeddings_dimensionality_reducer.reduce_dimensionality(
        [x for x in df['title_summary_embedding']], n_components=pca_percent)

    # Merge the PCA components with the original dataframe
    pca_df = pd.DataFrame(embeddings_reduced_dims,
                          columns=[f'PC{i + 1}' for i in range(embeddings_reduced_dims.shape[1])])

    pca_df = pd.concat([df, pca_df], axis=1)

    pca_df.drop(columns=['title_summary_embedding'], inplace=True)
    return pca, pca_df



def load_longformer_data():
    longformer_embedded_df = pd.read_csv('data/preprocessed/csvs/raw_data_with_openai_embeddings.csv').drop(columns=['title_summary_embedding'])
    longformer_embeddings = pd.DataFrame(pd.read_pickle('data/preprocessed/longformer_embeddings_array.pkl'),
                                         columns=[f"embedding_{i}" for i in range(768)])

    longformer_embedded_df = pd.concat([longformer_embedded_df, longformer_embeddings], axis=1)

    longformer_preprocessed_merged = merge_embedded_data(longformer_embedded_df)

    return longformer_preprocessed_merged


def load_openai_data():
    openai_embedded_df = pd.read_csv('data/preprocessed/csvs/raw_data_with_openai_embeddings.csv')
    openai_preprocessed_merged = merge_embedded_data(openai_embedded_df)

    return openai_preprocessed_merged


# Load data
openai_embedded_df = load_openai_data()
# longformer_embedded_df = load_longformer_data()


# Reduce dimensionality of OpenAI embeddings
openai_pca, openai_embedded_df = get_pca_embeddings(openai_embedded_df, pca_percent=0.9)

pickle_object(openai_pca, 'data/preprocessed/openai/openai_pca.pkl')
pickle_object(openai_embedded_df, 'data/preprocessed/openai/openai_pca_embeddings.pkl')

# Reduce dimensionality of Longformer embeddings
# longformer_pca, longformer_embedded_df = get_pca_embeddings(longformer_embedded_df, pca_percent=0.9)


