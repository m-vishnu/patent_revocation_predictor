import pandas as pd

def load_translated_dataset(filepath, file_is_csv=False):
    if file_is_csv:
        translated_df = pd.read_csv(filepath)
    else:
        translated_df = pd.read_pickle(filepath)

    return translated_df