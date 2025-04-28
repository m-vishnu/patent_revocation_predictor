
from src.data_utils.data_preprocessor import load_tfidf_preprocessed_data
import warnings

# Ask pandas to ignore ChainedAssignmentWarning
warnings.filterwarnings('ignore')

# import os # For debugging only
# os.chdir("D:\\Documents\\Personal\\projects\\sandoz\\sandoz_data_task")

WRITE_INTERMEDIATE_OUTPUTS = True

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    preprocessed_dataset = load_tfidf_preprocessed_data('data/preprocessed/all_en_df.pkl')

    if WRITE_INTERMEDIATE_OUTPUTS:
        preprocessed_dataset.to_pickle("data/preprocessed/translated_tfidf.pkl")





    print("Done")






