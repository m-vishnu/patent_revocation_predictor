import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import regexp_tokenize
import src.data_utils.data_loader as data_loader
from sklearn.preprocessing import OneHotEncoder


def get_token_map(df, series_name: str):
    all_tokens = []
    for i in range(len(df)):
        tokens = custom_tokenize(df.iloc[i][series_name])
        all_tokens.extend([x for x in tokens if x not in all_tokens])

    tokens_mapper = {all_tokens[i]: i for i in range(len(all_tokens))}
    return tokens_mapper


def get_cleaned_proprietor_info(df, csvpath: str = 'data/working/proprietor-mapping.csv') -> pd.DataFrame:
    mapping = pd.read_csv(csvpath, sep=',')[['Patent Proprietor', 'Map1']]
    mapping.set_index('Patent Proprietor', inplace=True)
    mapping_dict = mapping.to_dict()

    proprietors = df['Patent Proprietor']

    # If the mapping is not found, keep the original value, else use the value in mapping_dict['Map1']
    cleaned_proprietor_info = pd.Series(
        [x if mapping_dict['Map1'].get(x) is None else mapping_dict['Map1'].get(x) for x in proprietors],
        index=proprietors.index
    ).fillna(proprietors)

    df.loc[:, 'Patent Proprietor'] = cleaned_proprietor_info

    return df


def get_top_proprietor_info_deduped(df, top_n=20) -> pd.DataFrame:
    df = df.copy()

    # dedup proprietor info
    df = get_cleaned_proprietor_info(df)

    # Keep top_n proprietors, change others to 'Other'
    top_proprietors = df['Patent Proprietor'].value_counts().nlargest(top_n).index.tolist()
    df['Patent Proprietor'] = df['Patent Proprietor'].apply(
        lambda x: x if x in top_proprietors else 'Other')

    return df


def insert_top_provisions_flags(df, provisions_column: str, top_n: int = 35,
                                tokens_mapper: dict = None) -> pd.DataFrame:
    """
    Inserts top_n provisions flags into the DataFrame based on the given provisions column.

    Parameters:
    - df: pandas.DataFrame
    - provisions_column: name of the provisions column in the DataFrame
    - top_n: number of top provisions to keep
    - tokens_mapper: dictionary mapping tokens to indices

    Returns:
    - DataFrame with new provision features
    """
    df.copy()

    if tokens_mapper is None:
        tokens_mapper = get_token_map(df, provisions_column)

    # Initialize an empty DataFrame with the same index as df
    one_hot_df = pd.DataFrame(0, index=df.index, columns=tokens_mapper.keys())

    for i in range(len(df)):
        tokens = custom_tokenize(df.iloc[i][provisions_column])
        for token in tokens:
            one_hot_df.iloc[i][token] = 1

    column_totals = one_hot_df.sum(axis=0)
    top_columns = column_totals.nlargest(top_n).index.tolist()

    rest_of_columns = [col for col in one_hot_df.columns if col not in top_columns]

    pruned_one_hot_encoded_df = one_hot_df[top_columns].copy()
    pruned_one_hot_encoded_df['Other'] = one_hot_df[rest_of_columns].sum(axis=1)

    # Add the prefix 'Provisions_' to the column names
    pruned_one_hot_encoded_df.columns = ['Provisions_' + col for col in pruned_one_hot_encoded_df.columns]

    # Add these columns to the original DataFrame
    df = pd.concat([df, pruned_one_hot_encoded_df], axis=1)

    return df


def insert_top_ipc_flags(df, ipcs_column: str, top_n: int = 15, ipc_length=5,
                         tokens_mapper: dict = None) -> pd.DataFrame:
    """
    Inserts top_n IPC flags into the DataFrame based on the given IPCs column.

    Parameters:
    - df: pandas.DataFrame
    - ipcs_column: name of the IPCs column in the DataFrame
    - top_n: number of top IPCs to keep
    - ipc_length: length of the IPC string to keep
    - tokens_mapper: dictionary mapping tokens to indices

    Returns:
    - DataFrame with new IPC features
    """

    df.copy()

    df[ipcs_column] = df[ipcs_column].apply(lambda x: ', '.join([x[:ipc_length]]))

    if tokens_mapper is None:
        tokens_mapper = get_token_map(df, ipcs_column)

    # Initialize an empty DataFrame with the same index as df
    one_hot_df = pd.DataFrame(0, index=df.index, columns=tokens_mapper.keys())

    for i in range(len(df)):
        tokens = custom_tokenize(df.iloc[i][ipcs_column])
        for token in tokens:
            one_hot_df.iloc[i][token] = 1

    column_totals = one_hot_df.sum(axis=0)
    top_columns = column_totals.nlargest(top_n).index.tolist()

    rest_of_columns = [col for col in one_hot_df.columns if col not in top_columns]

    pruned_one_hot_encoded_df = one_hot_df[top_columns].copy()
    pruned_one_hot_encoded_df['Other'] = one_hot_df[rest_of_columns].sum(axis=1)

    # Add the prefix 'IPC_' to the column names
    pruned_one_hot_encoded_df.columns = ['IPC_' + col for col in pruned_one_hot_encoded_df.columns]

    # Add these columns to the original DataFrame
    df = pd.concat([df, pruned_one_hot_encoded_df], axis=1)

    return df


def custom_ipc_one_hot_encoder(df, series_name: str, tokens_mapper: dict, keep_first_n_ipc_letters=6) -> pd.DataFrame:
    """
    Custom one-hot encoder that uses the provided tokens mapper to create a one-hot encoded DataFrame.
    """

    df[series_name] = df[series_name].apply(lambda x: ', '.join([x[:keep_first_n_ipc_letters]]))
    if tokens_mapper is None:
        tokens_mapper = get_token_map(df, series_name)

    # Initialize an empty DataFrame with the same index as df
    one_hot_df = pd.DataFrame(0, index=df.index, columns=tokens_mapper.keys())

    for i in range(len(df)):
        tokens = custom_tokenize(df.iloc[i][series_name])
        for token in tokens:
            one_hot_df.iloc[i][token] = 1

    return one_hot_df


def insert_top_opponents_flags(df, opponents_column: str, top_n: int = 15, tokens_mapper: dict = None,
                               drop_opponents_column=True) -> pd.DataFrame:
    """
    Inserts top_n opponents flags into the DataFrame based on the given opponents column.

    Parameters:
    - df: pandas.DataFrame
    - opponents_column: name of the opponents column in the DataFrame
    - top_n: number of top opponents to keep
    - tokens_mapper: dictionary mapping tokens to indices

    Returns:
    - DataFrame with new opponent features
    """

    df.copy()

    if tokens_mapper is None:
        tokens_mapper = get_token_map(df, opponents_column)

    # Initialize an empty DataFrame with the same index as df
    one_hot_df = pd.DataFrame(0, index=df.index, columns=tokens_mapper.keys())

    for i in range(len(df)):
        tokens = df.iloc[i][opponents_column].replace(', Inc.', ' Inc.').split(', ')
        for token in tokens:
            one_hot_df.iloc[i][token] = 1

    column_totals = one_hot_df.sum(axis=0)
    top_columns = column_totals.nlargest(top_n).index.tolist()

    rest_of_columns = [col for col in one_hot_df.columns if col not in top_columns]

    pruned_one_hot_encoded_df = one_hot_df[top_columns].copy()
    pruned_one_hot_encoded_df['Other'] = one_hot_df[rest_of_columns].sum(axis=1)

    # Add the prefix 'Opponent_' to the column names
    pruned_one_hot_encoded_df.columns = ['Opponent_' + col for col in pruned_one_hot_encoded_df.columns]

    # Add these columns to the original DataFrame
    df = pd.concat([df, pruned_one_hot_encoded_df], axis=1)

    if drop_opponents_column:
        df.drop(columns=[opponents_column], inplace=True)

    return df


def get_top_one_hot_encoded_ipcs(df, series_name: str, top_n: int = 15, ipc_length=5,
                                 tokens_mapper: dict = None) -> pd.DataFrame:
    one_hot_encoded_ipcs = custom_ipc_one_hot_encoder(df=df, series_name=series_name, tokens_mapper=tokens_mapper,
                                                      keep_first_n_ipc_letters=top_n)

    column_totals = one_hot_encoded_ipcs.sum(axis=0)
    top_ipcs = column_totals.nlargest(top_n).index.tolist()

    rest_of_ipcs = [col for col in one_hot_encoded_ipcs.columns if col not in top_ipcs]

    pruned_one_hot_encoded_df = one_hot_encoded_ipcs[top_ipcs].copy()
    pruned_one_hot_encoded_df['Other'] = one_hot_encoded_ipcs[rest_of_ipcs].sum(axis=1)

    return pruned_one_hot_encoded_df


def insert_date_features(df, date_column: str, date_format: str, keep_year: bool = True,
                         keep_month: bool = True) -> pd.DataFrame:
    """
    Inserts year and month features into the DataFrame based on the given date column.

    Parameters:
    - df: pandas.DataFrame
    - date_column: name of the date column in the DataFrame
    - date_format: format of the date string
    - keep_year: whether to keep the year feature
    - keep_month: whether to keep the month feature

    Returns:
    - DataFrame with new year and month features
    """

    df.copy()
    df[date_column] = pd.to_datetime(df[date_column], format=date_format, errors='coerce')

    if keep_year:
        df['year'] = df[date_column].dt.year

    if keep_month:
        df['month'] = df[date_column].dt.month

    return df.drop(columns=[date_column])


def custom_tokenize(text: str):
    """
    Custom tokenizer that uses regex to split the text into tokens.
    Replaces all spaces with underscores before tokenization.
    """
    # Replace spaces with underscores
    text = text.replace(" ", "")

    # Define a regex pattern for tokenization
    pattern = r"\s|[\.,;']"
    tokens = regexp_tokenize(text, pattern, gaps=True)
    return tokens


def custom_one_hot_encoder(df, series_name: str, tokens_mapper: dict = None) -> pd.DataFrame:
    """
    Custom one-hot encoder that uses the provided tokens mapper to create a one-hot encoded DataFrame.
    """
    if tokens_mapper is None:
        tokens_mapper = get_token_map(df, series_name)

    # Initialize an empty DataFrame with the same index as df
    one_hot_df = pd.DataFrame(0, index=df.index, columns=tokens_mapper.keys())

    for i in range(len(df)):
        tokens = custom_tokenize(df.iloc[i][series_name])
        for token in tokens:
            one_hot_df.iloc[i][token] = 1

    return one_hot_df


def get_top_one_hot_encoded_values(df, series_name: str, top_n: int = 35, tokens_mapper: dict = None) -> pd.DataFrame:
    one_hot_encoded_df = custom_one_hot_encoder(df=df, series_name=series_name, tokens_mapper=tokens_mapper)

    column_totals = one_hot_encoded_df.sum(axis=0)
    top_columns = column_totals.nlargest(top_n).index.tolist()

    rest_of_columns = [col for col in one_hot_encoded_df.columns if col not in top_columns]

    pruned_one_hot_encoded_df = one_hot_encoded_df[top_columns].copy()
    pruned_one_hot_encoded_df['Other'] = one_hot_encoded_df[rest_of_columns].sum(axis=1)

    return pruned_one_hot_encoded_df


def insert_num_opponents_representatives(df, opponents_column: str, representatives_column: str) -> pd.DataFrame:
    """
    Inserts the number of opponents and representatives into the DataFrame.

    Parameters:
    - df: pandas.DataFrame
    - opponents_column: name of the opponents column in the DataFrame
    - representatives_column: name of the representatives column in the DataFrame

    Returns:
    - DataFrame with new opponent and representative count features
    """

    df.copy()

    # Get the number of opponents
    df['num_opponents'] = get_opponent_count(df)

    # Get the number of representatives
    df['num_representatives'] = get_representative_count(df)

    return df


def get_opponent_count(df) -> pd.Series:
    opponent_cols = [f'Opponent {i + 1}' for i in range(20)]
    nOpponents = 20 - df[opponent_cols].isna().sum(axis=1)
    return nOpponents


def get_representative_count(df) -> pd.Series:
    representative_cols = [f'Representative {i + 1}' for i in range(20)]
    nRepresentatives = 20 - df[representative_cols].isna().sum(axis=1)
    return nRepresentatives


def get_columns_to_keep() -> list:
    """
    Returns a list of columns to keep in the DataFrame.
    """
    return ['Decision date', 'IPC pharma', 'IPCs',
            'Language', 'Title of Invention', 'Patent Proprietor', 'Headword',
            'Provisions', 'Summary', 'Order status manual', 'Opponents']


def tfidf_transform(series,
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=0.01,
                    max_df=0.8):
    """
    Transforms a text series into TF-IDF matrix.

    Parameters:
    - series: pandas.Series of text data
    - max_features: max number of features
    - stop_words: stopword strategy ('english' or custom list)
    - ngram_range: tuple for n-gram size, e.g., (1,2) for unigrams + bigrams
    - min_df: min number of docs a term must appear in
    - max_df: max proportion of docs a term can appear in

    Returns:
    - tfidf_matrix (sparse matrix)
    - feature_names (list of feature/column names)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words=stop_words,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df
    )

    tfidf_matrix = vectorizer.fit_transform(series.fillna(''))
    feature_names = vectorizer.get_feature_names_out()

    return tfidf_matrix, feature_names


def insert_tfidf_features(df, text_column: str, ngram_range=(1, 2), min_df=0.01, max_df=0.8) -> pd.DataFrame:
    """
    Inserts TF-IDF features into the DataFrame based on the given text column.

    Parameters:
    - df: pandas.DataFrame
    - text_column: name of the text column in the DataFrame
    - ngram_range: tuple for n-gram size, e.g., (1,2) for unigrams + bigrams
    - min_df: min number of docs a term must appear in
    - max_df: max proportion of docs a term can appear in

    Returns:
    - DataFrame with new TF-IDF features
    """

    tfidf_matrix, feature_names = tfidf_transform(df[text_column],
                                                  ngram_range=ngram_range,
                                                  min_df=min_df,
                                                  max_df=max_df
                                                  )

    # Convert sparse matrix to DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    # Add these columns to the original DataFrame
    tfidf_df.index = df.index  # Ensure the indices match
    df = pd.concat([df, tfidf_df], axis=1)

    return df


def drop_duplicate_languages(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function returns unique cases without duplication across languages. Keeps only English if English + other lang is avbl, then keeps only german, and takes french for the rest.
    """
    en_cases = df[df['Language'] == 'EN']

    # Find the case numbers in df that are not in en_cases
    case_numbers = [x for x in df['Case number'].unique() if x not in en_cases['Case number'].unique()]

    # For the rest of the cases, keep the german versions only (in case they exist in both DE and FR)
    rest_de_df = df[(df['Case number'].isin(case_numbers)) & (df['Language'] == 'DE')]

    all_cases = pd.concat([en_cases, rest_de_df])

    # If there are any cases in french only, add them last.
    rest_fr_df = df[(~df['Case number'].isin(all_cases['Case number'])) & (df['Language'] == 'FR')]

    all_cases = pd.concat([all_cases, rest_fr_df])

    return all_cases


def load_tfidf_preprocessed_data(filepath='data/preprocessed/all_en_df.pkl'):
    print("Applying following preprocessing logic:\nSummary, headword, and Title are translated"
          "\nConcatenate Title and Summary \nCreate ID column\nGenerate count for nOpponents and nReps"
          "\nKeep Year and Month only\nClean up Proprietor (manual)\nKeep top 35 provisions, other"
          "\nKeep top 15 IPCs, Keep top 15 Opponents plus others as Binary flags\nApply TF-IDF on Summary+Title"
          "\nApply TFIDF on headword")
    raw_data = data_loader.load_translated_dataset(filepath, file_is_csv=False)

    raw_data['text_content'] = raw_data['Title of Invention'] + ':\n\n' + raw_data['Summary']

    raw_data['id'] = raw_data['Decision date'] + "_" + raw_data['Case number'] + "_" + raw_data[
        'Application number'] + "_" + raw_data['Publication number']

    raw_data['nOpponents'] = get_opponent_count(raw_data)
    raw_data['nRepresentatives'] = get_representative_count(raw_data)

    keep_columns = get_columns_to_keep()
    raw_data = raw_data[keep_columns + ['text_content']]

    raw_data = get_top_proprietor_info_deduped(raw_data)

    raw_data = insert_date_features(raw_data, date_column='Decision date', date_format='%Y-%m-%d', keep_year=True,
                                    keep_month=True)

    # Get the one-hot encoding for categorical variables (a) Provisions (35) , (b) IPCs (15 @5, 10@4?, 13@6) , and (c)
    raw_data = insert_top_provisions_flags(raw_data, provisions_column='Provisions', top_n=35, tokens_mapper=None)

    raw_data = insert_top_ipc_flags(raw_data, ipcs_column='IPCs', top_n=15, ipc_length=6, tokens_mapper=None)

    raw_data = insert_top_opponents_flags(raw_data, opponents_column='Opponents', top_n=15, tokens_mapper=None,
                                          drop_opponents_column=True)
    # Get the number of opponents

    # raw_data = insert_num_opponents_representatives(raw_data, opponents_column='Opponents', representatives_column='Representatives')

    # Get the TF-IDF vectorization of the text content
    raw_data = insert_tfidf_features(raw_data,
                                     text_column='text_content',
                                     ngram_range=(1, 2),
                                     min_df=0.20,
                                     max_df=0.9)

    raw_data = insert_tfidf_features(raw_data, text_column='Headword', ngram_range=(1, 1), min_df=0.01, max_df=0.8)

    # Finally, drop the following columns: Title of Invention, Summary, text_content, Headword, IPCs
    raw_data = raw_data.drop(
        columns=['Title of Invention', 'Provisions', 'Summary', 'text_content', 'Headword', 'IPCs'])

    raw_data = raw_data[raw_data['Order status manual'].isin(
        ['appeal dismissed', 'decision under appeal is set aside', 'patent revoked'])]

    # one hot encode the cleaned proprietor, language
    one_hot_encoded_language_df = pd.get_dummies(raw_data['Language'], prefix='Language', dtype=float)

    one_hot_encoded_proprietor_df = pd.get_dummies(raw_data['Patent Proprietor'], prefix='Proprietor', dtype=float)

    # Add these columns to the original DataFrame
    raw_data = pd.concat([raw_data, one_hot_encoded_language_df, one_hot_encoded_proprietor_df], axis=1)
    raw_data.drop(columns=['Language', 'Patent Proprietor'], inplace=True)

    return raw_data


def merge_embedded_data(embeded_df: pd.DataFrame, basic_filepath='data/preprocessed/all_en_df.pkl'):
    print("Applying following preprocessing logic:\nSummary, headword, and Title are translated"
          "\n\nCreate ID column\nGenerate count for nOpponents and nReps"
          "\nKeep Year and Month only\nClean up Proprietor (manual)\nKeep top 35 provisions, other"
          "\nKeep top 15 IPCs, Keep top 15 Opponents plus others as Binary flags\nApply TF-IDF on Summary+Title"
          "\nApply TFIDF on headword")

    raw_data = data_loader.load_translated_dataset(basic_filepath, file_is_csv=False)
    raw_data['title_summary_embedding'] = embeded_df['title_summary_embedding']

    #
    #
    # raw_data['id'] = raw_data['Decision date'] + "_" + raw_data['Case number'] + "_" + raw_data[
    #     'Application number'] + "_" + raw_data['Publication number']

    raw_data['nOpponents'] = get_opponent_count(raw_data)
    raw_data['nRepresentatives'] = get_representative_count(raw_data)


    raw_data = get_top_proprietor_info_deduped(raw_data)

    raw_data = insert_date_features(raw_data, date_column='Decision date', date_format='%Y-%m-%d', keep_year=True,
                                    keep_month=True)

    # Get the one-hot encoding for categorical variables (a) Provisions (35) , (b) IPCs (15 @5, 10@4?, 13@6) , and (c)
    raw_data = insert_top_provisions_flags(raw_data, provisions_column='Provisions', top_n=35, tokens_mapper=None)

    raw_data = insert_top_ipc_flags(raw_data, ipcs_column='IPCs', top_n=15, ipc_length=6, tokens_mapper=None)

    raw_data = insert_top_opponents_flags(raw_data, opponents_column='Opponents', top_n=15, tokens_mapper=None,
                                          drop_opponents_column=True)


    raw_data = insert_tfidf_features(raw_data, text_column='Headword', ngram_range=(1, 1), min_df=0.01, max_df=0.8)

    # Finally, drop the following columns: Title of Invention, Summary, text_content, Headword, IPCs
    raw_data = raw_data.drop(
        columns=['Title of Invention', 'Provisions', 'Summary', 'Headword', 'IPCs'])

    raw_data = raw_data[raw_data['Order status manual'].isin(
        ['appeal dismissed', 'decision under appeal is set aside', 'patent revoked'])]

    # one hot encode the cleaned proprietor, language
    one_hot_encoded_language_df = pd.get_dummies(raw_data['Language'], prefix='Language', dtype=float)

    one_hot_encoded_proprietor_df = pd.get_dummies(raw_data['Patent Proprietor'], prefix='Proprietor', dtype=float)

    # Add these columns to the original DataFrame
    raw_data = pd.concat([raw_data, one_hot_encoded_language_df, one_hot_encoded_proprietor_df], axis=1)
    raw_data.drop(columns=['Language', 'Patent Proprietor'], inplace=True)

    # Drop 20 col pairs
    raw_data = raw_data.drop(columns=[f'Opponent {i + 1}' for i in range(20)])
    raw_data = raw_data.drop(columns=[f'Representative {i + 1}' for i in range(20)])

    drop_cols = ['Case number', 'Application number', 'Publication number','IPC biosimilar', 'Keywords', 'Decisions cited', 'Decision reasons',
       'Order', 'Order status', 'Order status web']#, 'id']
    raw_data = raw_data.drop(columns=drop_cols)

    return raw_data
