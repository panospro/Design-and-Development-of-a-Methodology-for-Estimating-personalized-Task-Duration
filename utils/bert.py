import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
import joblib
import os
from sklearn.decomposition import PCA
import itertools
import re

def run_bert_for_all_combinations(df):
    # Remove all those and keep the above
    columns = ["title", "body", "comments", "commitMessages"]
    # Generate all combinations of the columns
    combinations = []
    for r in range(1, len(columns) + 1):
        combinations.extend(itertools.combinations(columns, r))

    # Sort combinations based on the order in 'columns'
    sorted_combinations = sorted(combinations, key=lambda comb: [columns.index(col) for col in comb])

    # Run the get_bert_embeddings function for each combination
    for combination in sorted_combinations:
        columns_list = list(combination)
        print(columns_list)
        get_bert_embeddings(df, n_components=10, columns=columns_list, method='stand')

# Apply pca on bert data
def apply_pca(df, n_components=10, method='stand'):
    # Apply PCA to reduce the dimensionality of the embeddings
    embedding_columns = df.columns[df.columns.str.match(r'^\d+$')]  # Assumes embedding columns are named as numbers
    print(f"Applying PCA on {len(embedding_columns)} embedding columns.")

    if len(embedding_columns) == 0:
        print("No embedding columns found for PCA.")
        return df

    # Choose the scaler based on the method
    if method == 'stand':
        scaler = StandardScaler()
        print("Using StandardScaler for standardization.")
    elif method == 'norm':
        scaler = MinMaxScaler()
        print("Using MinMaxScaler for normalization.")
    else:
        raise ValueError("Invalid method. Choose 'stand' for standardization or 'norm' for normalization.")

    scaled_embeddings = scaler.fit_transform(df[embedding_columns])

    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(scaled_embeddings)
    
    # Create DataFrame for PCA components
    pca_columns = [f'pca_{i}' for i in range(pca_embeddings.shape[1])]
    pca_df = pd.DataFrame(pca_embeddings, columns=pca_columns, index=df.index)
    
    # Drop the original embedding columns
    df = df.drop(columns=embedding_columns)
    
    # Concatenate the PCA embeddings with the original dataframe
    df = pd.concat([df, pca_df], axis=1)
    
    print(f"PCA reduced the dimensionality to {pca_df.shape[1]} components.")
    
    return df

# Check if cached embeddings exist, else create bert analyze
def get_bert_embeddings(df, n_components, columns, method, max_length=512):
    def combine_text_columns(df, columns):
        CLEAN_TEXT = re.compile(r'@\w+|https?://\S+|www\.\S+|[^\w\s.,!?\'-]|_')
        REMOVE_IMAGE_LINKS = re.compile(r'!\[.*?\]\(.*?\)')
        REMOVE_MARKDOWN_LINKS = re.compile(r'\[.*?\]\(.*?\)')

        def preprocess_text(text):
            """Clean and prepare text for tokenization by lowercasing, removing mentions, URLs, 
            image links, markdown links, and non-alphanumeric characters (except for spaces and basic punctuation), 
            and normalizing whitespace."""
            text = REMOVE_IMAGE_LINKS.sub('', text)
            text = REMOVE_MARKDOWN_LINKS.sub('', text)
            cleaned_text = CLEAN_TEXT.sub(' ', text.lower()).strip()
            # Replace multiple spaces with a single space and ensure proper spacing after periods
            cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)
            cleaned_text = re.sub(r'\.\s*', '. ', cleaned_text)
            return cleaned_text

        df['combined_text'] = df[columns].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
        df['combined_text'] = df['combined_text'].apply(preprocess_text)
        combined_text_df = df[['combined_text']]
        combined_text_df.to_csv('combined_text.csv', index=False)
        return df

    def encode_text(text):
        inputs = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy().flatten()

    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    cache_file = f"bert_results/bert_embeddings.pkl"

    if not columns:
        return df.drop(columns=["comments", "title", "body", "commitMessages"])

    if os.path.exists(cache_file):
        combine_text_columns(df.copy(), columns)
        print(f"Loading cached BERT embeddings for {columns}...")
        embeddings_df = joblib.load(cache_file)
    else:
        # Combine text columns into a single column
        df_combined = combine_text_columns(df.copy(), columns)

        # Generate BERT embeddings for the combined text with a progress bar
        tqdm.pandas(desc="Generating BERT embeddings")
        df_combined['bert_embeddings'] = df_combined['combined_text'].progress_apply(encode_text)

        # Convert embeddings into separate columns
        embeddings_df = pd.DataFrame(df_combined['bert_embeddings'].tolist(), index=df_combined.index)

        # Drop the original text columns, combined text column, bert embeddings, and other unused columns
        df_combined = df_combined.drop(columns=["comments", "title", "body", "commitMessages", "combined_text", "bert_embeddings"])

        # Concatenate the embeddings with the original dataframe
        embeddings_df = pd.concat([df_combined, embeddings_df], axis=1)

        # Ensure all column names are strings
        embeddings_df.columns = embeddings_df.columns.astype(str)

        # Cache the embeddings
        joblib.dump(embeddings_df, cache_file)
        print(f"BERT embeddings cached for {columns}.")

    return apply_pca(embeddings_df, n_components, method)
