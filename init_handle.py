import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import time

# Configuration
DATA_DIR = "."
OUTPUT_DIR = "processed_data"
ABT_FILE = os.path.join(DATA_DIR, "Abt.csv")
BUY_FILE = os.path.join(DATA_DIR, "Buy.csv")
MAPPING_FILE = os.path.join(DATA_DIR, "abt_buy_perfectMapping.csv")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "entity_pairs.csv")

# Simple Stopwords List (English)
STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
])

def tokenize(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    # Split by non-alphanumeric
    tokens = re.split(r'[^a-z0-9]+', text)
    return [t for t in tokens if t and t not in STOPWORDS]

def get_jaccard_sim(str1, str2):
    set1 = set(tokenize(str1))
    set2 = set(tokenize(str2))
    if not set1 and not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def get_levenshtein_sim(str1, str2):
    if not isinstance(str1, str): str1 = ""
    if not isinstance(str2, str): str2 = ""
    s1 = str1.lower()
    s2 = str2.lower()
    if not s1 and not s2:
        return 1.0
    dist = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1 - (dist / max_len) if max_len > 0 else 1.0

def apply_blocking(abt, buy):
    print("Applying Token Blocking...")
    # Tokenize all names
    abt['tokens'] = abt['name'].apply(lambda x: set(tokenize(x)))
    buy['tokens'] = buy['name'].apply(lambda x: set(tokenize(x)))
    
    # Create inverted index for Buy
    token_to_buy_indices = {}
    for idx, tokens in zip(buy.index, buy['tokens']):
        for token in tokens:
            if token not in token_to_buy_indices:
                token_to_buy_indices[token] = []
            token_to_buy_indices[token].append(idx)
            
    # Find pairs
    pairs = set()
    for abt_idx, tokens in zip(abt.index, abt['tokens']):
        candidates = set()
        for token in tokens:
            if token in token_to_buy_indices:
                candidates.update(token_to_buy_indices[token])
        
        for buy_idx in candidates:
            pairs.add((abt.loc[abt_idx, 'id'], buy.loc[buy_idx, 'id']))
            
    print(f"Blocking reduced pairs from {len(abt)*len(buy)} to {len(pairs)}")
    
    # Convert to DataFrame
    pair_list = list(pairs)
    pairs_df = pd.DataFrame(pair_list, columns=['idAbt', 'idBuy'])
    
    # Merge back details
    pairs_df = pd.merge(pairs_df, abt[['id', 'name', 'description']].rename(columns={'id': 'idAbt', 'name': 'nameAbt', 'description': 'descAbt'}), on='idAbt')
    pairs_df = pd.merge(pairs_df, buy[['id', 'name', 'description']].rename(columns={'id': 'idBuy', 'name': 'nameBuy', 'description': 'descBuy'}), on='idBuy')
    
    return pairs_df

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--blocking', action='store_true', help='Enable blocking')
    args = parser.parse_args()

    print("Loading data...")
    try:
        abt = pd.read_csv(ABT_FILE, encoding='ISO-8859-1')
        buy = pd.read_csv(BUY_FILE, encoding='ISO-8859-1')
        mapping = pd.read_csv(MAPPING_FILE)
    except Exception as e:
        print(f"Error loading files: {e}")
        # Try default encoding if ISO-8859-1 fails
        abt = pd.read_csv(ABT_FILE)
        buy = pd.read_csv(BUY_FILE)
        mapping = pd.read_csv(MAPPING_FILE)

    print(f"Abt shape: {abt.shape}")
    print(f"Buy shape: {buy.shape}")
    
    # Handle NaNs
    abt['name'] = abt['name'].fillna('')
    abt['description'] = abt['description'].fillna('')
    buy['name'] = buy['name'].fillna('')
    buy['description'] = buy['description'].fillna('')

    # Create Cartesian Product
    if args.blocking:
        pairs = apply_blocking(abt, buy)
        global OUTPUT_FILE
        OUTPUT_FILE = os.path.join(OUTPUT_DIR, "entity_pairs_blocked.csv")
    else:
        print("Generating Cartesian Product...")
        abt['key'] = 1
        buy['key'] = 1
        # Only keep necessary columns to save memory
        abt_subset = abt[['id', 'name', 'description', 'key']].rename(columns={'id': 'idAbt', 'name': 'nameAbt', 'description': 'descAbt'})
        buy_subset = buy[['id', 'name', 'description', 'key']].rename(columns={'id': 'idBuy', 'name': 'nameBuy', 'description': 'descBuy'})
        
        # Warning: This creates M*N rows.
        pairs = pd.merge(abt_subset, buy_subset, on='key').drop('key', axis=1)
    
    print(f"Total pairs generated: {len(pairs)}")

    # Labeling
    print("Labeling pairs...")
    # Create a set of true pairs for fast lookup
    true_pairs = set(zip(mapping['idAbt'], mapping['idBuy']))
    
    # Vectorized labeling
    # We need to ensure types match. IDs might be int or str.
    # Check types in mapping
    # mapping ids seem to be int based on previous `head`.
    
    def check_label(row):
        return 1 if (row['idAbt'], row['idBuy']) in true_pairs else 0

    # Using map with a tuple index might be faster
    # Let's construct a MultiIndex on pairs to check
    # Or just simple apply (slow).
    # Faster: Merge with mapping
    mapping['label'] = 1
    # Ensure column names match for merge
    pairs = pd.merge(pairs, mapping[['idAbt', 'idBuy', 'label']], on=['idAbt', 'idBuy'], how='left')
    pairs['label'] = pairs['label'].fillna(0).astype(int)
    
    print("Computing Similarities...")
    
    # 1. Name Cosine Similarity
    # We can compute this efficiently using sklearn
    # Concatenate all names to build vocabulary
    print(" - Name Cosine...")
    corpus = pd.concat([abt['name'], buy['name']]).unique()
    vectorizer = CountVectorizer(stop_words='english')
    vectorizer.fit(corpus)
    
    # Transform
    # Doing this for 1M pairs individually is slow.
    # Better: Transform all unique Abt names and Buy names, then compute dot product for pairs.
    # But we have the pairs dataframe.
    # Let's compute vectors for the columns in pairs dataframe? No, too redundant.
    # We can use the indices.
    
    # Create a lookup for vectors
    abt_vecs = vectorizer.transform(abt['name'].fillna(''))
    buy_vecs = vectorizer.transform(buy['name'].fillna(''))
    
    # Map IDs to indices in the vector matrix
    abt_id_to_idx = {row['id']: i for i, row in abt.iterrows()}
    buy_id_to_idx = {row['id']: i for i, row in buy.iterrows()}
    
    # We need to compute cosine sim for each pair.
    # Sim(A, B) = (A . B) / (|A| |B|)
    # Sklearn cosine_similarity computes pairwise for matrices.
    # We can't compute full matrix (1000x1000) - wait, we CAN. It's small.
    # 1000x1000 is 1M floats, which is 4MB or 8MB. Totally fine.
    
    print("   (Computing full similarity matrix strategy)")
    name_sim_matrix = cosine_similarity(abt_vecs, buy_vecs) # Shape (nAbt, nBuy)
    
    # Now map these values back to the pairs dataframe
    # pairs has idAbt and idBuy.
    # We need to map idAbt -> row_idx, idBuy -> col_idx
    
    # Create mapping arrays
    pairs['abt_idx'] = pairs['idAbt'].map(abt_id_to_idx)
    pairs['buy_idx'] = pairs['idBuy'].map(buy_id_to_idx)
    
    # Extract values
    # numpy indexing: matrix[rows, cols]
    pairs['name_cos'] = name_sim_matrix[pairs['abt_idx'], pairs['buy_idx']]
    
    # 2. Description Cosine Similarity
    print(" - Description Cosine...")
    desc_vectorizer = CountVectorizer(stop_words='english')
    # Handle potential empty descriptions
    # fillna was done earlier
    desc_corpus = pd.concat([abt['description'], buy['description']]).unique()
    # Some descriptions might be empty strings, which is fine
    try:
        desc_vectorizer.fit(desc_corpus)
        abt_desc_vecs = desc_vectorizer.transform(abt['description'])
        buy_desc_vecs = desc_vectorizer.transform(buy['description'])
        desc_sim_matrix = cosine_similarity(abt_desc_vecs, buy_desc_vecs)
        pairs['desc_cos'] = desc_sim_matrix[pairs['abt_idx'], pairs['buy_idx']]
    except ValueError:
        # Case where vocabulary is empty
        pairs['desc_cos'] = 0.0

    # 3. Name Jaccard
    print(" - Name Jaccard (This might take a while)...")
    # Optimization: Pre-tokenize
    abt['name_tokens'] = abt['name'].apply(lambda x: set(tokenize(x)))
    buy['name_tokens'] = buy['name'].apply(lambda x: set(tokenize(x)))
    
    abt_token_lookup = abt.set_index('id')['name_tokens']
    buy_token_lookup = buy.set_index('id')['name_tokens']
    
    # Apply to pairs
    # Defining a function to call on rows is slow.
    # Using list comprehension with zip is faster.
    
    def calc_jaccard(s1, s2):
        if not s1 and not s2: return 0.0
        numer = len(s1.intersection(s2))
        denom = len(s1.union(s2))
        return numer / denom if denom > 0 else 0.0

    # Extract series
    p_abt_tokens = pairs['idAbt'].map(abt_token_lookup)
    p_buy_tokens = pairs['idBuy'].map(buy_token_lookup)
    
    pairs['name_jac'] = [calc_jaccard(a, b) for a, b in zip(p_abt_tokens, p_buy_tokens)]

    # 4. Description Jaccard
    print(" - Description Jaccard...")
    abt['desc_tokens'] = abt['description'].apply(lambda x: set(tokenize(x)))
    buy['desc_tokens'] = buy['description'].apply(lambda x: set(tokenize(x)))
    
    abt_desc_lookup = abt.set_index('id')['desc_tokens']
    buy_desc_lookup = buy.set_index('id')['desc_tokens']
    
    p_abt_desc_tokens = pairs['idAbt'].map(abt_desc_lookup)
    p_buy_desc_tokens = pairs['idBuy'].map(buy_desc_lookup)
    
    pairs['desc_jac'] = [calc_jaccard(a, b) for a, b in zip(p_abt_desc_tokens, p_buy_desc_tokens)]

    # 5. Name Edit Similarity (Levenshtein)
    print(" - Name Edit Similarity (This is the slowest part)...")
    # We can optimize by computing only for unique pairs if there are duplicates, but Cartesian product is unique pairs.
    # Pure python Levenshtein on 1M pairs is very slow.
    # Let's try to see if we can avoid computing for very different lengths? No, metric is needed.
    # We will accept the slowness or use a progress print.
    # To avoid hanging without info, I'll print progress.
    
    name_pairs = list(zip(pairs['nameAbt'], pairs['nameBuy']))
    n = len(name_pairs)
    lev_scores = []
    start_time = time.time()
    
    # Using a slightly faster approach?
    # If Levenshtein module is not available, we are stuck with python.
    # 1M pairs is too much for python loop in a reasonable generic "pair programming" session if it takes hours.
    # However, I must implement it.
    # I'll use a simple counter to print status every 10000 iterations.
    
    for i, (n1, n2) in enumerate(name_pairs):
        lev_scores.append(get_levenshtein_sim(n1, n2))
        if i % 50000 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (n - i) / rate
            print(f"   Processed {i}/{n} pairs. Est remaining: {remaining/60:.1f} min")

    pairs['name_lev'] = lev_scores

    # Save
    print(f"Saving to {OUTPUT_FILE}...")
    # Keep only necessary columns
    out_cols = ['idAbt', 'idBuy', 'name_cos', 'name_jac', 'name_lev', 'desc_cos', 'desc_jac', 'label']
    pairs[out_cols].to_csv(OUTPUT_FILE, index=False)
    
    # Stats
    pos_count = pairs['label'].sum()
    total_count = len(pairs)
    neg_count = total_count - pos_count
    print("-" * 30)
    print(f"Total Pairs: {total_count}")
    print(f"Positive Examples: {pos_count}")
    print(f"Negative Examples: {neg_count}")
    print("-" * 30)

if __name__ == "__main__":
    main()
