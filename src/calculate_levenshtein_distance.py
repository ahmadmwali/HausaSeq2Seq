import pandas as pd
import re
from nltk.corpus import words as nltk_words
from nltk import download
from collections import defaultdict, Counter
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import textdistance
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os

download('words')
english_vocab = set(w.lower() for w in nltk_words.words())

def extract_lowercase_words(texts):
    """Extracts all lowercase purely alphabetic words from a list of texts."""
    words = []
    for text in texts:
        tokens = re.findall(r'\b[a-z]+\b', str(text).lower())  # only lowercase alpha words
        words.extend(tokens)
    return words

def bucket_words(word_list):
    """Buckets words based on their length into length ranges (L-1 to L+1)."""
    buckets = defaultdict(list)
    for word in word_list:
        # Key is defined by length-1 and length+1, e.g., a 5-char word goes into "4_6"
        key = f"{len(word)-1}_{len(word)+1}"
        buckets[key].append(word)
    return buckets

def compute_clusters(buckets, max_words=100):
    """
    Clusters typos within each bucket based on Normalized Levenshtein distance
    using DBSCAN. Returns pairwise distances and cluster sizes.
    """
    cluster_sizes = []
    distances = []

    for key, words in tqdm(buckets.items(), desc="Clustering Buckets"):
        if len(words) < 2:
            continue
        sampled_words = list(set(words))[:max_words]  # limit size per bucket for efficiency
        if len(sampled_words) < 2:
            continue
        
        # Compute pairwise normalized Levenshtein distance
        dist_matrix = pairwise_distances(
            [[w] for w in sampled_words],
            metric=lambda a, b: textdistance.levenshtein.normalized_distance(a[0], b[0])
        )
        
        # Collect upper triangular part of distance matrix (excluding diagonal)
        distances.extend(dist_matrix[np.triu_indices(len(sampled_words), 1)])
        
        # Cluster using DBSCAN
        clustering = DBSCAN(eps=0.4, min_samples=2, metric='precomputed')
        labels = clustering.fit_predict(dist_matrix)
        counts = Counter(labels)
        for label, count in counts.items():
            if label != -1:  # ignore noise
                cluster_sizes.append(count)
    return distances, cluster_sizes

def plot_results(natural_dists, synthetic_dists, natural_clusters, synthetic_clusters, output_dir):
    """Generates and saves the comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set(style="whitegrid")

    # Normalized Levenshtein Distance Distribution
    plt.figure(figsize=(12,5))
    sns.histplot(natural_dists, bins=30, kde=True, color="blue", label="Natural")
    sns.histplot(synthetic_dists, bins=30, kde=True, color="red", label="Synthetic", alpha=0.5)
    plt.title("Normalized Levenshtein Distance Distribution")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "levenshtein_distance_distribution.png"))
    print(f"Saved distance distribution plot to {os.path.join(output_dir, 'levenshtein_distance_distribution.png')}")
    plt.close()

    # Typo Cluster Size Distribution
    plt.figure(figsize=(12,5))
    sns.histplot(natural_clusters, bins=30, color="blue", label="Natural", alpha=0.8, stat="density", common_norm=False)
    sns.histplot(synthetic_clusters, bins=30, color="red", label="Synthetic", alpha=0.5, stat="density", common_norm=False)
    plt.title("Typo Cluster Sizes (DBSCAN $\\epsilon=0.4$, $min\_samples=2$)")
    plt.xlabel("Cluster Size")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "typo_cluster_size_distribution.png"))
    print(f"Saved cluster size distribution plot to {os.path.join(output_dir, 'typo_cluster_size_distribution.png')}")
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate and compare Levenshtein distance and cluster sizes between natural and synthetic typos.")
    parser.add_argument(
        "--natural_file",
        type=str,
        default="/content/Files/sent_train.tsv",
        help="Path to the natural language dataset TSV file (e.g., Twitter data)."
    )
    parser.add_argument(
        "--synthetic_file",
        type=str,
        default="/content/Files/train.tsv",
        help="Path to the synthetic noise dataset TSV file."
    )
    parser.add_argument(
        "--natural_text_col",
        type=str,
        default="tweet",
        help="Name of the column containing text in the natural dataset."
    )
    parser.add_argument(
        "--synthetic_text_col",
        type=str,
        default="Noisy",
        help="Name of the column containing noisy text in the synthetic dataset."
    )
    parser.add_argument(
        "--max_words_per_bucket",
        type=int,
        default=100,
        help="Maximum number of unique words to sample per length bucket for clustering (to manage runtime)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./levenshtein_analysis_results",
        help="Directory to save the generated plots."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load your datasets
    print(f"Loading natural data from: {args.natural_file}")
    natural_df = pd.read_csv(args.natural_file, sep='\t')
    print(f"Loading synthetic data from: {args.synthetic_file}")
    synthetic_df = pd.read_csv(args.synthetic_file, sep='\t')

    # Subsample synthetic dataset to match Twitter dataset size
    if len(synthetic_df) > len(natural_df):
        print(f"Subsampling synthetic data from {len(synthetic_df)} to match natural data size: {len(natural_df)}")
        synthetic_df = synthetic_df.sample(n=len(natural_df), random_state=42)

    # Extract lowercase words
    print("Extracting words...")
    natural_words = extract_lowercase_words(natural_df[args.natural_text_col])
    synthetic_words = extract_lowercase_words(synthetic_df[args.synthetic_text_col])

    # Remove words that are in standard English vocabulary
    print("Filtering out English vocabulary...")
    natural_typos = [w for w in natural_words if w not in english_vocab and len(w) > 2]
    synthetic_typos = [w for w in synthetic_words if w not in english_vocab and len(w) > 2]
    
    print(f"Natural typos count: {len(natural_typos)}, Unique: {len(set(natural_typos))}")
    print(f"Synthetic typos count: {len(synthetic_typos)}, Unique: {len(set(synthetic_typos))}")

    # Bucket typos based on word length
    natural_buckets = bucket_words(natural_typos)
    synthetic_buckets = bucket_words(synthetic_typos)

    # Cluster typos based on normalized Levenshtein distance
    print(f"\nComputing clusters and distances for natural data (max_words/bucket={args.max_words_per_bucket})...")
    natural_dists, natural_clusters = compute_clusters(natural_buckets, max_words=args.max_words_per_bucket)
    
    print(f"\nComputing clusters and distances for synthetic data (max_words/bucket={args.max_words_per_bucket})...")
    synthetic_dists, synthetic_clusters = compute_clusters(synthetic_buckets, max_words=args.max_words_per_bucket)
    
    # Visualization
    print("\nGenerating plots...")
    plot_results(natural_dists, synthetic_dists, natural_clusters, synthetic_clusters, args.output_dir)

    # Top 10 most common words
    print("\n--- Top 10 Common Typos ---")
    print("Top 10 Natural (Twitter) words:", Counter(natural_typos).most_common(10))
    print("Top 10 Synthetic words:", Counter(synthetic_typos).most_common(10))

if __name__ == "__main__":
    main()

# Example Usage:
#
# 1. Run with default file paths (assuming files are in /content/Files):
# python calculate_levenshtein_distance.py
#
# 2. Run with custom file paths and output directory:
# python calculate_levenshtein_distance.py \
#     --natural_file /path/to/my/twitter_data.tsv \
#     --synthetic_file /path/to/my/synthetic_data.tsv \
#     --natural_text_col 'text_column' \
#     --synthetic_text_col 'error_column' \
#     --max_words_per_bucket 200 \
#     --output_dir ./custom_analysis_plots
#
# 3. Run with local files (assuming files are in the current directory, if running locally outside of Colab):
# python calculate_levenshtein_distance.py \
#     --natural_file ./sent_train.tsv \
#     --synthetic_file ./train.tsv \
#     --output_dir ./local_results