"""
calculate_levenshtein_distance.py

Validates the synthetic noise pipeline by comparing the error distribution of:
  - Natural noisy Twitter Hausa data  (reference)
  - Synthetically noised formal Hausa text  (generated)

If the two Levenshtein distance distributions align, it confirms that the
synthetic noise pipeline (Table 1 params) is a good proxy for real Twitter noise
— justifying its use as seq2seq training data for text normalisation.

Pipeline:
  1. Download clean formal Hausa text from GDrive via gdown.
  2. Load Twitter data  →  N = len(twitter_df).
  3. Randomly sample ~N sentences from the clean formal text.
  4. Apply synthetic noise (Table 1 params) to each sampled sentence.
  5. Compute pairwise normalised Levenshtein distances within length-bucketed
     word groups for both corpora.
  6. Run DBSCAN clustering and plot distance + cluster-size distributions.

Usage:
    python src/calculate_levenshtein_distance.py
"""

import re
import os
import random
import time
from collections import defaultdict, Counter

import gdown
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textdistance
from nltk.corpus import words as nltk_words
from nltk import download as nltk_download
from sklearn.cluster import DBSCAN
from tqdm import tqdm


# Configuration

# GDrive folder containing clean formal Hausa text files
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1o2Yg0PiU3nI7z24TaVkzu320g65JZfpO"
DOWNLOAD_DIR      = "./data/supplementary/gdrive_clean_text"

# Paths
TWITTER_TRAIN_PATH = "./data/supplementary/twitter_data/train.tsv"
TOP_WORDS_PATH     = "./data/top_100_hausa_natural.txt"
PLOT_OUTPUT_DIR    = "./data/supplementary"

# Which column in the Twitter TSV holds the tweet text
TWEET_COL = "tweet"

# Files to exclude from the clean corpus (religious texts, per paper)
EXCLUDE_FILENAMES = {"Bible.txt", "Tanzil.txt"}

# Paper noise configuration from Table 1
PAPER_NOISE_LEVELS = {
    "random_spacing":        0.02,
    "remove_spaces":         0.15,
    "incorrect_characters":  0.02,
    "delete_characters":     0.005,
    "duplicate_characters":  0.01,
    "substitute_characters": 0.001,
    "transpose_characters":  0.01,
    "delete_chunk":          0.0015,
    "insert_chunk":          0.001,
}

# Word-level noising probabilities
PROB_NOISE_FREQUENT_WORD = 0.02
PROB_NOISE_OTHER_WORD    = 0.40

HAUSA_CHARACTERS = "abcdefghijklmnopqrstuvwxyz'ƙɗɓyABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# Data acquisition

def download_gdrive_folder(url: str, output_dir: str) -> str:
    """Download a GDrive folder using gdown. Skips if already present."""
    os.makedirs(output_dir, exist_ok=True)
    existing = [f for f in os.listdir(output_dir) if f.endswith(".txt")]
    if existing:
        print(f"Download directory already contains {len(existing)} .txt file(s). Skipping download.")
        return output_dir
    print(f"Downloading clean text from GDrive → {output_dir} …")
    gdown.download_folder(url, output=output_dir, quiet=False, use_cookies=False)
    print("  Download complete.")
    return output_dir


def load_clean_sentences(folder: str, exclude: set, encoding: str = "utf-8") -> list:
    """
    Read all .txt files in *folder* (excluding names in *exclude*),
    clean and segment into sentences. Returns a flat list of strings.
    """
    all_sentences = []
    txt_files = sorted(f for f in os.listdir(folder) if f.endswith(".txt"))
    if not txt_files:
        print(f"  [warn] No .txt files found in {folder}.")
        return all_sentences
    for fname in txt_files:
        if fname in exclude:
            print(f"  Skipping excluded file: {fname}")
            continue
        fpath = os.path.join(folder, fname)
        try:
            with open(fpath, "r", encoding=encoding, errors="replace") as fh:
                text = fh.read()
            cleaned   = _clean_text(text)
            sentences = _segment_sentences(cleaned)
            print(f"  {fname}: {len(sentences):,} sentences")
            all_sentences.extend(sentences)
        except Exception as exc:
            print(f"  [warn] Could not read {fpath}: {exc}")
    return all_sentences


def _clean_text(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"NBSP", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bshit\b", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _segment_sentences(text: str) -> list:
    pattern = re.compile(
        r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!)\s+"
    )
    sentences = []
    for para in text.split("\n"):
        para = para.strip()
        if para:
            sentences.extend(s.strip() for s in pattern.split(para) if s.strip())
    return sentences


# Character-level noise functions

def _random_spacing(text: str, p: float) -> str:
    if not p or not text:
        return text
    out = []
    for i, ch in enumerate(text):
        if ch.isalnum() and random.random() < p and i > 0 and text[i - 1] != " ":
            out.append(" ")
        out.append(ch)
    return re.sub(r" +", " ", "".join(out)).strip()


def _remove_spaces(text: str, p: float) -> str:
    if not p or not text:
        return text
    chars = list(text)
    for i in range(len(chars) - 2, 0, -1):
        if chars[i] == " " and chars[i - 1] != " " and chars[i + 1] != " ":
            if random.random() < p:
                chars.pop(i)
    return "".join(chars).strip()


def _incorrect_characters(text: str, p: float) -> str:
    if not p or not text:
        return text
    rmap = {"ɓ": "b", "ɗ": "d", "ƙ": "k", "ƴ": "y",
            "Ɓ": "B", "Ɗ": "D", "Ƙ": "K", "Ƴ": "Y"}
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch in rmap and random.random() < p:
            chars[i] = rmap[ch]
    return "".join(chars)


def _delete_characters(text: str, p: float) -> str:
    if not p or not text:
        return text
    return "".join(ch for ch in text if ch == " " or random.random() > p)


def _duplicate_characters(text: str, p: float) -> str:
    if not p or not text:
        return text
    out = []
    for ch in text:
        out.append(ch)
        if ch != " " and random.random() < p:
            out.append(ch)
    return "".join(out)


def _substitute_characters(text: str, p: float) -> str:
    if not p or not text:
        return text
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch != " " and random.random() < p:
            chars[i] = random.choice(HAUSA_CHARACTERS)
    return "".join(chars)


def _transpose_characters(text: str, p: float) -> str:
    if not p or not text or len(text) < 2:
        return text
    chars = list(text)
    i = 0
    while i < len(chars) - 1:
        if chars[i] != " " and chars[i + 1] != " " and random.random() < p:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            i += 2
        else:
            i += 1
    return "".join(chars)


def _delete_chunk(text: str, p: float, size: int = 2) -> str:
    if not p or not text or len(text) < size:
        return text
    out, idx = [], 0
    while idx < len(text):
        chunk = text[idx: idx + size]
        if len(chunk) == size and any(c != " " for c in chunk) and random.random() < p:
            idx += size
        else:
            out.append(text[idx])
            idx += 1
    return "".join(out)


def _insert_chunk(text: str, p: float, size: int = 2) -> str:
    if not p:
        return text
    if not text:
        return (
            "".join(random.choice(HAUSA_CHARACTERS) for _ in range(size))
            if random.random() < p else text
        )
    out = []
    for ch in text:
        out.append(ch)
        if ch != " " and random.random() < p:
            out.extend(random.choice(HAUSA_CHARACTERS) for _ in range(size))
    if random.random() < p / 20:
        out = [random.choice(HAUSA_CHARACTERS) for _ in range(size)] + out
    return "".join(out)


# Word-level differential noising

def _noise_word(word: str, noise: dict) -> str:
    w = _incorrect_characters(word, noise.get("incorrect_characters", 0))
    w = _substitute_characters(w,   noise.get("substitute_characters", 0))
    w = _delete_characters(w,       noise.get("delete_characters", 0))
    w = _delete_chunk(w,            noise.get("delete_chunk", 0))
    w = _duplicate_characters(w,    noise.get("duplicate_characters", 0))
    w = _insert_chunk(w,            noise.get("insert_chunk", 0))
    w = _transpose_characters(w,    noise.get("transpose_characters", 0))
    return w


def apply_noise(sentence: str, noise: dict, top_words: set,
                prob_freq: float, prob_other: float) -> str:
    """
    Apply differential character-level noise to *sentence*.
    Words in *top_words* (frequent natural words) are noised with *prob_freq*;
    all other words with *prob_other*. Spacing noise is applied sentence-wide.
    """
    if not sentence:
        return ""
    parts  = re.split(r"(\s+)", sentence)
    noised = []
    for part in parts:
        if not part.strip():
            noised.append(part)
            continue
        threshold = prob_freq if (top_words and part.lower() in top_words) else prob_other
        noised.append(_noise_word(part, noise) if random.random() < threshold else part)
    result = "".join(noised)
    result = _random_spacing(result, noise.get("random_spacing", 0))
    result = _remove_spaces(result,  noise.get("remove_spaces",  0))
    return re.sub(r"\s+", " ", result).strip()


# Analysis helpers

def load_top_words(path: str) -> set:
    if not path or not os.path.exists(path):
        print(f"  [warn] top-words file not found: {path}")
        return set()
    with open(path, "r", encoding="utf-8") as fh:
        return {line.strip().lower() for line in fh if line.strip()}


def extract_non_english_tokens(texts: list, en_vocab: set, min_len: int = 3) -> list:
    """Return tokens that are not in the English vocab (proxy for Hausa/noisy words)."""
    tokens = []
    for text in texts:
        for tok in re.findall(r"\b[a-z']+\b", str(text).lower()):
            if tok not in en_vocab and len(tok) >= min_len:
                tokens.append(tok)
    return tokens


def comparison_groups(word_list: list) -> list:
    """
    Group unique words into overlapping windows of length [L-1, L, L+1].
    Returns a list of word lists.
    """
    by_len = defaultdict(list)
    for w in set(word_list):
        by_len[len(w)].append(w)
    groups = []
    for length in sorted(by_len):
        group = list(by_len[length])
        if length - 1 in by_len:
            group.extend(by_len[length - 1])
        if length + 1 in by_len:
            group.extend(by_len[length + 1])
        unique = sorted(set(group))
        if len(unique) >= 2:
            groups.append(unique)
    return groups


def compute_distances_and_clusters(
    groups: list,
    max_per_group: int = 200,
    eps: float = 0.4,
    min_samples: int = 2,
) -> tuple:
    all_dists, cluster_sizes = [], []
    for group in tqdm(groups, desc="  Processing groups"):
        if len(group) < 2:
            continue
        sample = random.sample(group, max_per_group) if len(group) > max_per_group else group
        if len(sample) < 2:
            continue
        dm = np.array(
            [[textdistance.levenshtein.normalized_distance(a, b) for b in sample]
             for a in sample]
        )
        idx = np.triu_indices(len(sample), k=1)
        if idx[0].size:
            all_dists.extend(dm[idx].tolist())
        if len(sample) >= min_samples:
            try:
                labels = DBSCAN(
                    eps=eps, min_samples=min_samples, metric="precomputed"
                ).fit_predict(dm)
                for lbl, cnt in Counter(labels).items():
                    if lbl != -1:
                        cluster_sizes.append(cnt)
            except Exception as exc:
                print(f"  [warn] DBSCAN error: {exc}")
    return all_dists, cluster_sizes


# Plotting

def plot_distributions(nat_dists, syn_dists, nat_clusters, syn_clusters, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # — Distance distribution —
    fig, ax = plt.subplots(figsize=(10, 5))
    if nat_dists:
        sns.histplot(nat_dists, bins=30, kde=True, color="steelblue",
                     label="Twitter (natural)", stat="density", ax=ax)
    if syn_dists:
        sns.histplot(syn_dists, bins=30, kde=True, color="tomato",
                     label="Synthetic", stat="density", alpha=0.65, ax=ax)
    ax.set_title(
        "Normalised Levenshtein Distance Distribution\n"
        "(within length-bucketed comparison groups)"
    )
    ax.set_xlabel("Normalised Levenshtein Distance")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    p1 = os.path.join(out_dir, "lev_distance_distribution.png")
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"  Saved → {p1}")

    # Cluster size distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    if nat_clusters:
        sns.histplot(nat_clusters, color="steelblue", label="Twitter (natural)",
                     discrete=True, ax=ax)
    if syn_clusters:
        sns.histplot(syn_clusters, color="tomato", label="Synthetic",
                     discrete=True, alpha=0.65, ax=ax)
    ax.set_title("DBSCAN Cluster Size Distribution")
    ax.set_xlabel("Cluster Size")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    p2 = os.path.join(out_dir, "cluster_size_distribution.png")
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"  Saved → {p2}")


def print_stats(arr, label):
    if not arr:
        print(f"  {label}: no data")
        return
    a = np.array(arr)
    print(f"  {label}: n={len(a):,}  mean={a.mean():.4f}  "
          f"median={np.median(a):.4f}  std={a.std():.4f}")


# Main

def main():
    t0 = time.time()
    sns.set(style="whitegrid")

    # Download clean formal Hausa text from GDrive
    download_gdrive_folder(GDRIVE_FOLDER_URL, DOWNLOAD_DIR)

    # Load Twitter data, determine target size N
    print(f"\nLoading Twitter data from {TWITTER_TRAIN_PATH} …")
    twitter_df = pd.read_csv(TWITTER_TRAIN_PATH, sep="\t")
    col = TWEET_COL if TWEET_COL in twitter_df.columns else twitter_df.columns[0]
    twitter_texts = twitter_df[col].dropna().astype(str).tolist()
    N = len(twitter_texts)
    print(f"  {N:,} tweets loaded — this is the target sample size.")

    # Load all clean sentences from the downloaded corpus
    print(f"\nLoading clean sentences from {DOWNLOAD_DIR} …")
    clean_sentences = load_clean_sentences(DOWNLOAD_DIR, EXCLUDE_FILENAMES)
    total_clean = len(clean_sentences)
    print(f"  Total clean sentences available: {total_clean:,}")
    if total_clean == 0:
        raise RuntimeError("No clean sentences loaded. Check the download directory and file contents.")

    # Randomly sample ~N clean sentences
    sample_size = min(N, total_clean)
    print(f"\nRandomly sampling {sample_size:,} sentences from clean corpus …")
    sampled_clean = random.sample(clean_sentences, sample_size)

    # Apply synthetic noise from Table 1 params
    print("Loading top Hausa natural words …")
    top_words = load_top_words(TOP_WORDS_PATH)
    print(f"  {len(top_words)} words loaded.")

    print(f"Applying synthetic noise to {sample_size:,} sampled sentences …")
    synthetic_noisy = [
        apply_noise(
            sentence=s,
            noise=PAPER_NOISE_LEVELS,
            top_words=top_words,
            prob_freq=PROB_NOISE_FREQUENT_WORD,
            prob_other=PROB_NOISE_OTHER_WORD,
        )
        for s in tqdm(sampled_clean, desc="  Noising")
    ]

    # Load NLTK English vocab (used to filter for Hausa/noisy tokens)
    print("\nLoading NLTK English vocabulary …")
    try:
        nltk_words.words()
    except LookupError:
        nltk_download("words")
    en_vocab = {w.lower() for w in nltk_words.words()}

    # Extract non-English tokens from each corpus
    print("Extracting non-English tokens …")
    nat_tokens = extract_non_english_tokens(twitter_texts,   en_vocab)
    syn_tokens = extract_non_english_tokens(synthetic_noisy, en_vocab)
    print(f"  Twitter   : {len(nat_tokens):,} tokens  ({len(set(nat_tokens)):,} unique)")
    print(f"  Synthetic : {len(syn_tokens):,} tokens  ({len(set(syn_tokens)):,} unique)")

    # Build length-bucketed comparison groups
    print("\nBuilding comparison groups (word length ±1 windows) …")
    nat_groups = comparison_groups(nat_tokens)
    syn_groups = comparison_groups(syn_tokens)
    print(f"  Twitter   : {len(nat_groups)} groups")
    print(f"  Synthetic : {len(syn_groups)} groups")

    # Compute pairwise Levenshtein distances + DBSCAN clusters
    print("\nComputing distances + clusters — Twitter data …")
    nat_dists, nat_clusters = compute_distances_and_clusters(nat_groups)

    print("\nComputing distances + clusters — Synthetic data …")
    syn_dists, syn_clusters = compute_distances_and_clusters(syn_groups)

    # Plot
    print("\nPlotting …")
    plot_distributions(nat_dists, syn_dists, nat_clusters, syn_clusters, PLOT_OUTPUT_DIR)

    # Print summary statistics
    print("\nDistance distribution summary:")
    print_stats(nat_dists,    "Twitter   distances")
    print_stats(syn_dists,    "Synthetic distances")

    print("\nCluster size summary:")
    print_stats(nat_clusters, "Twitter   cluster sizes")
    print_stats(syn_clusters, "Synthetic cluster sizes")

    print(f"\nTop-20 most frequent natural  tokens : {Counter(nat_tokens).most_common(20)}")
    print(f"Top-20 most frequent synthetic tokens : {Counter(syn_tokens).most_common(20)}")

    print(f"\nFinished in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()