#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import random
import argparse
import pandas as pd
import torch
from tqdm import tqdm
import sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# EXACT NOISE FUNCTIONS FROM get_data.py
HAUSA_CHARACTERS = "abcdefghijklmnopqrstuvwxyz'ƙɗɓyABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

def _random_spacing(text, probability):
    if probability == 0 or not text: return text
    new_text = []
    for i, char in enumerate(text):
        if char.isalnum() and random.random() < probability:
            if i > 0 and text[i-1] != ' ':
                new_text.append(' ')
        new_text.append(char)
    return re.sub(r' +', ' ', "".join(new_text)).strip()

def _remove_spaces(text, probability):
    if probability == 0 or not text: return text
    new_text = list(text)
    for i in range(len(new_text) - 2, 0, -1):
        if new_text[i] == ' ':
            if new_text[i-1] != ' ' and new_text[i+1] != ' ':
                if random.random() < probability:
                    new_text.pop(i)
    return "".join(new_text).strip()

def _incorrect_characters(text, probability):
    if probability == 0 or not text: return text
    replacements = {'ɓ': 'b', 'ɗ': 'd', 'ƙ': 'k', 'ƴ': 'y',
                    'Ɓ': 'B', 'Ɗ': 'D', 'Ƙ': 'K', 'Ƴ': 'Y'}
    new_text_list = list(text)
    for i, char in enumerate(new_text_list):
        if char in replacements and random.random() < probability:
            new_text_list[i] = replacements[char]
    return "".join(new_text_list)

def _delete_characters(text, probability):
    if probability == 0 or not text: return text
    return ''.join(char for char in text if char == ' ' or random.random() > probability)

def _duplicate_characters(text, probability):
    if probability == 0 or not text: return text
    new_text = []
    for char in text:
        new_text.append(char)
        if char != ' ' and random.random() < probability:
            new_text.append(char)
    return ''.join(new_text)

def _substitute_characters(text, probability):
    if probability == 0 or not text: return text
    new_text_list = list(text)
    for i, char in enumerate(new_text_list):
        if char != ' ' and random.random() < probability:
            new_char = random.choice(HAUSA_CHARACTERS)
            new_text_list[i] = new_char
    return "".join(new_text_list)

def _transpose_characters(text, probability):
    if probability == 0 or not text or len(text) < 2: return text
    new_text_list = list(text)
    i = 0
    while i < len(new_text_list) - 1:
        if new_text_list[i] != ' ' and new_text_list[i+1] != ' ' and random.random() < probability:
            new_text_list[i], new_text_list[i+1] = new_text_list[i+1], new_text_list[i]
            i += 2
        else:
            i += 1
    return "".join(new_text_list)

def _delete_char_chunk(text, probability, chunk_size=2):
    if probability == 0 or not text or len(text) < chunk_size: return text
    result_chars = []
    idx = 0
    text_len = len(text)
    while idx < text_len:
        current_chunk = text[idx : idx + chunk_size]
        if len(current_chunk) == chunk_size and any(c != ' ' for c in current_chunk) and random.random() < probability:
            idx += chunk_size
        else:
            result_chars.append(text[idx])
            idx += 1
    return "".join(result_chars)

def _insert_random_char_chunk(text, probability, chunk_size=2):
    if probability == 0: return text
    new_text_list = []
    if not text:
        if random.random() < probability:
            for _ in range(chunk_size):
                new_text_list.append(random.choice(HAUSA_CHARACTERS))
        return "".join(new_text_list)
    
    for char in text:
        new_text_list.append(char)
        if char != ' ' and random.random() < probability:
            for _ in range(chunk_size):
                new_text_list.append(random.choice(HAUSA_CHARACTERS))
            if random.random() < (probability / 20): 
                chunk_to_insert = [random.choice(HAUSA_CHARACTERS) for _ in range(chunk_size)]
                new_text_list = chunk_to_insert + new_text_list
    return "".join(new_text_list)

def apply_exact_noise(text, top_words_set, noise_levels, prob_freq, prob_other):
    words = text.split()
    noisy_words = []
    for word in words:
        clean_word = re.sub(r'[^\w\s]', '', word).lower()
        prob = prob_freq if clean_word in top_words_set else prob_other
        
        if random.random() < prob:
            # Apply char level functions
            word = _incorrect_characters(word, noise_levels.get('incorrect_characters', 0))
            word = _substitute_characters(word, noise_levels.get('substitute_characters', 0))
            word = _duplicate_characters(word, noise_levels.get('duplicate_characters', 0))
            word = _transpose_characters(word, noise_levels.get('transpose_characters', 0))
            word = _delete_characters(word, noise_levels.get('delete_characters', 0))
            word = _delete_char_chunk(word, noise_levels.get('delete_chunk', 0))
            word = _insert_random_char_chunk(word, noise_levels.get('insert_chunk', 0))
        noisy_words.append(word)

    noisy_text = " ".join(noisy_words)
    noisy_text = _random_spacing(noisy_text, noise_levels['random_spacing'])
    noisy_text = _remove_spaces(noisy_text, noise_levels['remove_spaces'])
    return noisy_text


# PIPELINE FUNCTIONS
def prep_flores(hau_parquet, eng_parquet, output_tsv, top_words_path):
    print(f"Reading Hausa parquet: {hau_parquet}")
    df_hau = pd.read_parquet(hau_parquet)
    print(f"Reading English parquet: {eng_parquet}")
    df_eng = pd.read_parquet(eng_parquet)

    # Reading from the 'text' column based on your parquet schema
    hausa_sentences = df_hau['text'].tolist()
    english_sentences = df_eng['text'].tolist()

    # Load top 100 words with fallback to dummy array
    top_words_set = set()
    if os.path.exists(top_words_path):
        with open(top_words_path, 'r', encoding='utf-8') as f:
            top_words_set = set(line.strip().lower() for line in f if line.strip())
        print(f"Loaded {len(top_words_set)} top natural words from {top_words_path} for differential noising.")
    else:
        dummy_top_words = [
            "wannan", "kuma", "dai", "mai", "nigeria", "ina", "wani", "kenan", 
            "gaskiya", "kawai", "duk", "wai", "muna", "yasa", "haka", "idan", 
            "wlh", "zai", "yake", "rahama", "har", "daga", "wanda", "daya", 
            "buhari", "toh", "iya", "cikin", "kara", "kawo", "irin", "yanzu", 
            "akwai", "bbc", "kowa", "wata", "kyau", "duniya", "wallahi", "yau", 
            "banza", "baki", "gaba", "abin", "kano", "abun", "saboda", "lafiya", 
            "suna", "kamar", "domin", "fara", "sarki", "kuwa", "mutane", "abinda", 
            "hakan", "kashe", "samu", "ruwa", "kasar", "kudi", "gani", "kudin", 
            "bai", "ciki", "ubangiji", "karya", "cewa", "hankali", "shiga", "yadda", 
            "zuwa", "arewa", "mutum", "ayi", "bayan", "aikin", "rashin", "kare", 
            "aiki", "magana", "gida", "zama", "kuka", "koma", "kam"
        ]
        top_words_set = set(dummy_top_words)
        print(f"Warning: File not found at {top_words_path}. Using {len(top_words_set)} fallback dummy words instead.")

    # Exact config from get_data.py
    char_level_noise_config = {
        'random_spacing': 0.02, 'remove_spaces': 0.15, 'incorrect_characters': 0.02,
        'delete_characters': 0.005, 'duplicate_characters': 0.01, 'substitute_characters': 0.001,
        'transpose_characters': 0.01, 'delete_chunk': 0.0015, 'insert_chunk': 0.001
    }
    prob_frequent_noised = 0.02
    prob_other_noised = 0.4

    noisy_sentences = []
    print("Injecting EXACT synthetic noise distribution into Hausa sentences...")
    random.seed(42) # Ensure reproducibility
    
    for text in tqdm(hausa_sentences):
        noisy_text = apply_exact_noise(
            text, top_words_set, char_level_noise_config, 
            prob_frequent_noised, prob_other_noised
        )
        noisy_sentences.append(noisy_text)

    df_out = pd.DataFrame({
        "Original_Hausa": hausa_sentences,
        "Noisy_Hausa": noisy_sentences,
        "English_Ref": english_sentences
    })

    os.makedirs(os.path.dirname(output_tsv), exist_ok=True)
    df_out.to_csv(output_tsv, sep='\t', index=False)
    print(f"Prepared FLORES dataset saved to {output_tsv}")
    print("NEXT STEP: Run predict_on_dataframe.py using this TSV to generate 'Cleaned_Hausa'!")

def eval_mt(input_tsv, output_tsv, batch_size):
    print(f"Loading dataset from {input_tsv}...")
    df = pd.read_csv(input_tsv, sep='\t')

    required_cols = ["Original_Hausa", "Noisy_Hausa", "Cleaned_Hausa", "English_Ref"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}. Did you run the prediction script yet?")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading NLLB-200 Translator on {device}...")

    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="hau_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    eng_id = tokenizer.convert_tokens_to_ids("eng_Latn")

    def translate_list(texts):
        translations = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, forced_bos_token_id=eng_id, max_length=256)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translations.extend(decoded)
        return translations

    print("\n[1/3] Translating ORIGINAL Hausa -> English...")
    df["Translated_Original"] = translate_list(df["Original_Hausa"].tolist())

    print("\n[2/3] Translating NOISY Hausa -> English...")
    df["Translated_Noisy"] = translate_list(df["Noisy_Hausa"].tolist())

    print("\n[3/3] Translating CLEANED Hausa -> English...")
    df["Translated_Cleaned"] = translate_list(df["Cleaned_Hausa"].tolist())

    df.to_csv(output_tsv, sep='\t', index=False)
    print(f"Translations saved to {output_tsv}")

    # Calculate Metrics
    print("\nCalculating SacreBLEU and chrF++ scores...")
    refs = [df["English_Ref"].tolist()] 

    metrics = []
    conditions = [
        ("Original (Gold)", "Translated_Original"), 
        ("Noisy (Raw)", "Translated_Noisy"), 
        ("Cleaned (Our Model)", "Translated_Cleaned")
    ]
    
    for condition_name, col_name in conditions:
        preds = df[col_name].tolist()
        bleu = sacrebleu.corpus_bleu(preds, refs).score
        chrf = sacrebleu.corpus_chrf(preds, refs).score
        metrics.append({"Data Variant": condition_name, "BLEU": round(bleu, 2), "chrF++": round(chrf, 2)})

    metrics_df = pd.DataFrame(metrics)
    print("\n==================================================")
    print("  FLORES+ Machine Translation Evaluation Results")
    print("==================================================")
    print(metrics_df.to_string(index=False))
    print("==================================================\n")

def main():
    parser = argparse.ArgumentParser(description="FLORES+ preparation and MT evaluation script.")
    parser.add_argument("--mode", choices=["prep", "eval"], required=True, help="Use 'prep' to extract parquets/add noise. Use 'eval' to translate/score.")
    parser.add_argument("--hau_parquet", type=str, default="./data/supplementary/FLORES+/hau_Latn.parquet")
    parser.add_argument("--eng_parquet", type=str, default="./data/supplementary/FLORES+/eng_Latn.parquet")
    parser.add_argument("--tsv_file", type=str, default="./data/supplementary/FLORES+/flores_pipeline.tsv", help="The working TSV file.")
    parser.add_argument("--out_file", type=str, default="./data/supplementary/FLORES+/flores_final_results.tsv", help="Where to save the translations.")
    parser.add_argument("--top_words", type=str, default="./data/top_100_hausa_natural.txt", help="Path to top 100 words file")
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()

    if args.mode == "prep":
        prep_flores(args.hau_parquet, args.eng_parquet, args.tsv_file, args.top_words)
    elif args.mode == "eval":
        eval_mt(args.tsv_file, args.out_file, args.batch_size)

if __name__ == "__main__":
    main()