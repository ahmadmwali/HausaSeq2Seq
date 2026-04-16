#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import random
import time
import csv
import json
import shutil
import argparse
from tqdm import tqdm
import gdown  # Make sure to run `pip install gdown`

class TextProcessor:
    HAUSA_CHARACTERS = "abcdefghijklmnopqrstuvwxyz'ƙɗɓyABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    DEFAULT_NOISE_LEVELS = {
        'random_spacing': 0.01,
        'remove_spaces': 0.1,
        'incorrect_characters': 0.01,
        'delete_characters': 0.01,
        'duplicate_characters': 0.01,
        'substitute_characters': 0.01,
        'transpose_characters': 0.01,
        'delete_chunk': 0.005,
        'insert_chunk': 0.005
    }

    def __init__(self, input_files, train_output_path, val_output_path, test_output_path,
                     train_size, val_size, test_size,
                     include_religious=False,
                     religious_files=None,
                     noise_levels=None, 
                     top_natural_words_path=None, 
                     prob_noise_frequent_word=0.25, 
                     prob_noise_other_word=0.8,    
                     encoding='utf-8'):
        self.input_files = input_files
        self.train_output_path = train_output_path
        self.val_output_path = val_output_path
        self.test_output_path = test_output_path

        self.train_size = int(train_size) if train_size is not None and int(train_size) > 0 else 0
        self.val_size = int(val_size) if val_size is not None and int(val_size) > 0 else 0
        self.test_size = int(test_size) if test_size is not None and int(test_size) > 0 else 0

        self.include_religious = include_religious
        self.religious_files = religious_files if religious_files is not None else ["Bible.txt", "Tanzil.txt"]

        self.noise_levels = self.DEFAULT_NOISE_LEVELS.copy()
        if noise_levels:
            self.noise_levels.update(noise_levels)

        self.encoding = encoding

        # Load top natural words
        self.top_natural_words_set = self._load_top_natural_words(top_natural_words_path)
        self.prob_noise_frequent_word = prob_noise_frequent_word
        self.prob_noise_other_word = prob_noise_other_word
        print(f"Loaded {len(self.top_natural_words_set)} top natural words.")
        print(f"Probability to noise frequent words: {self.prob_noise_frequent_word}")
        print(f"Probability to noise other words: {self.prob_noise_other_word}")


        for key in self.DEFAULT_NOISE_LEVELS:
            if key not in self.noise_levels:
                print(f"Warning: Char noise level for '{key}' not specified. Using default: {self.DEFAULT_NOISE_LEVELS[key]}")
                self.noise_levels[key] = self.DEFAULT_NOISE_LEVELS[key]
            elif not (0 <= self.noise_levels.get(key, 0) <= 1):
                raise ValueError(f"Char noise probability for '{key}' must be between 0 and 1.")

        if noise_levels: 
            for key in noise_levels:
                if key not in self.DEFAULT_NOISE_LEVELS:
                    print(f"Warning: Custom char noise level '{key}' provided but not in default set. Will be used if corresponding method exists.")
                if not (0 <= self.noise_levels.get(key, 0) <= 1):
                    raise ValueError(f"Char noise probability for '{key}' must be between 0 and 1.")

        if not (0 <= self.prob_noise_frequent_word <= 1):
            raise ValueError("prob_noise_frequent_word must be between 0 and 1.")
        if not (0 <= self.prob_noise_other_word <= 1):
            raise ValueError("prob_noise_other_word must be between 0 and 1.")

        if self.train_size == 0 and self.val_size == 0 and self.test_size == 0:
            raise ValueError("At least one of train_size, val_size, or test_size must be a positive integer.")

    def _load_top_natural_words(self, path):
        if not path:
            print("Warning: No path provided for top natural words list. Differential noising for frequent words will not be active.")
            return set()
        try:
            with open(path, 'r', encoding=self.encoding) as f:
                return set(line.strip().lower() for line in f if line.strip())
        except FileNotFoundError:
            print(f"Warning: Top natural words file not found at {path}. Differential noising for frequent words will not be active.")
            return set()
        except Exception as e:
            print(f"Warning: Could not read top natural words file {path} due to {e}. Differential noising for frequent words will not be active.")
            return set()

    def _merge_files(self):
        print("Merging files...")
        merged_content = []
        files_to_process = self.input_files[:]

        if not self.include_religious:
            print("Excluding religious files...")
            files_to_process = [
                fpath for fpath in files_to_process
                if not any(fpath.endswith(os.path.sep + r_file) or os.path.basename(fpath) == r_file for r_file in self.religious_files)
            ]
            print(f"Files remaining after exclusion: {[os.path.basename(f) for f in files_to_process]}")

        for fname in files_to_process:
            try:
                with open(fname, "r", encoding=self.encoding) as infile:
                    print(f"Reading: {fname}")
                    merged_content.append(infile.read())
            except FileNotFoundError:
                print(f"Warning: File not found - {fname}. Skipping.")
            except Exception as e:
                print(f"Warning: Could not read file {fname} due to {e}. Skipping.")
        print("File merging complete.")
        return "\n".join(merged_content) if merged_content else ""

    def _clean_text(self, text):
        print("Cleaning text...")
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'NBSP', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        print("Text cleaning complete.")
        return text

    def _segment_sentences(self, text):
        print("Segmenting text into sentences...")
        sentence_pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!)\s+')
        sentences = []
        paragraphs = text.split('\n')
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph:
                para_sentences = sentence_pattern.split(paragraph)
                sentences.extend([s.strip() for s in para_sentences if s.strip()])
        print(f"Segmented into {len(sentences)} sentences.")
        sentences = [s for s in sentences if s]
        print(f"Keeping {len(sentences)} non-empty sentences after segmentation.")
        return sentences

    def _random_spacing(self, text, probability):
        if probability == 0 or not text: return text
        new_text = []
        for i, char in enumerate(text):
            if char.isalnum() and random.random() < probability:
                if i > 0 and text[i-1] != ' ':
                    new_text.append(' ')
            new_text.append(char)
        return re.sub(r' +', ' ', "".join(new_text)).strip()

    def _remove_spaces(self, text, probability):
        if probability == 0 or not text: return text
        new_text = list(text)
        for i in range(len(new_text) - 2, 0, -1):
            if new_text[i] == ' ':
                if new_text[i-1] != ' ' and new_text[i+1] != ' ':
                    if random.random() < probability:
                        new_text.pop(i)
        return "".join(new_text).strip()

    def _incorrect_characters(self, text, probability):
        if probability == 0 or not text: return text
        replacements = {'ɓ': 'b', 'ɗ': 'd', 'ƙ': 'k', 'ƴ': 'y',
                        'Ɓ': 'B', 'Ɗ': 'D', 'Ƙ': 'K', 'Ƴ': 'Y'}
        new_text_list = list(text)
        for i, char in enumerate(new_text_list):
            if char in replacements and random.random() < probability:
                new_text_list[i] = replacements[char]
        return "".join(new_text_list)

    def _delete_characters(self, text, probability):
        if probability == 0 or not text: return text
        return ''.join(char for char in text if char == ' ' or random.random() > probability)

    def _duplicate_characters(self, text, probability):
        if probability == 0 or not text: return text
        new_text = []
        for char in text:
            new_text.append(char)
            if char != ' ' and random.random() < probability:
                new_text.append(char)
        return ''.join(new_text)

    def _substitute_characters(self, text, probability):
        if probability == 0 or not text: return text
        new_text_list = list(text)
        for i, char in enumerate(new_text_list):
            if char != ' ' and random.random() < probability:
                new_char = random.choice(self.HAUSA_CHARACTERS)
                new_text_list[i] = new_char
        return "".join(new_text_list)

    def _transpose_characters(self, text, probability):
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

    def _delete_char_chunk(self, text, probability, chunk_size=2):
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

    def _insert_random_char_chunk(self, text, probability, chunk_size=2):
        if probability == 0: return text
        new_text_list = []
        if not text:
            if random.random() < probability:
                for _ in range(chunk_size):
                    new_text_list.append(random.choice(self.HAUSA_CHARACTERS))
            return "".join(new_text_list)
        
        for char in text:
            new_text_list.append(char)
            if char != ' ' and random.random() < probability:
                for _ in range(chunk_size):
                    new_text_list.append(random.choice(self.HAUSA_CHARACTERS))
                if random.random() < (probability / 20): 
                    chunk_to_insert = [random.choice(self.HAUSA_CHARACTERS) for _ in range(chunk_size)]
                    new_text_list = chunk_to_insert + new_text_list
        return "".join(new_text_list)

    def _apply_char_level_noise_to_word(self, word_text):
        if not word_text or not word_text.strip(): 
            return word_text
        noisy_w = word_text 
        noisy_w = self._incorrect_characters(noisy_w, self.noise_levels.get('incorrect_characters', 0))
        noisy_w = self._substitute_characters(noisy_w, self.noise_levels.get('substitute_characters', 0))
        noisy_w = self._duplicate_characters(noisy_w, self.noise_levels.get('duplicate_characters', 0))
        noisy_w = self._transpose_characters(noisy_w, self.noise_levels.get('transpose_characters', 0))
        noisy_w = self._delete_characters(noisy_w, self.noise_levels.get('delete_characters', 0))
        noisy_w = self._delete_char_chunk(noisy_w, self.noise_levels.get('delete_chunk', 0))
        noisy_w = self._insert_random_char_chunk(noisy_w, self.noise_levels.get('insert_chunk', 0))
        return noisy_w

    def _apply_noise(self, text):
        words = text.split()
        noisy_words = []
        for word in words:
            clean_word = re.sub(r'[^\w\s]', '', word).lower()
            if clean_word in self.top_natural_words_set:
                prob = self.prob_noise_frequent_word
            else:
                prob = self.prob_noise_other_word
            
            if random.random() < prob:
                word = self._apply_char_level_noise_to_word(word)
            noisy_words.append(word)

        noisy_text = " ".join(noisy_words)
        noisy_text = self._random_spacing(noisy_text, self.noise_levels['random_spacing'])
        noisy_text = self._remove_spaces(noisy_text, self.noise_levels['remove_spaces'])
        return noisy_text

    def _save_to_tsv(self, data, filepath):
        if not data or not filepath:
            return
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding=self.encoding, newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(["Clean", "Noisy"])
            for clean, noisy in data:
                writer.writerow([clean, noisy])
        print(f"Saved {len(data)} lines to {filepath}")

    def process(self):
        merged_text = self._merge_files()
        if not merged_text:
            print("No text to process.")
            return

        cleaned_text = self._clean_text(merged_text)
        clean_sentences = self._segment_sentences(cleaned_text)

        total_sentences = len(clean_sentences)
        print(f"Total clean sentences generated: {total_sentences}")

        if total_sentences == 0:
            print("No sentences found after cleaning and segmentation. Aborting.")
            return

        print("Shuffling sentences...")
        random.shuffle(clean_sentences)

        train_count = min(self.train_size, total_sentences)
        remaining = total_sentences - train_count
        val_count = min(self.val_size, remaining)
        remaining -= val_count
        test_count = min(self.test_size, remaining)

        print(f"Splitting into: Train={train_count}, Validation={val_count}, Test={test_count}")
        train_sentences = clean_sentences[:train_count]
        val_sentences = clean_sentences[train_count : train_count + val_count]
        test_sentences = clean_sentences[train_count + val_count : train_count + val_count + test_count]

        splits_to_process = []
        if train_count > 0: splits_to_process.append(('Train', self.train_output_path, train_sentences))
        if val_count > 0: splits_to_process.append(('Validation', self.val_output_path, val_sentences))
        if test_count > 0: splits_to_process.append(('Test', self.test_output_path, test_sentences))

        for name, filepath, subset in splits_to_process:
            if filepath and subset:
                print(f"Generating noise for {name} set...")
                noisy_data = []
                for clean_sentence in tqdm(subset):
                    noisy_sentence = self._apply_noise(clean_sentence)
                    noisy_data.append((clean_sentence, noisy_sentence))
                self._save_to_tsv(noisy_data, filepath)

def parse_args():
    parser = argparse.ArgumentParser(description="Download and generate noisy Hausa seq2seq datasets.")
    
    # Paths and Directories
    parser.add_argument("--output_dir", type=str, default="./Data", help="Directory to save final TSV files and supplementary data.")
    parser.add_argument("--temp_dir", type=str, default="./Original_Texts_Temp", help="Temporary directory for downloaded raw texts (deleted after).")
    
    # Dataset Sizes
    parser.add_argument("--train_size", type=int, default=10000, help="Number of sentences for the training set.")
    parser.add_argument("--val_size", type=int, default=100, help="Number of sentences for the validation set.")
    parser.add_argument("--test_size", type=int, default=100, help="Number of sentences for the test set.")
    
    # Noise Probabilities
    parser.add_argument("--prob_frequent_noised", type=float, default=0.02, help="Probability to apply noise to top frequent words.")
    parser.add_argument("--prob_other_noised", type=float, default=0.4, help="Probability to apply noise to all other words.")
    
    # General Options
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--skip_downloads", action="store_true", help="Skip downloading files if you already have them locally.")

    return parser.parse_args()


# Main Data Generation Execution
if __name__ == "__main__":
    args = parse_args()
    
    # Apply the global random seed for full reproducibility
    random.seed(args.seed)
    
    print("\n--- Starting Data Processing ---")
    
    # Ensure required directories exist
    original_texts_dir = args.temp_dir
    output_dir = args.output_dir
    os.makedirs(original_texts_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Downloads.
    if not args.skip_downloads:
        # Download the ORIGINAL texts folder
        url = "https://drive.google.com/drive/folders/1o2Yg0PiU3nI7z24TaVkzu320g65JZfpO"
        print(f"Downloading Original Texts from {url} to {original_texts_dir} ...")
        gdown.download_folder(url, output=original_texts_dir, quiet=False, use_cookies=False)

        # Download the SUPPLEMENTARY folder into Data/supplementary
        supp_url = "https://drive.google.com/drive/folders/1ilf1zNuTiE7kTLbFuuvbKGNcuDQxH0la"
        supp_dir = os.path.join(output_dir, "supplementary")
        print(f"Downloading Supplementary files from {supp_url} to {supp_dir} ...")
        gdown.download_folder(supp_url, output=supp_dir, quiet=False, use_cookies=False)
    else:
        print("Skipping downloads as requested.")

    input_files_list = [
        os.path.join(original_texts_dir, "Other_Texts.txt"),
        os.path.join(original_texts_dir, "Wikimedia.txt"),
        os.path.join(original_texts_dir, "Wikipedia.txt"),
        os.path.join(original_texts_dir, "Bible.txt"),     # Can be excluded
        os.path.join(original_texts_dir, "Tanzil.txt")     # Can be excluded
    ]

    # Create a txt of top 100 natural words file if it doesn't exist in directory
    top_100_words_path = os.path.join(output_dir, "top_100_hausa_natural.txt")
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
    
    with open(top_100_words_path, "w", encoding="utf-8") as f:
        for word in dummy_top_words:
            f.write(word + "\n")
    print(f"Created dummy top natural words file: {top_100_words_path}")

    # Define output paths based on parsed args
    train_path = os.path.join(output_dir, "train.tsv")
    val_path = os.path.join(output_dir, "validation.tsv")
    test_path = os.path.join(output_dir, "test.tsv")

    char_level_noise_config = {
        'random_spacing': 0.02,
        'remove_spaces': 0.15,
        'incorrect_characters': 0.02,  
        'delete_characters': 0.005,    
        'duplicate_characters': 0.01,  
        'substitute_characters': 0.001,
        'transpose_characters': 0.01, 
        'delete_chunk': 0.0015,         
        'insert_chunk': 0.001,          
    }
    
    # Initialize and run the Text Processor
    processor = TextProcessor(
        input_files=input_files_list,
        train_output_path=train_path,
        val_output_path=val_path,
        test_output_path=test_path,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        include_religious=False,
        noise_levels=char_level_noise_config, 
        top_natural_words_path=top_100_words_path, 
        prob_noise_frequent_word=args.prob_frequent_noised,
        prob_noise_other_word=args.prob_other_noised
    )
    
    processor.process()
    
    # Delete the temporary folder
    print(f"Cleaning up: Deleting temporary folder {original_texts_dir} ...")
    if os.path.exists(original_texts_dir):
        shutil.rmtree(original_texts_dir)
        print("Cleanup complete.")
    
    print(f"\nDifferential noisy data generation complete. Check files in {output_dir}")