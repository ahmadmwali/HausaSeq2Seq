# -*- coding: utf-8 -*-

import os
import re
import random
import time
import csv
import argparse
import json
from tqdm import tqdm

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
                     noise_levels=None, # For character-level noise functions
                     top_natural_words_path=None, # Path to file with top natural words
                     prob_noise_frequent_word=0.25, # Prob to noise a word if it's in top_natural_words_set
                     prob_noise_other_word=0.8,    # Prob to noise a word if it's NOT in top_natural_words_set
                     encoding='utf-8'):
        self.input_files = input_files
        self.train_output_path = train_output_path
        self.val_output_path = val_output_path
        self.test_output_path = test_output_path

        self.train_size = int(train_size) if train_size is not None and int(train_size) > 0 else 0
        self.val_size = int(val_size) if val_size is not None and int(val_size) > 0 else 0
        self.test_size = int(test_size) if test_size is not None and int(test_size) > 0 else 0

        self.include_religious = include_religious
        self.religious_files = religious_files if religious_files is not None else ["Files/Bible.txt", "Files/Tanzil.txt"]

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

        if noise_levels: # Validate any extra keys provided in noise_levels
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
                # Read words, strip whitespace, convert to lowercase, and filter empty lines
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

    # --- Character Level Noise Functions (Mostly Unchanged Internally) ---
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
        if probability == 0 or not text or len(text) < chunk_size:
            return text

        result_chars = []
        idx = 0
        text_len = len(text)
        while idx < text_len:
            current_chunk = text[idx : idx + chunk_size]
            if len(current_chunk) == chunk_size and \
               any(c != ' ' for c in current_chunk) and \
               random.random() < probability:
                idx += chunk_size
            else:
                result_chars.append(text[idx])
                idx += 1
        return "".join(result_chars)

    def _insert_random_char_chunk(self, text, probability, chunk_size=2):
        if probability == 0 : return text # Allow empty text for insertion start

        new_text_list = []
        if not text: # Special case for empty input text
            if random.random() < probability:
                    for _ in range(chunk_size):
                        new_text_list.append(random.choice(self.HAUSA_CHARACTERS))
            return "".join(new_text_list)

        # Insert after characters
        for char_idx, char_val in enumerate(text):
            new_text_list.append(char_val)
            if char_val != ' ' and random.random() < probability: # Insert after non-space
                for _ in range(chunk_size):
                    new_text_list.append(random.choice(self.HAUSA_CHARACTERS))

        # Small chance to insert at the very beginning for non-empty text
        if random.random() < (probability / 20): # Reduced chance for prepend on non-empty
            chunk_to_insert = [random.choice(self.HAUSA_CHARACTERS) for _ in range(chunk_size)]
            new_text_list = chunk_to_insert + new_text_list

        return "".join(new_text_list)

    # Helper for Applying Character-Level Noise to a Single Word
    def _apply_char_level_noise_to_word(self, word_text):
        """Applies all configured character-level noise types to a single word."""
        if not word_text or not word_text.strip(): # Avoid noising empty or whitespace-only words
            return word_text

        noisy_w = word_text
        # Apply character-altering noises. Spacing noises are not applied at word level.
        noisy_w = self._incorrect_characters(noisy_w, self.noise_levels.get('incorrect_characters', 0))
        noisy_w = self._substitute_characters(noisy_w, self.noise_levels.get('substitute_characters', 0))
        noisy_w = self._delete_characters(noisy_w, self.noise_levels.get('delete_characters', 0))
        noisy_w = self._delete_char_chunk(noisy_w, self.noise_levels.get('delete_chunk', 0))
        noisy_w = self._duplicate_characters(noisy_w, self.noise_levels.get('duplicate_characters', 0))
        noisy_w = self._insert_random_char_chunk(noisy_w, self.noise_levels.get('insert_chunk', 0)) # Applied to word
        noisy_w = self._transpose_characters(noisy_w, self.noise_levels.get('transpose_characters', 0))
        return noisy_w

    # --- Main Noise Application Method ---
    def _apply_differential_noise(self, sentence_text):
        """
        Applies noise differentially:
        - Words in top_natural_words_set get noised with prob_noise_frequent_word.
        - Other words get noised with prob_noise_other_word.
        - Sentence-level spacing noises are applied at the end.
        """
        if not sentence_text: return ""

        # Simple space-based tokenization. For more robustness, consider a proper tokenizer.
        # This preserves original spacing between words to some extent.
        words = re.split(r'(\s+)', sentence_text) # Split by space, keeping spaces

        noised_parts = []
        for part in words:
            if not part.strip(): # If it's only whitespace
                noised_parts.append(part) # Keep the spacing
                continue

            # Determine if this word part should be noised at the character level
            word_lower = part.lower()
            should_noise_this_word = False
            if self.top_natural_words_set and word_lower in self.top_natural_words_set:
                if random.random() < self.prob_noise_frequent_word:
                    should_noise_this_word = True
            else: # Word not in top list, or top list is empty
                if random.random() < self.prob_noise_other_word:
                    should_noise_this_word = True

            if should_noise_this_word:
                noised_parts.append(self._apply_char_level_noise_to_word(part))
            else:
                noised_parts.append(part) # Keep original word part

        reconstructed_sentence = "".join(noised_parts)

        # Apply sentence-level spacing noises to the reconstructed sentence
        final_noisy_sentence = self._random_spacing(reconstructed_sentence, self.noise_levels.get('random_spacing', 0))
        final_noisy_sentence = self._remove_spaces(final_noisy_sentence, self.noise_levels.get('remove_spaces', 0))

        return re.sub(r'\s+', ' ', final_noisy_sentence).strip()


    def process_and_save(self):
        start_time = time.time()
        print("Starting text processing pipeline...")
        merged_text = self._merge_files()
        if not merged_text:
            print("Error: No text obtained after merging files. Aborting.")
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

        for split_name, output_path, sentences in splits_to_process:
            print(f"\nProcessing and writing {split_name} set to {output_path}...")
            rows_written = 0
            skipped_empty_noisy = 0
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created directory for {split_name} output: {output_dir}")

            try:
                with open(output_path, 'w', encoding=self.encoding, newline='') as outfile:
                    writer = csv.writer(outfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(['Noisy', 'Clean'])
                    for clean_sentence in tqdm(sentences, desc=f"Generating {split_name} data"):
                        # noisy_sentence = self._apply_noise(clean_sentence)
                        noisy_sentence = self._apply_differential_noise(clean_sentence)

                        if noisy_sentence:
                            writer.writerow([noisy_sentence, clean_sentence])
                            rows_written += 1
                        else:
                            skipped_empty_noisy += 1
                print(f"Done writing {split_name} set. {rows_written} rows written.")
                if skipped_empty_noisy > 0:
                    print(f"Warning for {split_name} set: Skipped {skipped_empty_noisy} rows because the noisy version became empty.")
            except Exception as e:
                print(f"Error writing {split_name} TSV file: {e}")

        end_time = time.time()
        duration = end_time - start_time
        print(f"\nProcessing pipeline finished in {duration:.2f} seconds.")

def parse_args():
    parser = argparse.ArgumentParser(description='Process Hausa text with differential noise for training data generation.')
    
    # Required arguments
    parser.add_argument('--input_files', nargs='+', required=True,
                        help='List of input text files to process')
    parser.add_argument('--train_output', required=True,
                        help='Output path for training TSV file')
    parser.add_argument('--val_output', required=True,
                        help='Output path for validation TSV file')
    parser.add_argument('--test_output', required=True,
                        help='Output path for test TSV file')
    
    # Dataset size arguments
    parser.add_argument('--train_size', type=int, default=440000,
                        help='Number of training samples (default: 440000)')
    parser.add_argument('--val_size', type=int, default=5000,
                        help='Number of validation samples (default: 5000)')
    parser.add_argument('--test_size', type=int, default=5000,
                        help='Number of test samples (default: 5000)')
    
    # Religious text handling
    parser.add_argument('--include_religious', action='store_true',
                        help='Include religious texts in processing')
    parser.add_argument('--religious_files', nargs='+',
                        help='List of religious file names to exclude (if --include_religious is not set)')
    
    # Top natural words configuration
    parser.add_argument('--top_natural_words_path', type=str,
                        help='Path to file containing top natural/frequent words')
    parser.add_argument('--prob_noise_frequent_word', type=float, default=0.02,
                        help='Probability to noise frequent words (default: 0.02)')
    parser.add_argument('--prob_noise_other_word', type=float, default=0.4,
                        help='Probability to noise other words (default: 0.4)')
    
    # Character-level noise probabilities - can be passed as JSON string
    parser.add_argument('--noise_config', type=str,
                        help='JSON string or path to JSON file with noise configuration')
    
    # Encoding
    parser.add_argument('--encoding', type=str, default='utf-8',
                        help='Text encoding (default: utf-8)')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    print("Setting up text processing with provided arguments...")
    
    # Build noise configuration from arguments
    char_level_noise_config = None
    if args.noise_config:
        try:
            # Try to load as JSON file first
            if os.path.isfile(args.noise_config):
                with open(args.noise_config, 'r') as f:
                    char_level_noise_config = json.load(f)
                print(f"Loaded noise configuration from file: {args.noise_config}")
            else:
                # Try to parse as JSON string
                char_level_noise_config = json.loads(args.noise_config)
                print("Loaded noise configuration from JSON string")
        except json.JSONDecodeError as e:
            print(f"Error parsing noise configuration JSON: {e}")
            print("Using default noise levels")
        except Exception as e:
            print(f"Error loading noise configuration: {e}")
            print("Using default noise levels")
    
    if char_level_noise_config:
        print(f"Using char_level_noise_config: {char_level_noise_config}")
    else:
        print("Using default noise configuration")
    
    processor = TextProcessor(
        input_files=args.input_files,
        train_output_path=args.train_output,
        val_output_path=args.val_output,
        test_output_path=args.test_output,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        include_religious=args.include_religious,
        religious_files=args.religious_files,
        noise_levels=char_level_noise_config,
        top_natural_words_path=args.top_natural_words_path,
        prob_noise_frequent_word=args.prob_noise_frequent_word,
        prob_noise_other_word=args.prob_noise_other_word,
        encoding=args.encoding
    )
    
    processor.process_and_save()
    
    print(f"\n--- TextProcessor run complete. ---")
    print(f"Train: {args.train_output}")
    print(f"Validation: {args.val_output}")
    if args.test_size > 0:
        print(f"Test: {args.test_output}")

# Example Usage:
# 
# Basic usage with default noise configuration:
# python text_processor.py \
#     --input_files /path/to/file1.txt /path/to/file2.txt /path/to/file3.txt \
#     --train_output ./output/train.tsv \
#     --val_output ./output/validation.tsv \
#     --test_output ./output/test.tsv \
#     --train_size 440000 \
#     --val_size 5000 \
#     --test_size 5000 \
#     --top_natural_words_path ./top_100_hausa_natural.txt \
#     --prob_noise_frequent_word 0.02 \
#     --prob_noise_other_word 0.4
#
# Usage with custom noise configuration (JSON string):
# python text_processor.py \
#     --input_files /path/to/file1.txt /path/to/file2.txt \
#     --train_output ./output/train.tsv \
#     --val_output ./output/validation.tsv \
#     --test_output ./output/test.tsv \
#     --train_size 100000 \
#     --val_size 5000 \
#     --test_size 5000 \
#     --noise_config '{"random_spacing": 0.02, "remove_spaces": 0.15, "incorrect_characters": 0.02, "delete_characters": 0.005, "duplicate_characters": 0.01, "substitute_characters": 0.001, "transpose_characters": 0.01, "delete_chunk": 0.0015, "insert_chunk": 0.001}'
#
# Usage with noise configuration from JSON file:
# First create a JSON file (e.g., noise_config.json):
# {
#     "random_spacing": 0.02,
#     "remove_spaces": 0.15,
#     "incorrect_characters": 0.02,
#     "delete_characters": 0.005,
#     "duplicate_characters": 0.01,
#     "substitute_characters": 0.001,
#     "transpose_characters": 0.01,
#     "delete_chunk": 0.0015,
#     "insert_chunk": 0.001
# }
#
# Then run:
# python text_processor.py \
#     --input_files ./texts/file1.txt ./texts/file2.txt ./texts/file3.txt \
#     --train_output ./output/train.tsv \
#     --val_output ./output/validation.tsv \
#     --test_output ./output/test.tsv \
#     --noise_config ./noise_config.json \
#     --top_natural_words_path ./top_100_hausa_natural.txt
#
# Usage excluding religious texts:
# python text_processor.py \
#     --input_files ./texts/*.txt \
#     --train_output ./output/train.tsv \
#     --val_output ./output/validation.tsv \
#     --test_output ./output/test.tsv \
#     --religious_files Bible.txt Tanzil.txt Quran.txt \
#     --top_natural_words_path ./top_100_hausa_natural.txt
#
# Usage including religious texts:
# python text_processor.py \
#     --input_files ./texts/*.txt \
#     --train_output ./output/train.tsv \
#     --val_output ./output/validation.tsv \
#     --test_output ./output/test.tsv \
#     --include_religious \
#     --top_natural_words_path ./top_100_hausa_natural.txt
