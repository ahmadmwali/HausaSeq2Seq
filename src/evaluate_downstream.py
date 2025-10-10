import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report as sk_classification_report 
import numpy as np
import argparse
import os
import torch


from datasets import Dataset
import evaluate
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding, 
    set_seed
)

try:
    metric_accuracy = evaluate.load("accuracy")
    metric_precision = evaluate.load("precision")
    metric_recall = evaluate.load("recall")
    metric_f1 = evaluate.load("f1")
    print("Successfully loaded metrics using 'evaluate' library.")
except Exception as e:
    print(f"Error loading metrics using 'evaluate': {e}")
    print("Please ensure 'evaluate' library is installed: pip install evaluate")
    metric_accuracy = None # Set to None or handle otherwise

def compute_metrics(eval_pred):
    """Computes metrics for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = metric_accuracy.compute(predictions=predictions, references=labels)
    precision = metric_precision.compute(predictions=predictions, references=labels, average="weighted")
    recall = metric_recall.compute(predictions=predictions, references=labels, average="weighted")
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="weighted")

    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"],
    }

# Main Training and Evaluation Function
def train_and_evaluate_transformer(
    df,
    text_column,
    label_column,
    model_name,
    model_checkpoint,
    output_dir_base="./results",
    num_epochs=3, 
    batch_size=16, 
    learning_rate=2e-5, 
    seed=42
    ):
    """
    Trains a Transformer sentiment analysis model and evaluates its performance.

    Args:
        df (pd.DataFrame): DataFrame containing the text and labels.
        text_column (str): Name of the column containing the text data.
        label_column (str): Name of the column containing the labels.
        model_name (str): A name for the model run (e.g., "Raw Tweets", "Cleaned Tweets").
        model_checkpoint (str): Path or name of the pre-trained model from Hugging Face Hub.
        output_dir_base (str): Base directory to save training outputs.
        num_epochs (int): Number of training epochs.
        batch_size (int): Training and evaluation batch size.
        learning_rate (float): Learning rate for the optimizer.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary containing the model's F1 score, accuracy,
              precision, recall, and the full classification report string.
    """
    print(f"\n--- Training and Evaluating Transformer Model: {model_name} ---")
    print(f"Using text column: '{text_column}', label column: '{label_column}'")
    print(f"Using model checkpoint: '{model_checkpoint}'")

    set_seed(seed) 

    # Data Preprocessing 
    # Handle potential NaN values
    df_clean = df.dropna(subset=[text_column, label_column])
    if df_clean.empty:
        print(f"DataFrame is empty after dropping NaNs for columns '{text_column}' and '{label_column}'. Skipping model.")
        return {
            "model_name": model_name, "accuracy": 0, "precision": 0, "recall": 0,
            "f1_score": 0, "classification_report": "No data to train or evaluate."
        }

    # Ensure text data is string
    df_clean[text_column] = df_clean[text_column].astype(str)
    df_clean[label_column] = df_clean[label_column].astype(str)

    # Encode labels
    label_encoder = LabelEncoder()
    df_clean['label_encoded'] = label_encoder.fit_transform(df_clean[label_column])
    num_labels = len(label_encoder.classes_)
    class_names = label_encoder.classes_
    # Create mappings for model config
    id2label = {i: label for i, label in enumerate(class_names)}
    label2id = {label: i for i, label in id2label.items()}

    print(f"Found {num_labels} unique labels: {class_names}")
    print(f"Label mapping: {label2id}")

    # Check if there's enough data for splitting and stratification
    if len(df_clean) < 2 or num_labels < 1:
         print(f"Not enough data or classes ({len(df_clean)} rows, {num_labels} unique labels) to proceed. Skipping model.")
         return {
             "model_name": model_name, "accuracy": 0, "precision": 0, "recall": 0,
             "f1_score": 0, "classification_report": "Not enough data/classes for splitting."
         }

    # Split Data 
    test_size = 0.2
    train_df, eval_df = train_test_split(
        df_clean,
        test_size=test_size,
        random_state=seed,
        stratify=df_clean['label_encoded'] if num_labels > 1 else None # Stratify if possible
    )
    if num_labels <= 1:
        print("Warning: Only one class present. Cannot stratify.")

    print(f"Training set size: {len(train_df)}")
    print(f"Evaluation set size: {len(eval_df)}")

    # Convert pandas DataFrames to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df[[text_column, 'label_encoded']].rename(columns={'label_encoded': 'label'}))
    eval_dataset = Dataset.from_pandas(eval_df[[text_column, 'label_encoded']].rename(columns={'label_encoded': 'label'}))


    # Load Tokenizer and Model 
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
    except OSError as e:
        print(f"Error loading model/tokenizer '{model_checkpoint}': {e}")
        print("Please ensure the model name is correct and you have internet access.")
        return {
            "model_name": model_name, "accuracy": 0, "precision": 0, "recall": 0,
            "f1_score": 0, "classification_report": f"Failed to load model: {e}"
        }
    except Exception as e: # Catch other potential errors
         print(f"An unexpected error occurred loading model/tokenizer: {e}")
         return {
            "model_name": model_name, "accuracy": 0, "precision": 0, "recall": 0,
            "f1_score": 0, "classification_report": f"Model/Tokenizer loading error: {e}"
        }


    # Tokenize Data 
    def tokenize_function(examples):
        return tokenizer(examples[text_column], truncation=True, max_length=256) # Adjust max_length if needed

    print("Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    # Remove original text column as it's not needed by the model
    tokenized_train_dataset = tokenized_train_dataset.remove_columns([text_column])
    tokenized_eval_dataset = tokenized_eval_dataset.remove_columns([text_column])
    # Set format for PyTorch
    tokenized_train_dataset.set_format("torch")
    tokenized_eval_dataset.set_format("torch")

    # Data Collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    # Setup Training 
    output_dir = os.path.join(output_dir_base, f"{model_name.replace(' ', '_')}_{os.path.basename(model_checkpoint)}")
    print(f"Training output directory: {output_dir}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01, 
        eval_strategy="epoch", 
        save_strategy="epoch",     
        load_best_model_at_end=True, 
        metric_for_best_model="f1", 
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=50, 
        save_total_limit=2, 
        fp16=torch.cuda.is_available(), 
        report_to="none", 
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train and Evaluate 
    print("Starting training...")
    try:
        trainer.train()
        print("Training finished.")

        print("Evaluating model...")
        eval_results = trainer.evaluate()
        print(f"Evaluation results for {model_name}: {eval_results}")

        # Get Detailed Report 
        print("Generating detailed classification report...")
        predictions = trainer.predict(tokenized_eval_dataset)
        y_pred_labels = np.argmax(predictions.predictions, axis=-1)
        y_true_labels = predictions.label_ids

        report = sk_classification_report(
            y_true_labels,
            y_pred_labels,
            target_names=class_names,
            zero_division=0
        )
        print("\nClassification Report:")
        print(report)

        return {
            "model_name": model_name,
            "accuracy": eval_results.get("eval_accuracy", 0),
            "precision": eval_results.get("eval_precision", 0),
            "recall": eval_results.get("eval_recall", 0),
            "f1_score": eval_results.get("eval_f1", 0),
            "classification_report": report
        }

    except Exception as e:
        print(f"\nAn error occurred during training or evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {
            "model_name": model_name, "accuracy": 0, "precision": 0, "recall": 0,
            "f1_score": 0, "classification_report": f"Training/Evaluation Error: {e}"
        }

# Dummy File Creation 
def create_dummy_file(file_path):
    """Creates a dummy TSV file for demonstration."""
    print(f"\nAttempting to create a dummy file at '{file_path}'...")
    try:
        dummy_data = {
             'tweet': [
                 "Wannan fim din yana da ban sha'awa sosai!", # This film is very interesting!
                 "Abincin bai yi min dadi ba.", # The food was not tasty for me.
                 "Ban san abin da zan ce ba game da wannan.", # I don't know what to say about this.
                 "Ina matukar son wannan wakar.", # I really love this song.
                 "Labarin ya bata min rai.", # The news upset me.
                 "Yanayin yana da kyau yau.", # The weather is nice today.
                 "Muna alfahari da ku.", # We are proud of you.
                 "Wannan littafin yana da ban dariya.", # This book is funny.
                 "Na ji haushin hidimar.", # I was annoyed with the service.
                 "Komai ya tafi daidai.", # Everything went well.
                 "Na gaji da wannan aikin.", # I am tired of this work.
                 "Wannan wuri yana da nutsuwa.", # This place is calm.
                 "Sabis ɗin abokin ciniki ya kasance mai ban mamaki.", # Customer service was amazing.
                 "Ban gamsu da siyan ba.", # I am not satisfied with the purchase.
                 "Ba zan sake zuwa nan ba.", # I will not come here again.
                 "Yaron yana wasa a waje.", # The child is playing outside. (Neutral)
             ],
             'label': [
                 'positive', 'negative', 'neutral', 'positive', 'negative', 'neutral',
                 'positive', 'positive', 'negative', 'positive', 'negative', 'neutral',
                 'positive', 'negative', 'negative', 'neutral' # Labels for new examples
             ],
             'cleaned_tweet': [ 
                 "fim din ban sha'awa sosai",
                 "abincin bai yi dadi ba",
                 "ban san abin da zan ce",
                 "matukar son wakar",
                 "labarin bata min rai",
                 "yanayin kyau yau",
                 "muna alfahari da ku",
                 "littafin ban dariya",
                 "ji haushin hidimar",
                 "komai tafi daidai",
                 "gaji da aikin",
                 "wuri nutsuwa",
                 "sabis abokin ciniki ban mamaki", 
                 "ban gamsu da siyan ba",
                 "ba zan sake zuwa nan ba",
                 "yaron wasa waje",
             ]
         }
        dummy_df = pd.DataFrame(dummy_data)
        # Ensure directory exists if path includes directories
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        dummy_df.to_csv(file_path, sep='\t', index=False, encoding='utf-8') # Specify encoding
        print(f"Successfully created a dummy file '{file_path}'.")
        print("Please re-run the script, potentially adjusting arguments if needed,")
        print("OR replace this dummy file with your actual data.")
        print("The dummy data has columns: 'tweet', 'label', 'cleaned_tweet'")
    except Exception as e:
        print(f"Error creating dummy file at '{file_path}': {e}")
        print("Cannot proceed without data.")


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Transformer sentiment analysis models on TSV data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--file-path", "-f",
        required=True,
        help="Path to the input TSV file containing the sentiment data."
    )
    parser.add_argument(
        "--tweet-col",
        default="tweet",
        help="Name of the column containing the raw tweet text."
    )
    parser.add_argument(
        "--cleaned-col",
        default="cleaned_tweet",
        help="Name of the column containing the preprocessed/cleaned tweet text."
    )
    parser.add_argument(
        "--label-col",
        default="label",
        help="Name of the column containing the sentiment labels (e.g., 'positive', 'negative')."
    )
    parser.add_argument(
        "--model-checkpoint", "-m",
        default="bert-base-multilingual-cased",
        help="Hugging Face model identifier (e.g., 'bert-base-multilingual-cased')."
    )
    parser.add_argument(
        "--num-epochs", "-e",
        type=int,
        default=3, 
        help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=8, 
        help="Training and evaluation batch size."
    )
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=2e-5,
        help="Learning rate for the AdamW optimizer."
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./sentiment_results",
        help="Base directory to save model checkpoints and logs."
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--create-dummy",
        action="store_true",
        help="If the specified --file-path is not found, create a dummy file there and exit."
    )

    args = parser.parse_args()

    # Use parsed arguments
    file_path = args.file_path
    tweet_col = args.tweet_col
    cleaned_tweet_col = args.cleaned_col
    label_col = args.label_col
    model_checkpoint = args.model_checkpoint
    output_dir = args.output_dir

    print("--- Configuration ---")
    print(f"Data file: {file_path}")
    print(f"Raw text column: {tweet_col}")
    print(f"Cleaned text column: {cleaned_tweet_col}")
    print(f"Label column: {label_col}")
    print(f"Model checkpoint: {model_checkpoint}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Output Directory: {output_dir}")
    print(f"Seed: {args.seed}")
    print("---------------------")


    # Data Loading and Validation 
    try:
        if not os.path.exists(file_path):
            print(f"Error: The file '{file_path}' was not found.")
            if args.create_dummy:
                create_dummy_file(file_path)
            else:
                print("You can try running with the --create-dummy flag to create a sample file.")
            return

        df = pd.read_csv(file_path, sep='\t', quoting=3, encoding='utf-8') 
        print(f"\nSuccessfully loaded data from '{file_path}'")
        print(f"Columns found: {df.columns.tolist()}")
        print(f"Number of rows: {len(df)}")
        print("\nFirst 5 rows of the dataframe:")
        print(df.head())

        available_cols = df.columns.tolist()
        model1_possible = tweet_col in available_cols and label_col in available_cols
        model2_possible = cleaned_tweet_col in available_cols and label_col in available_cols

        results = []

        # Model 1: Raw Tweets 
        if model1_possible:
            metrics_model1 = train_and_evaluate_transformer(
                df.copy(), tweet_col, label_col, "Model 1 (Raw Tweets)",
                model_checkpoint=model_checkpoint,
                output_dir_base=output_dir,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed
            )
            results.append(metrics_model1)
        else:
            missing_cols = [col for col in [tweet_col, label_col] if col not in available_cols]
            print(f"\nSkipping Model 1 (Raw Tweets) due to missing columns: {missing_cols}")
            results.append({
                "model_name": "Model 1 (Raw Tweets)",
                "accuracy": "N/A", "precision": "N/A", "recall": "N/A", "f1_score": "N/A",
                "classification_report": f"Skipped due to missing columns: {missing_cols}"
            })

        # Model 2: Cleaned Tweets 
        if model2_possible:
            if tweet_col == cleaned_tweet_col and model1_possible:
                 print(f"\nSkipping Model 2 (Cleaned Tweets) because its text column ('{cleaned_tweet_col}') is the same as Model 1.")
                 results.append({
                     "model_name": "Model 2 (Cleaned Tweets)",
                     "accuracy": "N/A", "precision": "N/A", "recall": "N/A", "f1_score": "N/A",
                     "classification_report": f"Skipped: Text column same as Model 1 ('{tweet_col}')"
                 })
            else:
                metrics_model2 = train_and_evaluate_transformer(
                    df.copy(), cleaned_tweet_col, label_col, "Model 2 (Cleaned Tweets)",
                    model_checkpoint=model_checkpoint,
                    output_dir_base=output_dir,
                    num_epochs=args.num_epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    seed=args.seed
                )
                results.append(metrics_model2)
        else:
            missing_cols = [col for col in [cleaned_tweet_col, label_col] if col not in available_cols]
            if label_col in missing_cols and not model1_possible and label_col in [c for c in [tweet_col, label_col] if c not in available_cols]:
                 missing_cols.remove(label_col) 

            if missing_cols: 
                 print(f"\nSkipping Model 2 (Cleaned Tweets) due to missing columns: {missing_cols}")
                 results.append({
                     "model_name": "Model 2 (Cleaned Tweets)",
                     "accuracy": "N/A", "precision": "N/A", "recall": "N/A", "f1_score": "N/A",
                     "classification_report": f"Skipped due to missing columns: {missing_cols}"
                 })
            elif not model1_possible and not missing_cols:
                 pass 
        # Comparison of Metrics 
        print("\n\n--- Overall Metrics Comparison ---")
        print("------------------------------------------------------------------------------------")
        print(f"{'Model Name':<30} | {'Accuracy':<10} | {'Precision (w)':<15} | {'Recall (w)':<12} | {'F1-score (w)':<14}")
        print("------------------------------------------------------------------------------------")
        for res in results:
             acc = f"{res['accuracy']:.4f}" if isinstance(res['accuracy'], (float, np.floating)) else res['accuracy']
             prec = f"{res['precision']:.4f}" if isinstance(res['precision'], (float, np.floating)) else res['precision']
             rec = f"{res['recall']:.4f}" if isinstance(res['recall'], (float, np.floating)) else res['recall']
             f1 = f"{res['f1_score']:.4f}" if isinstance(res['f1_score'], (float, np.floating)) else res['f1_score']
             print(f"{res['model_name']:<30} | {acc:<10} | {prec:<15} | {rec:<12} | {f1:<14}")
        print("------------------------------------------------------------------------------------")
        print("(w) denotes weighted average metric.")

        print("\n--- Detailed Classification Reports ---")
        for res in results:
             print(f"\nReport for {res['model_name']}:")
             report_content = res.get('classification_report', 'Report not generated.')
             if isinstance(report_content, str):
                 print(report_content)
             else:
                 print("Report data is not in the expected string format.")
             print("-----------------------------------")


    except FileNotFoundError: 
        print(f"Error: The file '{file_path}' was not found.")
        if args.create_dummy:
            create_dummy_file(file_path)
        else:
            print("Please ensure the file exists and the path is correct.")
            print("You can try running with the --create-dummy flag.")
    except pd.errors.EmptyDataError:
         print(f"Error: The file '{file_path}' is empty.")
    except ImportError as e:
        print(f"Import Error: {e}. Please ensure all required libraries are installed:")
        print("pip install pandas transformers datasets scikit-learn torch accelerate") 
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("Please check your data file format (TSV), encoding, column names,")
        print("and ensure Hugging Face models can be downloaded (internet access).")
        import traceback
        print("\nTraceback:")
        traceback.print_exc()


if __name__ == '__main__':
    main()