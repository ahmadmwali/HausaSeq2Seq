import os
import pandas as pd
import torch
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def parse_args():
    p = argparse.ArgumentParser("Evaluate a seq2seq model on arbitrary CSVs.")
    p.add_argument("--model_id", type=str, required=True,
                   help="HF hub id or local path to a Seq2Seq model.")
    p.add_argument("--csv_path", type=str, required=True,
                   help="Path to CSV with columns: Original, Noisy, Clean.")
    p.add_argument("--output", type=str, default=None,
                   help="Optional path to save the predictions.")
    p.add_argument("--column", type=str, default="Noisy",
                   help="Column name to run generation on.")
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8)
    return p.parse_args()


def batched_generate(model, tokenizer, texts, batch_size, max_seq_length, max_new_tokens, num_beams, device):
    preds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating"):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length
        ).to(device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams
            )
        batch_preds = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        preds.extend([p.strip() for p in batch_preds])
    return preds

def generate_df(
    model_id: str,
    csv_path: str,
    output: str = None,
    batch_size: int = 8,
    max_seq_length: int = 256,
    max_new_tokens: int = 256,
    num_beams: int = 4,
    sep: str = "\t",
    input_col: str = "Noisy"
):
    """
    Load a TSV/CSV, run model on `input_col`, store predictions in a new column,
    and optionally save to disk.

    Returns:
        df (pd.DataFrame): dataframe including prediction column.
    """
    # --- Load CSV/TSV ---
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, sep=sep)

    required_cols = [input_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns {missing}. Found {list(df.columns)}")

    noisy_texts = df[input_col].astype(str).tolist()
    #clean_texts = df[target_col].astype(str).tolist()  # kept in case you want later use

    if len(noisy_texts) == 0:
        raise ValueError("No usable rows after cleaning.")

    # --- Load model/tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- Generate predictions ---
    preds = batched_generate(
        model=model,
        tokenizer=tokenizer,
        texts=noisy_texts,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        device=device
    )

    pred_col = f"Corrected_{model_id}"
    df[pred_col] = preds

    # --- Save if requested ---
    if output is None:
        output = csv_path

    df.to_csv(output, sep=sep, index=False)
    return df


def main():
    args = parse_args()
    generate_df(
        model_id=args.model_id,
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        output=args.output,
        sep="\t",
        input_col=args.column
    )


if __name__ == "__main__":
    main()
