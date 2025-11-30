#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import json
import pandas as pd
import numpy as np

# Optional but recommended
from helper_functions import compute_metrics_str

def simple_exact_match(preds, refs):
    preds_n = [str(p).strip() for p in preds]
    refs_n = [str(r).strip() for r in refs]
    return float(np.mean([p == r for p, r in zip(preds_n, refs_n)]))


def parse_args():
    p = argparse.ArgumentParser("Evaluate predictions already present in a CSV.")
    p.add_argument("--csv_path", type=str, default="out_2_c.tsv",
                   help="CSV with columns: Noisy, Clean, Corrected")
    p.add_argument("--output_json", type=str, default=None,
                   help="Optional path to save metrics JSON.")
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.csv_path, sep='\t')
    if os.path.exists('to_ignore.txt'):
        # ignored some indices where the test set had 0.5 similarity to the AfriTEVA
        to_ignore = [int(l) for l in open('to_ignore.txt', 'r', encoding='utf-8').read().splitlines()]
        df = df[df.index.isin(to_ignore)]

    #required = ["Noisy", "Clean", "Corrected"]
    #missing = [c for c in required if c not in df.columns]
    #if missing:
    #    raise ValueError(f"CSV missing required columns: {missing}")

    #df = df.dropna(subset=["Clean", "Corrected"])
    #preds = df["Corrected"].astype(str).tolist()
    preds = []
    all_metrics = []
    #refs = df["Clean"].astype(str).tolist()
    refs = df["Clean"].astype(str).str.lower().tolist()
    for col in df.columns:
        if 'Corrected' in col:
            #preds = df[col].astype(str).tolist()
            preds = df[col].astype(str).str.lower().tolist()
        else:
            continue
        metrics = compute_metrics_str(preds, refs)
        print(f"\n--- Evaluation {col} ---")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        all_metrics.append((col.replace("Corrected_", ""), metrics))
        
    evdf = pd.DataFrame(
        {
            "Model": [m[0] for m in all_metrics],
            **{k: [m[1][k] for m in all_metrics] for k in all_metrics[0][1].keys()}
        })
    # evdf = pd.DataFrame(all_metrics).to_csv("evaluation_results.csv", index=False)
    evdf = evdf.round(3)
    print(evdf)
    print(evdf.to_markdown())
    if args.output_json:
        evdf.to_json(args.output_json, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
