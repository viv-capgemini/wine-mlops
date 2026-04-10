"""Download the wine dataset from scikit-learn and save it as a CSV."""
from __future__ import annotations

import argparse
import os

import pandas as pd
from sklearn.datasets import load_wine


def parse_args():
    p = argparse.ArgumentParser("Download wine dataset from scikit-learn")
    p.add_argument(
        "--output",
        default="data/wine_sample.csv",
        help="Output CSV path (default: data/wine_sample.csv)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("Downloading wine dataset from scikit-learn...")
    wine = load_wine(as_frame=True)

    df = wine.frame.copy()

    # Rename 'target' to 'quality' to match train.py expectations
    df.rename(columns={"target": "quality"}, inplace=True)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")
    print(f"Columns: {list(df.columns)}")
    print(f"Target distribution:\n{df['quality'].value_counts().sort_index()}")


if __name__ == "__main__":
    main()
