from pathlib import Path
import pandas as pd

REQUIRED = ["product_id", "title", "description", "image_path"]

def load_catalog(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing catalog columns: {missing}")
    df["text"] = (df["title"].fillna("") + " " + df["description"].fillna("")).str.strip()
    return df

def load_qrels(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["query_id", "query", "product_id", "relevance"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing qrels columns: {missing}")
    return df
