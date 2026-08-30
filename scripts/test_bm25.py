import pandas as pd

from src.data.catalog import load_catalog
from src.retrieval.bm25 import BM25Retriever


catalog = load_catalog("data/catalog.csv")

retriever = BM25Retriever(catalog["text"].tolist())

queries = [
    ("q001", "black on-ear headphones"),
    ("q002", "women's white sandals"),
    ("q003", "shower caddy"),
    ("q004", "packing cubes for travel"),
    ("q005", "stainless steel dining table cover"),
]

for query_id, query in queries:
    order, scores = retriever.search(query, k=10)

    print("\n" + "=" * 80)
    print(query_id, ":", query)
    print("=" * 80)

    for rank, (idx, score) in enumerate(zip(order, scores), start=1):
        row = catalog.iloc[idx]

        print(
            f"{rank:2}. "
            f"{row['product_id']} | "
            f"{score:.4f} | "
            f"{row['title']}"
        )