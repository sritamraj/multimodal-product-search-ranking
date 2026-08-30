import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from src.data.catalog import load_catalog, load_qrels
from src.models.clip_encoder import CLIPEngine
from src.retrieval.bm25 import BM25Retriever
from src.evaluation.metrics import recall_at_k, mrr, ndcg_at_k
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--ranker", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open("configs/config.yaml"))
    catalog = load_catalog(args.catalog)
    qrels = load_qrels(args.qrels)
    idx = Path(args.index)
    text = np.load(idx / "text_vectors.npy")
    image = np.load(idx / "image_vectors.npy")
    multi = np.load(idx / "multimodal_vectors.npy")
    meta = json.loads((idx / "metadata.json").read_text())
    ids = meta["product_ids"]
    engine = CLIPEngine(cfg["model"]["name"])
    bm25 = BM25Retriever(catalog["text"].tolist())
    queries = qrels.groupby(
        ["query_id", "query"],
        sort=False
    )
    methods = [
        "BM25",
        "Text Encoder",
        "Image Encoder",
        "Multimodal"
    ]
    sums = {
        m: [0.0, 0.0, 0.0]
        for m in methods
    }
    reranker_scores = []
    for (qid, query), g in queries:
        relevant = [
            p
            for p, r in zip(g.product_id, g.relevance)
            if r > 0
        ]
        relevance = {
            p: int(r)
            for p, r in zip(g.product_id, g.relevance)
        }
        qv = engine.encode_text([query])[0]
        text_scores = text @ qv
        image_scores = image @ qv
        multi_scores = multi @ qv
        # Baselines
        order, _ = bm25.search(query, 10)
        baseline_methods = {
            "BM25": [ids[i] for i in order],
            "Text Encoder": [
                ids[i]
                for i in np.argsort(-text_scores)[:10]
            ],
            "Image Encoder": [
                ids[i]
                for i in np.argsort(-image_scores)[:10]
            ],
            "Multimodal": [
                ids[i]
                for i in np.argsort(-multi_scores)[:10]
            ],
        }
        for name, ranked in baseline_methods.items():
            sums[name][0] += recall_at_k(
                ranked, relevant, 10
            )
            sums[name][1] += mrr(
                ranked, relevant
            )
            sums[name][2] += ndcg_at_k(
                ranked, relevance, 10
            )
        if args.ranker:
            from joblib import load
            ranker = load(args.ranker)
            # Retrieve candidates using the original
            # multimodal retrieval score.
            candidates = np.argsort(-multi_scores)[
                :cfg["retrieval"]["ann_candidates"]
            ]
            X = np.vstack([
                ranker.feature_builder(
                    qv,
                    text[i],
                    image[i]
                )
                for i in candidates
            ])
            learned = ranker.model.predict_proba(X)[:, 1]
            # Normalize both signals before blending.
            base = multi_scores[candidates]
            base_min = base.min()
            base_max = base.max()
            if base_max > base_min:
                base_norm = (
                    (base - base_min)
                    / (base_max - base_min)
                )
            else:
                base_norm = np.zeros_like(base)
            learn_min = learned.min()
            learn_max = learned.max()
            if learn_max > learn_min:
                learned_norm = (
                    (learned - learn_min)
                    / (learn_max - learn_min)
                )
            else:
                learned_norm = np.zeros_like(learned)
            # Keep retrieval as the dominant signal.
            final_score = (
                0.70 * base_norm
                + 0.30 * learned_norm
            )
            order = candidates[
                np.argsort(-final_score)[:10]
            ]
            ranked = [ids[i] for i in order]
            reranker_scores.append(
                (
                    recall_at_k(
                        ranked,
                        relevant,
                        10
                    ),
                    mrr(
                        ranked,
                        relevant
                    ),
                    ndcg_at_k(
                        ranked,
                        relevance,
                        10
                    )
                )
            )
    rows = []
    for name, vals in sums.items():
        rows.append({
            "Model": name,
            "Recall@10": vals[0] / len(queries),
            "MRR": vals[1] / len(queries),
            "NDCG@10": vals[2] / len(queries)
        })
    if reranker_scores:
        a = np.mean(
            reranker_scores,
            axis=0
        )
        rows.append({
            "Model": "Multimodal + Hard Negatives",
            "Recall@10": a[0],
            "MRR": a[1],
            "NDCG@10": a[2]
        })
    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(
        parents=True,
        exist_ok=True
    )
    out.to_csv(
        args.output,
        index=False
    )
    print(out.to_string(index=False))
if __name__ == "__main__":
    main()
