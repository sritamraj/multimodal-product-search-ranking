import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from joblib import dump
from sklearn.linear_model import LogisticRegression
from src.data.catalog import load_catalog, load_qrels
from src.models.clip_encoder import CLIPEngine
from src.ranking.features import build_features
from src.ranking.ranker_artifact import RankerArtifact
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open("configs/config.yaml"))
    catalog = load_catalog(args.catalog)
    qrels = load_qrels(args.qrels)
    idx = Path(args.index)
    text = np.load(idx / "text_vectors.npy")
    image = np.load(idx / "image_vectors.npy")
    meta = json.loads((idx / "metadata.json").read_text())
    ids = meta["product_ids"]
    id_to_row = {p: i for i, p in enumerate(ids)}
    engine = CLIPEngine(cfg["model"]["name"])
    X = []
    y = []
    for (_, query), g in qrels.groupby(
        ["query_id", "query"], sort=False
    ):
        qv = engine.encode_text([query])[0]
        relevance = {
            p: int(r)
            for p, r in zip(g.product_id, g.relevance)
            if p in id_to_row
        }
        base_scores = (
            cfg["model"]["text_weight"] * (text @ qv)
            + cfg["model"]["image_weight"] * (image @ qv)
        )
        candidates = np.argsort(-base_scores)[
            :cfg["retrieval"]["ann_candidates"]
        ]
        candidate_ids = [ids[i] for i in candidates]
        # Compare every higher-relevance item against
        # lower-relevance items retrieved by the first stage.
        for p, p_rel in relevance.items():
            if p_rel <= 0:
                continue
            p_row = id_to_row[p]
            pf = build_features(
                qv,
                text[p_row],
                image[p_row],
            )
            negatives = [
                n for n in candidate_ids
                if relevance.get(n, 0) < p_rel
            ]
            negatives = negatives[
                :cfg["ranking"]["hard_negative_multiplier"]
            ]
            for n in negatives:
                n_row = id_to_row[n]
                nf = build_features(
                    qv,
                    text[n_row],
                    image[n_row],
                )
                # Positive/relevant item should score higher.
                X.append(pf - nf)
                y.append(1)
                # Reverse pair.
                X.append(nf - pf)
                y.append(0)
    if not X:
        raise RuntimeError(
            "No training pairs. Check qrels, product IDs "
            "and retrieval candidates."
        )
    X = np.asarray(X, dtype="float32")
    y = np.asarray(y, dtype="int64")
    print(f"Training examples: {len(y)}")
    print(f"Positive: {(y == 1).sum()}")
    print(f"Negative: {(y == 0).sum()}")
    model = LogisticRegression(
        max_iter=2000,
        C=0.1,
        class_weight="balanced",
        random_state=cfg["seed"],
    )
    model.fit(X, y)
    artifact = RankerArtifact(model)
    dump(artifact, args.output)
    print(f"Saved relevance-aware reranker to {args.output}")
    print("Learned coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
if __name__ == "__main__":
    main()
