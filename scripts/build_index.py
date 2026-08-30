import argparse, json
from pathlib import Path
import numpy as np
import yaml
from src.data.catalog import load_catalog
from src.models.clip_encoder import CLIPEngine
from src.retrieval.ann import ANNIndex
from src.retrieval.multimodal import fuse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open("configs/config.yaml"))
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    df = load_catalog(args.catalog)
    engine = CLIPEngine(cfg["model"]["name"])

    text = engine.encode_text(df["text"].tolist())
    image, valid = engine.encode_images(df["image_path"].tolist())
    multi = fuse(text, image, cfg["model"]["text_weight"], cfg["model"]["image_weight"])

    np.save(out/"text_vectors.npy", text)
    np.save(out/"image_vectors.npy", image)
    np.save(out/"multimodal_vectors.npy", multi)
    ANNIndex(multi).save(out/"multimodal.faiss")
    ANNIndex(text).save(out/"text.faiss")
    ANNIndex(image).save(out/"image.faiss")
    (out/"metadata.json").write_text(json.dumps({
        "product_ids": df.product_id.tolist(),
        "image_valid": valid.tolist(),
        "model_name": cfg["model"]["name"]
    }, indent=2))
    print(f"Built indexes for {len(df)} products at {out}")

if __name__ == "__main__":
    main()
