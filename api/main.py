from pathlib import Path
import json, numpy as np, yaml
from fastapi import FastAPI, Query
from src.models.clip_encoder import CLIPEngine

ROOT=Path(__file__).resolve().parents[1]
CFG=yaml.safe_load(open(ROOT/"configs/config.yaml"))
INDEX=ROOT/"artifacts/index"
app=FastAPI(title="Multimodal Product Search API")

_engine=None
_text=None
_image=None
_multi=None
_meta=None
_catalog=None

def load():
    global _engine,_text,_image,_multi,_meta,_catalog
    if _engine is None:
        import pandas as pd
        _engine=CLIPEngine(CFG["model"]["name"])
        _text=np.load(INDEX/"text_vectors.npy")
        _image=np.load(INDEX/"image_vectors.npy")
        _multi=np.load(INDEX/"multimodal_vectors.npy")
        _meta=json.loads((INDEX/"metadata.json").read_text())
        _catalog=pd.read_csv(ROOT/"data/catalog.csv")

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/search")
def search(q: str = Query(...), k: int = Query(10, ge=1, le=100)):
    load()
    qv=_engine.encode_text([q])[0]
    score=CFG["model"]["text_weight"]*(_text@qv)+CFG["model"]["image_weight"]*(_image@qv)
    order=np.argsort(-score)[:k]
    rows=[]
    for i in order:
        r=_catalog.iloc[int(i)]
        rows.append({"product_id":r.product_id,"title":r.title,
                     "description":r.description,"score":float(score[i]),
                     "image_path":r.image_path})
    return {"query":q,"results":rows}
