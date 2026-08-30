from pathlib import Path
import json, numpy as np, pandas as pd, streamlit as st, yaml
from PIL import Image
from src.models.clip_encoder import CLIPEngine

ROOT=Path(__file__).resolve().parents[1]
CFG=yaml.safe_load(open(ROOT/"configs/config.yaml"))
INDEX=ROOT/"artifacts/index"

st.set_page_config(page_title="Multimodal Product Search",layout="wide")
st.title("🛍️ Multimodal Product Search")
st.caption("Text + image product representations with ANN retrieval.")

@st.cache_resource
def load_assets():
    engine=CLIPEngine(CFG["model"]["name"])
    text=np.load(INDEX/"text_vectors.npy")
    image=np.load(INDEX/"image_vectors.npy")
    catalog=pd.read_csv(ROOT/"data/catalog.csv")
    return engine,text,image,catalog

query=st.text_input("Search products", "black running shoes")
k=st.slider("Top-K",1,20,10)
if st.button("Search"):
    engine,text,image,catalog=load_assets()
    qv=engine.encode_text([query])[0]
    scores=CFG["model"]["text_weight"]*(text@qv)+CFG["model"]["image_weight"]*(image@qv)
    order=np.argsort(-scores)[:k]
    for i in order:
        r=catalog.iloc[int(i)]
        cols=st.columns([1,4])
        with cols[0]:
            try:
                st.image(Image.open(r.image_path),width=150)
            except Exception:
                st.write("Image unavailable")
        with cols[1]:
            st.subheader(r.title)
            st.write(r.description)
            st.caption(f"Multimodal score: {scores[i]:.4f}")
