from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
class CLIPEngine:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    @torch.inference_mode()
    def encode_text(self, texts, batch_size=8):
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            emb = self.model.get_text_features(**inputs)
            if hasattr(emb, "pooler_output"):
                emb = emb.pooler_output
            emb = emb / emb.norm(dim=-1, keepdim=True)
            out.append(emb.cpu().numpy().astype("float32"))
        return np.vstack(out)
    @torch.inference_mode()
    def encode_images(self, paths, batch_size=4):
        out = []
        valid = []
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i+batch_size]
            images = []
            keep = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(img)
                    keep.append(True)
                except Exception:
                    images.append(Image.new("RGB", (224, 224), "white"))
                    keep.append(False)
            inputs = self.processor(
                images=images,
                return_tensors="pt"
            ).to(self.device)
            emb = self.model.get_image_features(**inputs)
            if hasattr(emb, "pooler_output"):
                emb = emb.pooler_output
            emb = emb / emb.norm(dim=-1, keepdim=True)
            out.append(emb.cpu().numpy().astype("float32"))
            valid.extend(keep)
        return np.vstack(out), np.asarray(valid)

