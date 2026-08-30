import numpy as np
def build_features(query_vec, product_text, product_image):
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    t = product_text / (np.linalg.norm(product_text) + 1e-12)
    im = product_image / (np.linalg.norm(product_image) + 1e-12)
    text_sim = float(q @ t)
    image_sim = float(q @ im)
    multimodal_sim = 0.65 * text_sim + 0.35 * image_sim
    # Interaction: reward products that are strong in both modalities.
    agreement = text_sim * image_sim
    return np.array(
        [
            text_sim,
            image_sim,
            multimodal_sim,
            agreement,
        ],
        dtype="float32",
    )
