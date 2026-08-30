import numpy as np

def l2_normalize(x):
    x = np.asarray(x, dtype="float32")
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def fuse(text_vectors, image_vectors, text_weight=0.65, image_weight=0.35):
    fused = text_weight * l2_normalize(text_vectors) + image_weight * l2_normalize(image_vectors)
    return l2_normalize(fused)

def score_query(query_text_vector, product_text_vectors, product_image_vectors,
                text_weight=0.65, image_weight=0.35):
    q = l2_normalize(query_text_vector)
    t = l2_normalize(product_text_vectors)
    im = l2_normalize(product_image_vectors)
    return text_weight * (q @ t.T) + image_weight * (q @ im.T)
