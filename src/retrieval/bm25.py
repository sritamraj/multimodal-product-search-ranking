import numpy as np
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, texts):
        self.tokens = [str(t).lower().split() for t in texts]
        self.bm25 = BM25Okapi(self.tokens)

    def search(self, query, k=10):
        scores = np.asarray(self.bm25.get_scores(str(query).lower().split()), dtype="float32")
        order = np.argsort(-scores)[:k]
        return order, scores[order]
