import faiss
import numpy as np

class ANNIndex:
    def __init__(self, vectors: np.ndarray):
        vectors = np.asarray(vectors, dtype="float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query_vectors, k=10):
        q = np.asarray(query_vectors, dtype="float32")
        return self.index.search(q, k)

    def save(self, path):
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        obj.index = faiss.read_index(str(path))
        return obj
