import numpy as np

def sample_hard_negatives(candidate_ids, relevant_ids, n=5):
    relevant_ids = set(relevant_ids)
    negatives = [x for x in candidate_ids if x not in relevant_ids]
    return negatives[:n]

def pairwise_examples(positive_features, negative_features):
    X, y = [], []
    for pos in positive_features:
        for neg in negative_features:
            X.append(pos - neg)
            y.append(1)
    return np.asarray(X, dtype="float32"), np.asarray(y, dtype="int64")
