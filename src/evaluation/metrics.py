import math

def recall_at_k(ranked_ids, relevant_ids, k=10):
    rel = set(relevant_ids)
    return float(bool(rel) and len(set(ranked_ids[:k]) & rel) / len(rel))

def mrr(ranked_ids, relevant_ids):
    rel = set(relevant_ids)
    for i, pid in enumerate(ranked_ids, 1):
        if pid in rel:
            return 1.0 / i
    return 0.0

def ndcg_at_k(ranked_ids, relevance, k=10):
    gains = [relevance.get(pid, 0) for pid in ranked_ids[:k]]
    dcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0
