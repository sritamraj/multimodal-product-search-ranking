from src.evaluation.metrics import recall_at_k,mrr,ndcg_at_k

def test_metrics():
    ranked=["a","b","c"]
    rel=["b"]
    relevance={"b":3}
    assert recall_at_k(ranked,rel,2)==1.0
    assert mrr(ranked,rel)==0.5
    assert ndcg_at_k(ranked,relevance,3)>0
