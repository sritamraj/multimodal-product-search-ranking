import numpy as np
from src.retrieval.multimodal import fuse

def test_fuse_shape():
    a=np.eye(2,dtype="float32")
    b=np.eye(2,dtype="float32")
    x=fuse(a,b)
    assert x.shape==(2,2)
