from src.ranking.features import build_features
class RankerArtifact:
    def __init__(self, model):
        self.model = model
    @staticmethod
    def feature_builder(qv, tv, iv):
        return build_features(qv, tv, iv)
