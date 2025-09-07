from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest

class OutlierScorer(BaseEstimator, TransformerMixin):
    def __init__(self, contamination=0.01, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.iso = None

    def fit(self, X, y=None):
        self.iso = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state
        )
        self.iso.fit(X)
        return self

    def transform(self, X, y=None):
        scores = self.iso.decision_function(X)  # higher = more normal
        X_new = X.copy()
        X_new["outlier_score"] = scores
        return X_new