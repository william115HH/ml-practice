import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

rng = np.random.default_rng(2)
hours = rng.uniform(0, 10, size=30)
score = 45 + 4.5 * hours + rng.normal(0, 5, size=30)
X = hours.reshape(-1, 1)

baseline = cross_val_score(LinearRegression(), X, score, cv=5)
print(f"plain line       mean R² {baseline.mean():7.3f}  std {baseline.std():6.3f}")

for alpha in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    model = make_pipeline(PolynomialFeatures(12), StandardScaler(), Ridge(alpha=alpha))
    scores = cross_val_score(model, X, score, cv=5)
    print(f"alpha {alpha:8}  mean R² {scores.mean():7.3f}  std {scores.std():6.3f}")