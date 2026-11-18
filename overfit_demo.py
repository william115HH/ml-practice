import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

rng = np.random.default_rng(0)
hours = rng.uniform(0, 10, size=30)
score = 45 + 4.5 * hours + rng.normal(0, 5, size=30)   # still a straight line plus noise

X = hours.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, score, test_size=0.5, random_state=0)

for degree in [1, 3, 8, 12]:
    model = make_pipeline(PolynomialFeatures(degree), StandardScaler(), Ridge(alpha=1))
    model.fit(X_train, y_train)
    print(f"degree {degree:2d}  train R² {model.score(X_train, y_train):7.3f}  test R² {model.score(X_test, y_test):9.3f}")