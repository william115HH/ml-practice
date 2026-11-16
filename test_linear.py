import numpy as np

from linear import fit, predict, fit_multi, predict_multi


def test_matches_polyfit():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    y = np.array([52, 55, 61, 64, 70, 73, 79, 82], dtype=float)
    w, b, x_mean, x_std = fit(x, y)
    expected = np.polyval(np.polyfit(x, y, 1), 5)
    assert abs(predict(5, w, b, x_mean, x_std) - expected) < 0.5


def test_perfect_line():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = 2 * x + 1
    w, b, x_mean, x_std = fit(x, y)
    assert abs(predict(10, w, b, x_mean, x_std) - 21) < 0.1


def test_constant_x():
    x = np.array([3, 3, 3, 3], dtype=float)
    y = np.array([10, 20, 30, 40], dtype=float)
    w, b, x_mean, x_std = fit(x, y)
    result = predict(3, w, b, x_mean, x_std)
    assert not np.isnan(result)
    assert abs(result - y.mean()) < 0.1

def test_multi_perfect():
    X = np.array([[1, 5], [2, 3], [3, 8], [4, 1], [5, 6], [6, 2], [7, 9]], dtype=float)
    y = 3*X[:,0] +2*X[:,1] + 1
    w, b, X_mean, X_std = fit_multi(X, y, steps=5000)
    result = predict_multi(np.array([[10.0,4.0]]), w, b, X_mean, X_std)
    assert abs(result[0]-39) < 0.1

def test_multi_constant_column():
    X = np.array([[1, 5], [2, 5], [3, 5], [4, 5], [5, 5]], dtype=float)
    y = np.array([3, 5, 7, 9, 11], dtype=float)   # y = 2*x1 + 1
    w, b, X_mean, X_std = fit_multi(X, y, steps=5000)
    result = predict_multi(np.array([[6.0, 5.0]]), w, b, X_mean, X_std)
    assert not np.isnan(result).any()
    assert abs(result[0]-13) < 0.1