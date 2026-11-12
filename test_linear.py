import numpy as np

from linear import fit, predict


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