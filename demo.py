import numpy as np

from linear import fit, predict, fit_multi, predict_multi

# hours studied -> exam score
x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
y = np.array([52, 55, 61, 64, 70, 73, 79, 82], dtype=float)
w, b, x_mean, x_std = fit(x, y)
print("Final:", w, b)
print(np.polyfit(x, y, 1))
print("Prediction at 5 hours:", predict(5, w, b, x_mean, x_std))

X = np.array([[1, 6], [2, 5], [3, 8], [4, 6], [5, 7], [6, 8], [7, 6], [8, 9]], dtype=float)
y = np.array([52, 55, 61, 64, 70, 73, 79, 82], dtype=float)
A = np.column_stack([X, np.ones(len(X))])
print(np.linalg.lstsq(A, y, rcond=None)[0])
w, b, X_mean, X_std = fit_multi(X,y)
print("Final:", w, b)
print("Prediction is ", predict_multi(np.array([[5, 7]]), w, b, X_mean, X_std))
