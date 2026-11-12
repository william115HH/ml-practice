import numpy as np

from linear import fit, predict

# hours studied -> exam score
x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
y = np.array([52, 55, 61, 64, 70, 73, 79, 82], dtype=float)
w, b, x_mean, x_std = fit(x, y)
print("Final:", w, b)
print(np.polyfit(x, y, 1))
print("Prediction at 5 hours:", predict(5, w, b, x_mean, x_std))