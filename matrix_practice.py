import numpy as np

X = np.array([[1, 2],
              [3, 4],
              [5, 6]], dtype=float)   # 3 students, 2 features
w = np.array([10.0, 1.0])

print(X.shape)
print((X @ w).shape)
print(X @ w)

a = np.array([1.0, 2.0, 3.0])   # shape (3,)
b = a.reshape(3, 1)
print((a - b).shape)