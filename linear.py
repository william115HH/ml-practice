import numpy as np

def predict(hours, w, b, x_mean, x_std):
    hours_scaled = (hours - x_mean) / x_std
    return w * hours_scaled + b

def fit(x, y, learning_rate=0.1, steps=1000, verbose=False):
    w = 0.0  # weight (slope)
    b = 0.0  # bias (intercept)

    x_mean = x.mean()
    x_std = x.std()
    if x_std == 0:
        x_std = 1.0
    x = (x-x_mean)/x_std
    for step in range(steps):
        predictions = w * x + b
        error = predictions - y
        loss = np.mean(error ** 2)

        grad_w = 2 * np.mean(error * x)
        grad_b = 2 * np.mean(error)

        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

        if step % 100 == 0 and verbose:
            print(f"step {step:4d}  loss {loss:8.3f}  w {w:.3f}  b {b:.3f}")
    return w, b, x_mean, x_std

def fit_multi(X, y, learning_rate=0.1, steps=1000, verbose=False):
    X_mean = X.mean(axis=0) #shape (n_features,)
    X_std = X.std(axis=0) #shape (n_features,)
    X_std[X_std == 0] = 1.0
    X_scaled = (X - X_mean) / X_std #shape (n_samples, n_features)

    w = np.zeros(X.shape[1])

    b = 0.0
    
    for step in range(steps):
        predictions = X_scaled @ w + b #shape (n_samples,)
        error = predictions - y #shape (n_samples,)
        assert predictions.shape == y.shape
        
        grad_w = 2 * (X_scaled.T @ error) / len(y) #shape (n_features,)
        grad_b = 2 * np.mean(error)

        w -= learning_rate * grad_w
        b-= learning_rate * grad_b

        if verbose and step % 100 == 0:
            print(f"step{step:4d}  loss {np.mean(error ** 2):8.3f} ")
    return w, b, X_mean, X_std


def predict_multi(X_new, w, b, X_mean, X_std):
    return ((X_new - X_mean)/X_std) @ w + b

