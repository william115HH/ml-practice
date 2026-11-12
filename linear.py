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
