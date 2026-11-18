# ml-practice

Linear regression from scratch with NumPy: gradient descent, feature scaling, and pytest tests that compare against `np.polyfit`.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas pytest
```

## Run

```
python3 demo.py
python3 -m pytest
```

## How it works

1. **Predict:** multiply the feature matrix by the weights and add the bias.
2. **Measure:** subtract the true values to get the error, and average the squared errors to get the loss.
3. **Gradient:** multiply the transposed features by the error to find how the loss changes with each weight.
4. **Nudge:** move each weight a small step (the learning rate) in the opposite direction, then repeat.

## What I learned

## What I learned

**Train vs. test.** A model's score on the data it trained on doesn't say much about its actual accuracy. In `overfit_demo.py`, the data is a straight line plus some random noise. As I raised the polynomial degree from 1 to 12, train R² rose from about 0.90 to 0.98, but test R² collapsed (as low as -13,204 on one seed), because the model was overfitting the line based on the noise.

**Regularization.** Ridge regression adds a penalty on large weights, which keeps the curve from swinging wildly between points like it did at degree 12. With Ridge, degree 12 test R² came back to roughly 0.80 to 0.92. Too much penalty underfits, though, at alpha = 1000, R² fell near zero.

**Cross-validation.** A single split of 25 or 15 test points is noisy, and the winning model changed from seed to seed. Averaging scores across 5 folds (`cv_demo.py`) gave a more fair comparison. Even then, a regularized degree 12 model only matched the plain straight line, so the simplest model that fits the pattern is usually the right choice.