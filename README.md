# ml-practice

Linear regression from scratch with NumPy: gradient descent, feature scaling, and pytest tests that compare against `np.polyfit`.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pytest
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