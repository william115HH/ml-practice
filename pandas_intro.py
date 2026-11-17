import pandas as pd 
import numpy as np

from linear import fit_multi, predict_multi

df = pd.read_csv("students.csv")

print(df)
print(df.shape)
print(df.describe())
print(df["score"].mean())
print(df[df["hours_studied"]>4])
print(df["hours_studied"].std(ddof=0))

X = df[["hours_studied", "hours_slept"]].to_numpy()
y = df["score"].to_numpy()
w, b, X_mean, X_std = fit_multi(X,y)
print(predict_multi(np.array([[5.0,7.0]]), w, b, X_mean, X_std))