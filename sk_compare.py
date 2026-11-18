import pandas as pd

from sklearn.linear_model import LinearRegression

df = pd.read_csv("students.csv")
X = df[["hours_studied", "hours_slept"]]
y = df["score"]

model = LinearRegression().fit(X,y)
print("coef:", model.coef_)
print("intercept:",model.intercept_)
new_student = pd.DataFrame([[5,7]], columns=["hours_studied","hours_slept"])
print("prediction:", model.predict(new_student))