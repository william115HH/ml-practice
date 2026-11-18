import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
hours = rng.uniform(0,10,size=100)
score = 45 + 4.5 * hours + rng.normal(0,5,size=100)

X = hours.reshape(-1,1) #shape (100,1)
y = score

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=2)
print(X_train.shape, X_test.shape)

model = LinearRegression().fit(X_train, y_train)
print("train R^2:", model.score(X_train,y_train))
print("test R^2:", model.score(X_test,y_test))

train_scores, test_scores = [], []
for seed in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
    model = LinearRegression().fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

print("train mean:", np.mean(train_scores), " std:", np.std(train_scores))
print("test  mean:", np.mean(test_scores), " std:", np.std(test_scores))
print("test > train in", sum(t > tr for t, tr in zip(test_scores, train_scores)), "of 20 splits")