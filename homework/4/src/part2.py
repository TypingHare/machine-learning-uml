from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# part (a)
# print(wine.DESCR)  # 178 samples, 13 features
wine = load_wine()
X = wine.data
t = wine.target

# part (b)
target_labels = wine.target_names
print(f"Class names for index 13: {target_labels[t[13]]}")
print(f"Class names for index 133: {target_labels[t[133]]}")

# part (c)
X_sc = MinMaxScaler().fit_transform(X)

# part (d)
X_train, X_test, t_train, t_test = train_test_split(X_sc, t, test_size=0.2)

# part (e)
lr = LogisticRegression()
lr.fit(X_train, t_train)

# part (f)
y_pred = lr.predict(X_test)
score = lr.score(X_test, t_test)
print(f"Score: {score}")

# part (g)
print("\n[Report]")
print(classification_report(t_test, y_pred, target_names=target_labels))
