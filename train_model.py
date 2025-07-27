import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier



x, y = load_iris(return_X_y=True)

clf = RandomForestClassifier()
clf.fit(x, y)


with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Model saved successfully!")

