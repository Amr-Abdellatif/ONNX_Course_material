from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

iris = load_iris()
X,y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
joblib.dump(clf, 'output/model.pkl', compress=9)

