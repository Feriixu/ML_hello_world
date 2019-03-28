import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
test_idx = [0,50,100]

# Training data
train_target = np.delete(iris.target, test_idx, axis=0)
train_data = np.delete(iris.data, test_idx, axis=0)

# Testing data
test_target = list(iris.target[test_idx])
test_data = list(iris.data[test_idx])

# Build tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)
    
print("Test Target:", test_target)
print("Prediction:", clf.predict(test_data))