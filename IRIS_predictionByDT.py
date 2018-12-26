import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

#visualize data
for i in range(len(iris.target)):
    print ('example %d: label %s, features %s'% (i, iris.target[i], iris.data[i]))

test_idx = [0, 50, 100]

#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)


#print ("this is actual data %s, and this is our prediction: %s" % (test_target, clf.predict(test_data)))



a = [[5.3, 3.8, 1.6, 0.5]]

print (clf.predict(a))