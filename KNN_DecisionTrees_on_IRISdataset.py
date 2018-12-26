

from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
Y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# various ML classifiers

#Decision Tree

from sklearn import tree
clf = tree.DecisionTreeClassifier()

#K-Nearest Neighbors (KNN)

from sklearn.neighbors import KNeighborsClassifier
clf1 = KNeighborsClassifier()

#KNN from Scratch

from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

class ScrappyKNN():
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, X_test):
        prediction = []
        for row in X_test:
            label = self.closest(row)
            prediction.append(label)
        return prediction

    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.Y_train[best_index]

clf2 = ScrappyKNN()

#Training all all the above and comparing their accuracy:

clf.fit(X_train, Y_train)
clf1.fit(X_train, Y_train)
clf2.fit(X_train, Y_train)

prediction = clf.predict(X_test)
prediction1 = clf1.predict((X_test))
prediction2 = clf2.predict((X_test))

from sklearn.metrics import accuracy_score

print (accuracy_score(Y_test, prediction))
print (accuracy_score(Y_test, prediction1))
print (accuracy_score(Y_test, prediction2))

