import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

# loading breast cancer dataset from sklearn
cancer_data = datasets.load_breast_cancer()

# assigning
X = cancer_data.data
y = cancer_data.target

# test and train data
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.2
)

# make predictions and model
clf = svm.SVC(kernel="linear")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)

# print accuracy of the model
print(accuracy)
