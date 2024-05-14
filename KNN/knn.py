import pandas as pd
import numpy as np
import sklearn
import sklearn.model_selection
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

# Load and read data
data = pd.read_csv("car.data")
col_name = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
data.columns = col_name
print(data.head())

# Convert to numeric and list data
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
doors = le.fit_transform(list(data["doors"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

# y being the target value
X = list(zip(buying, maint, doors, persons, lug_boot, safety))
y = list(cls)

# Train the model
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.2
)
# n_neighbors = 7 gave me the best accuracy of the model
model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)
print(acc)

# Test the model
predict = model.predict(X_test)
names = ["unnac", "acc", "good", "vgood"]
for x in range(len(predict)):
    print(
        "Predicted: ",
        names[predict[x]],
        "Data: ",
        X_test[x],
        "Actual: ",
        names[y_test[x]],
    )
    # Find distance of k-neighbors
    n = model.kneighbors([X_test[x]])
    print("N: ", n)
