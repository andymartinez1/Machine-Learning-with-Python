# %%
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn import linear_model
from sklearn.utils import shuffle
from matplotlib import style


# Read csv
data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data.head())

# Check for null values
print("Checking for null values\n", data.isnull().sum())

# x-axis will show data from above exluding G3. y-axis is G3
X = np.array(data.drop(["G3"], axis=1))
y = np.array(data["G3"])

# Setting train and test data using sklearn and looping to find the most accurate model
better_model = 0
for _ in range(50):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2
    )
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    if acc > better_model:
        better_model = acc

# Shows the train/test split
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print(better_model)

# Check coef and intercept
print("Coefficient: ", linear.coef_)
print("Intercept: ", linear.intercept_)

# Make a prediction
predict = linear.predict(x_test)
for x in range(len(predict)):
    print(predict[x], x_test[x], y_test[x])

# Data correlation
print("Data Correlation; ", data.corr()["studytime"]["G3"])

# Plotting
p = "studytime"
style.use("ggplot")
plt.scatter(data[p], data["G3"])
plt.xlabel("Studytime in Hours")
plt.ylabel("Final Grades")
plt.show()
