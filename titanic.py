import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Step One: Read the data
trainData = pd.read_csv("train.csv")
testData = pd.read_csv("test.csv")
# Trimming the data down to desirable features
trainData = trainData[['Survived', 'PassengerId', 'Pclass', 'Age', 'Sex', 'SibSp', 'Parch']]
testData = testData[['PassengerId', 'Pclass', 'Age', 'Sex', 'SibSp', 'Parch']]
missing_val_count_by_column = trainData.isnull().sum()  # looking at missing values count by columns
# print(missing_val_count_by_column[missing_val_count_by_column>0])


# Step Two: Wrangle the data
# Step Two.a: Impute missing data
trainData[["Sex"]] = trainData[["Sex"]].replace("male", 1)
trainData[["Sex"]] = trainData[["Sex"]].replace("female", 0)
testData[["Sex"]] = testData[["Sex"]].replace("male", 1)
testData[["Sex"]] = testData[["Sex"]].replace("female", 0)
my_imputer = SimpleImputer()  # imputing missing ages as an average of the test data column
trainData_imputed = pd.DataFrame(my_imputer.fit_transform(trainData.copy()))
trainData_imputed.columns = trainData.columns
testData_imputed = pd.DataFrame(my_imputer.fit_transform(testData.copy()))
testData_imputed.columns = testData.columns
# imputing complete and .trainData_imputed. is now a complete dataset

# Step Two.b: Create Visuals for the data
cols = ['Survived', 'Sex', 'Pclass', 'SibSp']
nr_rows = 2
nr_cols = 2
fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols * 3.5, nr_rows * 3))
for r in range(0, nr_rows):
    for c in range(0, nr_cols):
        i = r * nr_cols + c
        ax = axs[r][c]
        sns.countplot(trainData[cols[i]], hue=trainData["Survived"], ax=ax)
        ax.set_title(cols[i], fontsize=14, fontweight='bold')
        ax.legend(title="survived", loc='upper center')

plt.tight_layout()
plt.show()

# Step Three: Split the data
print(trainData_imputed.head())
predict = "Survived"
X_train = np.array(trainData_imputed.drop([predict], 1))
y_train = np.array(trainData_imputed[predict])
print(X_train, y_train)
# test_train_split?


# Step Four: Model the data
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Step Five: Predict future data
predictions = model.predict(testData)

# Step Six: Compare predictions and evaluate model
for x in range(len(predictions)):
    print(predictions[x],X[x], y[x])

#submission.to_csv('nedal-titanic', index=False)