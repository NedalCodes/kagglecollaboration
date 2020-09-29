import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
#from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_absolute_error


# Step One: Read the data
trainData = pd.read_csv("train.csv")
y = trainData.Survived
predict = 'Survived'
testData = pd.read_csv("test.csv")
ids = testData.PassengerId
trainData = trainData.drop(['Survived','Name','Ticket','Cabin','PassengerId'],axis=1)
testData = testData.drop(['Name','Ticket','Cabin','PassengerId'],axis=1)
missing_val_count_by_column = trainData.isnull().sum()  # looking at missing values count by columns
#print(missing_val_count_by_column[missing_val_count_by_column>0])

# Step Two: Wrangle the data
# Step Two.a: Convert Sex column to binary
trainData[["Sex"]] = trainData[["Sex"]].replace("male", 1)
trainData[["Sex"]] = trainData[["Sex"]].replace("female", 0)
testData[["Sex"]] = testData[["Sex"]].replace("male", 1)
testData[["Sex"]] = testData[["Sex"]].replace("female", 0)

#split the data so the model never sees the validation data, until time to validate
X_train,X_test,y_train,y_test= train_test_split(trainData,y,train_size=0.8,test_size=0.2,random_state=42)


#Determine data types per column
cat_cols = [cols for cols in trainData.columns
            if trainData[cols].dtype == 'object']
num_cols = [cols for cols in trainData.columns
            if trainData[cols].dtype in ['int64', 'float64']]
all_cols = cat_cols + num_cols
x = trainData[all_cols]

# Set up pipeline for data processing
numericalTransformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('feature_scaling', StandardScaler(with_mean=False))
])

categoricalTransformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder()),
    ('feature_scaling', StandardScaler(with_mean=False))
])
Preprocessor = ColumnTransformer(
    transformers=[
        ('num', numericalTransformer, num_cols),
        ('cat', categoricalTransformer, cat_cols)
                ]
                                )
# Set up a model
#Model 1
#model1 = RandomForestClassifier() #n_estimators=1500, random_state=42, max_depth=8
param_distribs_RFC = {
    'max_depth':[3,5,8,10,None],
    'n_estimators':[10,100,200,300,400,500,750,1000,1500,1600,1700,1800,1900,2000],
    'criterion':['gini','entropy',]
}

#Model 2
#model2 = xgb.XGBClassifier()
param_distribs_XGBC = {
    'n_estimators':range(100,2000,100),
    'learning_rate':[0.02,0.05,0.1]
}

#model3 = SGDclassifier()
#model4 = KNeighborsClassifier

# continue pipeline for rfc path
my_pipeline_RFC = Pipeline(steps=[('preprocessor', Preprocessor),
                              ('rfc model tuneing', GridSearchCV(RandomForestClassifier(),
                                                                 param_grid=param_distribs_RFC,cv=5,
                                                                 refit=True))
                              ])
my_pipeline_RFC.fit(X_train, y_train)
RFC_validation=my_pipeline_RFC.predict(X_test)
mae_RFC=mean_absolute_error(y_test,RFC_validation)
print('rfc mae: ' + str(mae_RFC*100))

my_pipeline_XGBC = Pipeline(steps=[('Preprocessor',Preprocessor),
                                   ('XGBC model tuneing',GridSearchCV(xgb.XGBClassifier(),
                                                                      param_grid=param_distribs_XGBC,cv=5,
                                                                      refit=True))
                                   ])

my_pipeline_XGBC.fit(X_train, y_train,XGBClassifier__early_stopping_rounds=5)
XGBC_validation=my_pipeline_XGBC.predict(X_test)
mae_XGBC=mean_absolute_error(y_test,XGBC_validation)
print('xgbc mae:' + str(mae_XGBC*100))

#predictions = my_pipeline.predict(x)
#print(predictions)

# mae=accuracy_score(y, predictions)
# print('the accuracy is:',mae*100)
#implement confusion matrix?
#scores = -1 * cross_val_score(my_pipeline, x, y, cv=10, scoring='neg_mean_absolute_error')
#print("MAE scores:\n", scores)
#print(scores.mean())


# Make predictions on test data
#p = my_pipeline.predict(testData)
# submission file creation
#res = pd.DataFrame(p, columns=['Survived'])
#submission = pd.concat([ids, res], axis=1)
#submission.to_csv('titanicsubmission2', index=False)
#print('file saved')
