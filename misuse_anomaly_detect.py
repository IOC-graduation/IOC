import warnings

warnings.filterwarnings(action='ignore')

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve

# -------------------------------------------------------------------------------------------------------------
# Read misuse detection train file
misuse_file = './preprocessed_data/Dataset_Misuse_AttributeSelection.csv'
misuse_df = pd.read_csv(misuse_file)
print(misuse_df)

# fill nan
misuse_df = misuse_df.replace({'AttackType': "NA"}, {'AttackType': "Normal"})
misuse_df = misuse_df.fillna('Normal')

misuse_x = misuse_df.drop('AttackType', axis=1)
misuse_y = misuse_df['AttackType'].values

#------------------------------------------------------
# Read anomaly detection train file
anomaly_file = './preprocessed_data/Dataset_Anomaly_AttributeSelection.csv'
anomaly_df = pd.read_csv(anomaly_file)

print(anomaly_df)

# Data encoding
encoder = LabelEncoder()

# encoding
anomaly_df['protocol_type'] = encoder.fit_transform(anomaly_df['protocol_type'])
anomaly_df['service'] = encoder.fit_transform(anomaly_df['service'])
anomaly_df['flag'] = encoder.fit_transform(anomaly_df['flag'])

anomaly_x = anomaly_df.drop('AttackType', axis=1)
anomaly_y = anomaly_df['AttackType'].values

#------------------------------------------------------
# Read test file
test_file = './preprocessed_data/testData.csv'
test_df = pd.read_csv(test_file)
test_df = test_df.drop('Unnamed: 0',axis = 1)
print(test_df)

test_y = test_df['AttackType'].values
test_df = test_df.drop('AttackType', axis=1)

# encoding
test_df['protocol_type'] = encoder.fit_transform(test_df['protocol_type'])
test_df['service'] = encoder.fit_transform(test_df['service'])
test_df['flag'] = encoder.fit_transform(test_df['flag'])

for i in range(len(test_y)):
    if test_y[i] != 'Normal':
        test_y[i] = 'Attack'

# Scale the data
# Scaling - MinMaxScaler, Normalizer, StandardScaler
MinMax_scaler = MinMaxScaler()
Standard_scaler = StandardScaler()
test_X = MinMax_scaler.fit_transform(test_df)

# -------------------------------------------------------------------------------------------------------------
#Parameter sets for each Classification models
logistic_params = {
   'C': [0.1, 1.0, 10.0],
   'solver': ['liblinear', 'lbfgs', 'sag'],
    'max_iter': [50, 100, 200]
}
decisionTree_params = {
   'max_depth': [None, 6, 8, 10, 12, 16, 20, 24],
   'min_samples_split': [2, 20, 50, 100, 200],
    'criterion': ['entropy', 'gini']
}
randomForest_params = {
   'n_estimators': [200, 500],
   'max_features': ['auto', 'sqrt', 'log2'],
   'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy']
}

gradient_params = {
   "n_estimators": [50, 100],
   "max_depth": [1, 2, 4],
    "learning_rate": [0.01, 0.1]
}

# -------------------------------------------------------------------------------------------------------------
# Scale the data
print("Misuse detection")
X_scaled = MinMax_scaler.fit_transform(misuse_x)

# misuse dectection model train
logistic = LogisticRegression().fit(X_scaled, misuse_y)
decisionTree = DecisionTreeClassifier().fit(X_scaled, misuse_y)
randomForest = RandomForestClassifier().fit(X_scaled, misuse_y)
gradientBoosting = GradientBoostingClassifier().fit(X_scaled, misuse_y)

# GridSearch
cv = KFold(n_splits=10, random_state=1, shuffle=True)

# Logistic Regression
gcv_logistic = GridSearchCV(logistic, param_grid=logistic_params, scoring='accuracy', cv=cv, verbose=1)
gcv_logistic.fit(X_scaled, misuse_y)
print("---------------------------------------------------------------")
print("Logistic Regression")
print('final params', gcv_logistic.best_params_)  # print best parameter value
print('best score', gcv_logistic.best_score_)  # best score
logistic_best = gcv_logistic.best_estimator_
gcv_logistic_score = gcv_logistic.best_score_

index = 0
result_lr = list(logistic_best.predict(test_X))
for i in range(len(result_lr)):
    if result_lr[i] != 'Normal':
        result_lr[i] = 'Attack'

# Confusion Matrix
print(confusion_matrix(test_y, result_lr))
print(classification_report(test_y, result_lr, target_names=['Attack', 'Normal']))

# Decision Tree
gcv_decisionTree = GridSearchCV(decisionTree, param_grid=decisionTree_params, scoring='accuracy', cv=cv, verbose=1)
gcv_decisionTree.fit(X_scaled, misuse_y)
print("---------------------------------------------------------------")
print("Decision Tree")
print('final params', gcv_decisionTree.best_params_)  # print best parameter value
print('best score', gcv_decisionTree.best_score_)  # best score
gcv_decisionTree_score = gcv_decisionTree.best_score_
decisionTree_best = gcv_decisionTree.best_estimator_

index = 0
result_dt = list(decisionTree_best.predict(test_X))
for i in range(len(result_dt)):
    if result_dt[i] != 'Normal':
        result_dt[i] = 'Attack'

# Confusion Matrix
print(confusion_matrix(test_y, result_dt))
print(classification_report(test_y, result_dt, target_names=['Attack', 'Normal']))

# Random Forest
gcv_randomForest = GridSearchCV(randomForest, param_grid=randomForest_params, scoring='accuracy', cv=cv, verbose=1)
gcv_randomForest.fit(X_scaled, misuse_y)
print("---------------------------------------------------------------")
print("Random Forest")
print('final params', gcv_randomForest.best_params_)  # print best parameter value
print('best score', gcv_randomForest.best_score_)  # best score
gcv_randomForest_score = gcv_randomForest.best_score_
randomForest_best = gcv_randomForest.best_estimator_

index = 0
result_rf = list(decisionTree_best.predict(test_X))
for i in range(len(result_rf)):
    if result_rf[i] != 'Normal':
        result_rf[i] = 'Attack'

# Confusion Matrix
print(confusion_matrix(test_y, result_rf))
print(classification_report(test_y, result_rf, target_names=['Attack', 'Normal']))

# Gradient Boosting
gcv_gradientBoosting = GridSearchCV(gradientBoosting, param_grid=gradient_params, scoring='accuracy', cv=cv, verbose=1)
gcv_gradientBoosting.fit(X_scaled, misuse_y)
print("---------------------------------------------------------------")
print("Gradient Boosting")
print('final params', gcv_gradientBoosting.best_params_)  # print best parameter value
print('best score', gcv_gradientBoosting.best_score_)  # best score
gradientBoosting_best = gcv_gradientBoosting.best_estimator_
gcv_gradientBoosting_score = gcv_gradientBoosting.best_score_

index = 0
result_gb = list(gradientBoosting_best.predict(test_X))
for i in range(len(result_gb)):
   if result_gb[i] != 'Normal':
       result_gb[i] = 'Attack'

# Confusion Matrix
print(confusion_matrix(test_y, result_gb))
print(classification_report(test_y, result_gb, target_names=['Attack', 'Normal']))

# Find best model & score

score = [gcv_decisionTree_score, gcv_randomForest_score, gcv_gradientBoosting_score]
models = [decisionTree_best, randomForest_best, gradientBoosting_best]
max_score = gcv_logistic_score
best_model = logistic_best
for i in range(len(score)):
    if max_score < score[i]:
        max_score = score[i]
        best_model = models[i]

print("Misuse detection best score: ", max_score)

# a new dataframe by collecting only normal results through best model
normal_X = pd.DataFrame(columns=test_df.columns)
normal_y = []
index = 0
result = list(best_model.predict(test_X))
for i in range(len(result)):
    if result[i] == 'Normal':
        normal_X.loc[index] = test_X[i]
        normal_y.append(test_y[i])
        index = index + 1
    else:
        #check
        print('index', i,  '---->', result[i])
        result[i] = 'Attack'

normal_y = np.array(normal_y)

# Confusion Matrix
print(confusion_matrix(test_y, result))
print(classification_report(test_y, result, target_names=['Attack', 'Normal']))

print(normal_X.columns)
print(normal_y)

# -------------------------------------------------------------------------
print("Anomaly detection")
value = anomaly_x.columns
X_scaled = MinMax_scaler.fit_transform(anomaly_x)


# misuse dectection model train
logistic = LogisticRegression().fit(X_scaled, anomaly_y)
decisionTree = DecisionTreeClassifier().fit(X_scaled, anomaly_y)
randomForest = RandomForestClassifier().fit(X_scaled, anomaly_y)
gradientBoosting = GradientBoostingClassifier().fit(X_scaled, anomaly_y)

# GridSearch
cv = KFold(n_splits=10, random_state=1, shuffle=True)
# Logistic Regression
gcv_logistic = GridSearchCV(logistic, param_grid=logistic_params, scoring='accuracy', cv=cv, verbose=1)
gcv_logistic.fit(X_scaled, anomaly_y)
print("---------------------------------------------------------------")
print("Logistic Regression")
print('final params', gcv_logistic.best_params_)  # print best parameter value
print('best score', gcv_logistic.best_score_)  # best score
logistic_best = gcv_logistic.best_estimator_
gcv_logistic_score = gcv_logistic.best_score_

index = 0
result_lr = list(logistic_best.predict(normal_X))
# Confusion Matrix
print(confusion_matrix(normal_y, result_lr))
print(classification_report(normal_y, result_lr, target_names=['Attack', 'Normal']))

# Decision Tree
gcv_decisionTree = GridSearchCV(decisionTree, param_grid=decisionTree_params, scoring='accuracy', cv=cv, verbose=1)
gcv_decisionTree.fit(X_scaled, anomaly_y)
print("---------------------------------------------------------------")
print("Decision Tree")
print('final params', gcv_decisionTree.best_params_)  # print best parameter value
print('best score', gcv_decisionTree.best_score_)  # best score
decisionTree_best = gcv_decisionTree.best_estimator_
gcv_decisionTree_score = gcv_decisionTree.best_score_

index = 0
result_dt = list(decisionTree_best.predict(normal_X))
# Confusion Matrix
print(confusion_matrix(normal_y, result_dt))
print(classification_report(normal_y, result_dt, target_names=['Attack', 'Normal']))

# Random Forest
gcv_randomForest = GridSearchCV(randomForest, param_grid=randomForest_params, scoring='accuracy', cv=cv, verbose=1)
gcv_randomForest.fit(X_scaled, anomaly_y)
print("---------------------------------------------------------------")
print("Random Forest")
print('final params', gcv_randomForest.best_params_)  # print best parameter value
print('best score', gcv_randomForest.best_score_)  # best score
randomForest_best = gcv_randomForest.best_estimator_
gcv_randomForest_score = gcv_randomForest.best_score_

index = 0
result_rf = list(randomForest_best.predict(normal_X))
# Confusion Matrix
print(confusion_matrix(normal_y, result_rf))
print(classification_report(normal_y, result_rf, target_names=['Attack', 'Normal']))

# Gradient Boosting
gcv_gradientBoosting = GridSearchCV(gradientBoosting, param_grid=gradient_params, scoring='accuracy', cv=cv, verbose=1)
gcv_gradientBoosting.fit(X_scaled, anomaly_y)
print("---------------------------------------------------------------")
print("Gradient Boosting")
print('final params', gcv_gradientBoosting.best_params_)  # print best parameter value
print('best score', gcv_gradientBoosting.best_score_)  # best score
gradientBoosting_best = gcv_gradientBoosting.best_estimator_
gcv_gradientBoosting_score = gcv_gradientBoosting.best_score_

index = 0
result_gb = list(gradientBoosting_best.predict(normal_X))
# Confusion Matrix
print(confusion_matrix(normal_y, result_gb))
print(classification_report(normal_y, result_gb, target_names=['Attack', 'Normal']))

# best model & score

score = [gcv_decisionTree_score, gcv_randomForest_score, gcv_gradientBoosting_score]
models = [decisionTree_best, randomForest_best, gradientBoosting_best]
max_score = gcv_logistic_score
best_model = logistic_best
for i in range(len(score)):
    if max_score < score[i]:
        max_score = score[i]
        best_model = models[i]

print("Anomaly detection best score: ", max_score)

result = list(best_model.predict(normal_X))
for i in range(len(result)):
    if result[i] != 'Normal':
        print('index', i, '---->', result[i])
        print(normal_X.iloc[i])

#check
# for i in range(len(normal_y)):
#     if normal_y[i] != 'Normal':
#         print('index', i, normal_y[i], '---->', result[i])
        # print(normal_X.iloc[i])

# Confusion Matrix
print(confusion_matrix(normal_y, result))
print(classification_report(normal_y, result, target_names=['Attack', 'Normal']))
