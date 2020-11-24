# 1. 클래시피케이션 두번하는거(교수님 생각)

import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Scaling - MinMaxScaler, Normalizer, StandardScaler
MinMax_scaler = MinMaxScaler()
Standard_scaler = StandardScaler()

# -------------------------------------------------------------------------------------------------------------
# misuse detection
misuse_file = 'Dataset_Misuse_AttributeSelection.csv'
misuse_df = pd.read_csv(misuse_file)
print(misuse_df)
misuse_df = misuse_df.fillna('Normal')

encoder = LabelEncoder()
#misuse_df['AttackType'] = encoder.fit_transform(misuse_df['AttackType'])

misuse_x = misuse_df.drop('AttackType', axis=1)
misuse_y = misuse_df['AttackType'].values

# anomaly detection
anomaly_file = 'Dataset_Anomaly_AttributeSelection.csv'
anomaly_df = pd.read_csv(anomaly_file)
print(anomaly_df)

#anomaly_df['AttackType'] = encoder.fit_transform(anomaly_df['AttackType'])

anomaly_x = anomaly_df.drop('AttackType', axis=1)
anomaly_y = anomaly_df['AttackType'].values

# test data
test_file = 'testData.csv'
test_df = pd.read_csv(test_file)
test_df = test_df.drop('Unnamed: 0',axis = 1)
#encoding
test_df['protocol_type'] = encoder.fit_transform(test_df['protocol_type'])
test_df['service'] = encoder.fit_transform(test_df['service'])
test_df['flag'] = encoder.fit_transform(test_df['flag'])

# Scale the data
print("Test data")
test_X = MinMax_scaler.fit_transform(test_df)

# -------------------------------------------------------------------------------------------------------------
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

value = misuse_x.columns
gradient_params = {
    "n_estimators": range(50, 100, 25),
    "max_depth": [1, 2, 4],
    "learning_rate": [0.0001, 0.001, 0.01, 0.1],
    "subsample": [0.7, 0.9],
    "max_features": list(range(1, len(misuse_x.columns), 2))
}

# Scale the data
print("Misuse detection")
X_scaled = MinMax_scaler.fit_transform(misuse_x)

#null값 확인
print('----- Missing data -----')
print(misuse_df.isnull().sum())
print('Total : ',misuse_df.isnull().sum().sum())
print()
print(anomaly_df.isnull().sum())
print('Total : ',anomaly_df.isnull().sum().sum())
print()

# misuse dectection model train
logistic = LogisticRegression().fit(X_scaled, misuse_y)
decisionTree = DecisionTreeClassifier().fit(X_scaled, misuse_y)
randomForest = RandomForestClassifier().fit(X_scaled, misuse_y)
gradientBoosting = GradientBoostingClassifier().fit(X_scaled, misuse_y)

# GridSearch
cv = KFold(n_splits=10, random_state=1)
# Logistic Regression
gcv_logistic = GridSearchCV(logistic, param_grid=logistic_params, scoring='accuracy', cv=cv, verbose=1)
gcv_logistic.fit(X_scaled,misuse_y)
print("---------------------------------------------------------------")
print("Logistic Regression")
print('final params', gcv_logistic.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv_logistic.best_score_) # 최고의 점수
logistic_best = gcv_logistic.best_estimator_
logistic_predict = ogistic_best.predict(test_X)
print(logistic_predict)

# Decision Tree
gcv_decisionTree = GridSearchCV(decisionTree, param_grid=decisionTree_params, scoring='accuracy', cv=cv, verbose=1)
gcv_decisionTree.fit(X_scaled,misuse_y)
print("---------------------------------------------------------------")
print("Decision Tree")
print('final params', gcv_decisionTree.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv_decisionTree.best_score_)      # 최고의 점수
decisionTree_best = gcv_decisionTree.best_estimator_
decisionTree_predict = decisionTree_best.predict(test_X)
print(decisionTree_predict)

# Random Forest
gcv_randomForest = GridSearchCV(randomForest, param_grid=randomForest_params, scoring='accuracy', cv=cv, verbose=1)
gcv_randomForest.fit(X_scaled,misuse_y)
print("---------------------------------------------------------------")
print("Random Forest")
print('final params', gcv_randomForest.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv_randomForest.best_score_)      # 최고의 점수
randomForest_best = gcv_randomForest.best_estimator_
randomForest_predict = randomForest_best.predict(test_X)
print(randomForest_predict)

##
### Gradient Boosting
##gcv_gradientBoosting = GridSearchCV(gradientBoosting, param_grid=gradient_params, scoring='accuracy', cv=cv, verbose=1)
##gcv_gradientBoosting.fit(X_scaled,misuse_y)
##print("---------------------------------------------------------------")
##print("Gradient Boosting")
##print('final params', gcv_gradientBoosting.best_params_)   # 최적의 파라미터 값 출력
##print('best score', gcv_gradientBoosting.best_score_)      # 최고의 점수
##gradientBoosting_best = gcv_gradientBoosting.best_estimator_
##gradientBoosting_predict = gradientBoosting_best.predict(test_X)
##print(gradientBoosting_predict)

# --------------------------------------
print("Anomaly detection")
value = anomaly_x.columns

X_scaled = MinMax_scaler.fit_transform(anomaly_x)

# misuse dectection model train
logistic = LogisticRegression().fit(X_scaled, misuse_y)
decisionTree = DecisionTreeClassifier().fit(X_scaled, misuse_y)
randomForest = RandomForestClassifier().fit(X_scaled, misuse_y)
gradientBoosting = GradientBoostingClassifier().fit(X_scaled, misuse_y)
logistic_predict = ogistic_best.predict(test_X)
print(logistic_predict)

# GridSearch
cv = KFold(n_splits=10, random_state=1)
# Logistic Regression
gcv_logistic = GridSearchCV(logistic, param_grid=logistic_params, scoring='accuracy', cv=cv, verbose=1)
gcv_logistic.fit(X_scaled,anomaly_y)
print("---------------------------------------------------------------")
print("Logistic Regression")
print('final params', gcv_logistic.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv_logistic.best_score_) # 최고의 점수
logistic_best = gcv_logistic.best_estimator_

# Decision Tree
gcv_decisionTree = GridSearchCV(decisionTree, param_grid=decisionTree_params, scoring='accuracy', cv=cv, verbose=1)
gcv_decisionTree.fit(X_scaled,anomaly_y)
print("---------------------------------------------------------------")
print("Decision Tree")
print('final params', gcv_decisionTree.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv_decisionTree.best_score_)      # 최고의 점수
decisionTree_best = gcv_decisionTree.best_estimator_
decisionTree_predict = decisionTree_best.predict(test_X)
print(decisionTree_predict)

# Random Forest
gcv_randomForest = GridSearchCV(randomForest, param_grid=randomForest_params, scoring='accuracy', cv=cv, verbose=1)
gcv_randomForest.fit(X_scaled,anomaly_y)
print("---------------------------------------------------------------")
print("Random Forest")
print('final params', gcv_randomForest.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv_randomForest.best_score_)      # 최고의 점수
randomForest_best = gcv_randomForest.best_estimator_
randomForest_predict = randomForest_best.predict(test_X)
print(randomForest_predict)

##
### Gradient Boosting
##gcv_gradientBoosting = GridSearchCV(gradientBoosting, param_grid=gradient_params, scoring='accuracy', cv=cv, verbose=1)
##gcv_gradientBoosting.fit(X_scaled,anomaly_y)
##print("---------------------------------------------------------------")
##print("Gradient Boosting")
##print('final params', gcv_gradientBoosting.best_params_)   # 최적의 파라미터 값 출력
##print('best score', gcv_gradientBoosting.best_score_)      # 최고의 점수
##gradientBoosting_best = gcv_gradientBoosting.best_estimator_
##gradientBoosting_predict = gradientBoosting_best.predict(test_X)
##print(gradientBoosting_predict)
