import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

# database 
from pymongo import MongoClient
import os

# clustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

# classification
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA


import warnings
warnings.filterwarnings(action='ignore')

# bring data from mongoDB
def bringFromDB():
    # Associate with the MongoDB instance with 27017 ports on localhost
    client = MongoClient('localhost', 27017)

    # Create a database called livestock_information
    db = client.ioc_information  # database
    database = db.ioc_info  # collection
    dict_info = database.find()  # find data from mongoDB

    print(type(dict_info[1]))  # data type is dictionary

    # change dictionary to dataframe
    df_info = pd.DataFrame(dict_info)

    print(type(df_info))  # checking
    
    return df_info

# LabelEncoder
# *수정됨
encoder = LabelEncoder()

# Load the data
# *수정됨
# clustering data
test_file = 'all_file_separate_test.csv'
test_df = pd.read_csv(test_file)

##for cols in test_df.columns:
##    test_df[cols] = encoder.fit_transform(test_df[cols])
##print(test_df)

# *수정됨
# classificaiton data
# data from MongoDB
train_df = bringFromDB()

train_df.drop('_id', axis = 1)  #drop data id
# classification encoding
train_df['AttackType'] = encoder.fit_transform(train_df['AttackType'])
##for cols in train_df.columns:
##    train_df[cols] = encoder.fit_transform(train_df[cols])


# Preprocess the data
# *수정됨
# null값 처리 필요
#null값 확인
##print('----- Missing data -----')
##print(train_df.isnull().sum())
##print('Total : ',train_df.isnull().sum().sum())
##print()
##print(test_df.isnull().sum())
##print('Total : ',test_df.isnull().sum().sum())
##print()

# test X / train_X,y
# *수정됨
X_test = test_df.drop('AttackType', axis=1)
X_train = train_df.drop('AttackType', axis=1)
y_train = train_df['AttackType'].values

#check
print(X_train , X_test, sep = '\n')
print(X_train.columns , X_test.columns, sep = '\n')

#combo box를 통해서 select된 column들 list
selected_cols = ['Unnamed: 0', 'duration', 'protocol_type', 'service', 'flag',
       'src_bytes', 'dst_bytes', 'urgent', 'hot', 'num_failed_logins',
       'logged_in', 'num_compromised','dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
       'dst_host_serror_rate', 'dst_host_srv_serror_rate',
       'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

X_train = X_train.loc[:, selected_cols]
#이건 test를 위해서 이렇게 진행 나중에는 빼야 할 것.

X_test = X_test.loc[:, selected_cols]
print(X_train, X_test,sep='\n')
#check
##print(X_train.columns , X_test.columns, sep = '\n')
##print(y_train)

# Scale the data
# Scaling - MinMaxScaler, Normalizer, StandardScaler
MinMax_scaler = MinMaxScaler()
Standard_scaler = StandardScaler()
X_scaled = MinMax_scaler.fit_transform(X_train)
X_test_scaled = MinMax_scaler.fit_transform(X_test)

# model
# modle별 파라미터 정의 필요
logistic = LogisticRegression()  # 완료
l_model = logistic.fit(X_scaled, y_train)
print(l_model.predict(X_test_scaled))
kneighbors = KNeighborsClassifier()  # 완료

decisionTree = DecisionTreeClassifier()  # 완료
randomForest = RandomForestClassifier() # 완료
gradientBoosting = GradientBoostingClassifier()  # 완료
gaussianNB = GaussianNB()  # 완료

# ----------------------------------------------------------------------------
# parameters for GridSearch
decisionTree_params = {
    'max_depth': [None, 6, 8, 10, 12, 16, 20, 24],
    'min_samples_split': [2, 20, 50, 100, 200],
    'criterion': ['entropy', 'gini']
}
logistic_params = {
    'C': [0.1, 1.0, 10.0],
    'solver': ['liblinear', 'lbfgs', 'sag'],
    'max_iter': [50, 100, 200]
}
randomForest_params = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
KNN_params = {
    'n_neighbors': [3, 5, 11, 19],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
randomForest_params = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy']
}

gradient_params = {
    "n_estimators": range(50, 100, 25),
    "max_depth": [1, 2, 4],
    "learning_rate": [0.0001, 0.001, 0.01, 0.1],
    "subsample": [0.7, 0.9],
    "max_features": list(range(1, len(X_train.columns), 2))
}

# 0 부터 1 까지 0.01간격으로 parameter range 설정
param_range = []
NBarangeSet = np.arange(0, 1, 0.001)
for i in range(len(NBarangeSet)):
    param_range.append([NBarangeSet[i], 1 - NBarangeSet[i]])
gaussian_params = dict(priors=param_range)

# ----------------------------------------------------------------------------
##
# model = [logistic, kneighbors, decisionTree, randomForest, gradientBoosting, gaussianNB]
# clf = GridSearchCV(svc, parameters)
# best parameter 나오면 그걸로 test

# GridSearch
cv = KFold(n_splits=10, random_state=1)
# Logistic Regression
gcv_logistic = GridSearchCV(logistic, param_grid=logistic_params, scoring='accuracy', cv=cv, verbose=1)
gcv_logistic.fit(X_scaled, y_train)
print("---------------------------------------------------------------")
print("Logistic Regression")
print('final params', gcv_logistic.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv_logistic.best_score_) # 최고의 점수
logistic_best = gcv_logistic.best_estimator_

# KNN
gcv_kneighbors = GridSearchCV(kneighbors, param_grid=KNN_params,scoring='accuracy', cv=cv, verbose=1)
gcv_kneighbors.fit(X_scaled,y_train)
print("---------------------------------------------------------------")
print("KNN")
print('final params', gcv_kneighbors.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv_kneighbors.best_score_)      # 최고의 점수
knn_best = gcv_kneighbors.best_estimator_

# Decision Tree
gcv_decisionTree = GridSearchCV(decisionTree, param_grid=decisionTree_params, scoring='accuracy', cv=cv, verbose=1)
gcv_decisionTree.fit(X_scaled,y_train)
print("---------------------------------------------------------------")
print("Decision Tree")
print('final params', gcv_decisionTree.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv_decisionTree.best_score_)      # 최고의 점수
decisionTree_best = gcv_decisionTree.best_estimator_

# Random Forest
gcv_randomForest = GridSearchCV(randomForest, param_grid=randomForest_params, scoring='accuracy', cv=cv, verbose=1)
gcv_randomForest.fit(X_scaled,y_train)
print("---------------------------------------------------------------")
print("Random Forest")
print('final params', gcv_randomForest.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv_randomForest.best_score_)      # 최고의 점수
randomForest_best = gcv_randomForest.best_estimator_

# Gradient Boosting
gcv_gradientBoosting = GridSearchCV(gradientBoosting, param_grid=gradient_params, scoring='accuracy', cv=cv, verbose=1)
gcv_gradientBoosting.fit(X_scaled,y_train)
print("---------------------------------------------------------------")
print("Gradient Boosting")
print('final params', gcv_gradientBoosting.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv_gradientBoosting.best_score_)      # 최고의 점수
gradientBoosting_best = gcv_gradientBoosting.best_estimator_

# GaussianNB
gcv_gaussianNB = GridSearchCV(gaussianNB, param_grid=gaussian_params, scoring='accuracy', cv=cv, verbose=1)
gcv_gaussianNB.fit(X_scaled,y_train)
print("---------------------------------------------------------------")
print("GaussianNB")
print('final params', gcv_gaussianNB.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv_gaussianNB.best_score_)      # 최고의 점수
gaussianNB_best = gcv_gaussianNB.best_estimator_

# VotingClassifier
eclf2 = VotingClassifier(estimators=[('lr', logistic_best),('knn', knn_best),('dt', decisionTree_best),
                                      ('rf', randomForest_best),('gb', gradientBoosting_best),('gnb', gaussianNB_best)],voting='soft')

eclf2 = eclf2.fit(X_scaled,y_train)
print('voting score', eclf2.score(X_test_scaled,Y_test))
# 아마 voting이 가장 score 높게 나올거야
# ----------------------------------------------------------------------------

# cluster
outlier_index = []  # for save outlier index
# make outlier dataframe
outlier = {}
data_outlier = DataFrame(outlier)

# scaler에 따라 데이터가 많이 변경되기에 가장 좋은 모델을 찾기 위해 scaling을 minmax, standard로 모두 진행하고 비교할 것
for scaler in [MinMax_scaler, Standard_scaler]:
    print(scaler)
    X_scaled = scaler.fit_transform(X_test)
    # Normalize the data
    X_normalized = normalize(X_scaled)
    X_normalized = pd.DataFrame(X_normalized)
    # visualizable
    pca = PCA(n_components=2)
    X_principal = pca.fit_transform(X_normalized)
    X_principal = pd.DataFrame(X_principal)
    X_principal.columns = ['P1', 'P2']
    print(X_principal.head())
    print('-----------------------')

    # k is for distinguishing cluster model, kmean is unsuitable to detect outlier.
    for k in range(2):
        # dbscan
        if k == 0:
            print("DBSCAN")
            # For find best result
            for eps_D in [0.05, 0.1, 0.5]:
                for min_samples_D in [20, 50, 100]:
                    for algorithm_DB in ["kd_tree", "brute"]:
                        for metric_DB in [1, 2]:
                            print("eps", eps_D, "min_samples", min_samples_D, "algorithm", algorithm_DB, "p", metric_DB)
                            df_principal = pd.DataFrame(X_principal)
                            df_principal.columns = ['P1', 'P2']
                            db_default = DBSCAN(eps=eps_D, min_samples=min_samples_D, algorithm=algorithm_DB,
                                                p=metric_DB).fit(df_principal)

                            # if labels value is -1, this value is outlier
                            labels = db_default.labels_
                            j = 0
                            for i in labels:
                                # print("i=",i,"j=",j)
                                if i == -1:
                                    outlier_index.append(j)
                                j = j + 1
                            print(outlier_index)

                            # make outlier dataframe
                            if not outlier_index:
                                print("Not exist outlier_index")
                            else:
                                for i in outlier_index:
                                    data_outlier = data_outlier.append(test_df.loc[i])
                                print(data_outlier)
                                # best classification model
                                # *수정됨
##                                predict = eclf2.predict(data_outlier)


        else:
            print('-----------------------')
            print("EM")
            # For find best result
            for n_components_EM in [2, 3, 4, 5, 6]:
                for max_iter_EM in [50, 100, 200, 300]:
                    print("n_components", n_components_EM, "max_iter", max_iter_EM)
                    df_principal = pd.DataFrame(df_principal)
                    df_principal.columns = ['P1', 'P2']
                    gmm = GaussianMixture(n_components=n_components_EM, max_iter=max_iter_EM).fit(df_principal)
                    labels = gmm.predict(df_principal)
                    j = 0
                    for i in labels:
                        # print("i=",i,"j=",j)
                        if i == -1:
                            outlier_index.append(j)
                        j = j + 1
                    print(outlier_index)

                    # make outlier dataframe
                    if not outlier_index:
                        print("Not exist outlier_index")
                    else:
                        for i in outlier_index:
                            data_outlier = data_outlier.append(df.loc[i])
                        print(data_outlier)
                        # best classification model

            print('-----------------------')
