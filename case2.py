# 2. 클래시피케이션 하고,
# 공격이라고 판별되지 않은 애들을 클러스터링 시켜서 아웃라이어가 있는지 보고 걔네도 의심하라고 출력 (우리들 생각)

import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

# -------------------------------------------------------------------------------------------------------------
# misuse detection
misuse_file = 'C:/Users/samsung/Desktop/Machine-Learning-Project-master/Dataset_Misuse_AttributeSelection.csv'
misuse_df = pd.read_csv(misuse_file)
print(misuse_df)

encoder = LabelEncoder()
misuse_df['AttackType'] = encoder.fit_transform(misuse_df['AttackType'])

misuse_x = misuse_df.drop('AttackType', axis=1)
misuse_y = misuse_df['AttackType'].values

# anomaly detection
anomaly_file = 'C:/Users/samsung/Desktop/Machine-Learning-Project-master/Dataset_Anomaly_AttributeSelection.csv'
anomaly_df = pd.read_csv(anomaly_file)
print(anomaly_df)

anomaly_df['AttackType'] = encoder.fit_transform(anomaly_df['AttackType'])

anomaly_x = anomaly_df.drop('AttackType', axis=1)
anomaly_y = anomaly_df['AttackType'].values

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
# Scaling - MinMaxScaler, Normalizer, StandardScaler
MinMax_scaler = MinMaxScaler()
Standard_scaler = StandardScaler()

print("Misuse detection")
X_scaled = MinMax_scaler.fit_transform(misuse_x)

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

# Decision Tree
gcv_decisionTree = GridSearchCV(decisionTree, param_grid=decisionTree_params, scoring='accuracy', cv=cv, verbose=1)
gcv_decisionTree.fit(X_scaled,misuse_y)
print("---------------------------------------------------------------")
print("Decision Tree")
print('final params', gcv_decisionTree.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv_decisionTree.best_score_)      # 최고의 점수
decisionTree_best = gcv_decisionTree.best_estimator_

# Random Forest
gcv_randomForest = GridSearchCV(randomForest, param_grid=randomForest_params, scoring='accuracy', cv=cv, verbose=1)
gcv_randomForest.fit(X_scaled,misuse_y)
print("---------------------------------------------------------------")
print("Random Forest")
print('final params', gcv_randomForest.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv_randomForest.best_score_)      # 최고의 점수
randomForest_best = gcv_randomForest.best_estimator_

# Gradient Boosting
gcv_gradientBoosting = GridSearchCV(gradientBoosting, param_grid=gradient_params, scoring='accuracy', cv=cv, verbose=1)
gcv_gradientBoosting.fit(X_scaled,misuse_y)
print("---------------------------------------------------------------")
print("Gradient Boosting")
print('final params', gcv_gradientBoosting.best_params_)   # 최적의 파라미터 값 출력
print('best score', gcv_gradientBoosting.best_score_)      # 최고의 점수
gradientBoosting_best = gcv_gradientBoosting.best_estimator_

# --------------------------------------
# purity가 가장 높게 나왔을 상황의 outlier를 보여주는게 어떤지
# clustering이 마지막 단계니 charty랑 purity 보여주는거 추가
# anomaly detection
# clustering
outlier_index = []  # for save outlier index
# make outlier dataframe
outlier = {}
data_outlier = DataFrame(outlier)

# test set에서 만든 normal 데이터 x,y 만들기

# scaler에 따라 데이터가 많이 변경되기에 가장 좋은 모델을 찾기 위해 scaling을 minmax, standard로 모두 진행하고 비교할 것
for scaler in [MinMax_scaler, Standard_scaler]:
    print(scaler)
    X_scaled = scaler.fit_transform(X_test) # test set에서 만든 normal 데이터 x로 변경
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
                                    data_outlier = data_outlier.append(df.loc[i]) # test set에서 만든 normal 데이터로 변경
                                print(data_outlier)


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
                            data_outlier = data_outlier.append(df.loc[i]) # test set에서 만든 normal 데이터로 변경
                        print(data_outlier)

            print('-----------------------')
