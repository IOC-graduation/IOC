import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings(action='ignore')

# Load the data
# clustering data
df = pd.read_csv('C:/Users/samsung/Desktop/머신러닝/ML-week-10/mushrooms.csv') #데이터 변경(GUI로 파일 입력받기)
# classificaiton data


# LabelEncoder
# classification, clustering 모두 encoding 필요
# 현재 clustering만 encoding
encoder = LabelEncoder()
for cols in df.columns:
    df[cols] = encoder.fit_transform(df[cols])
print(df)
# classification encoding

# Drop the class column from the data
# 데이터 변경(공격(yes or no)이 target)
# clustering
clustering_X = df.drop('class', axis=1)
clustering_y = df['class'].values #target
# classification
# clustering_x column = classification_x column
classificaiton_X = df.drop('class', axis=1)
classificaiton_y = df['class'].values #target

# Preprocess the data
# null값 처리 필요
# 우선 단순히 mean값으로 채우기(고민)
clustering_X = clustering_X.fillna(df.mean())

# Scale the data
# Scaling - MinMaxScaler, Normalizer, StandardScaler
MinMax_scaler = MinMaxScaler()
Standard_scaler = StandardScaler()

# classification
# library
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# model
# modle별 파라미터 정의 필요
logistic = LogisticRegression().fit(classificaiton_X,classificaiton_y)
kneighbors = KNeighborsClassifier().fit(classificaiton_X,classificaiton_y)
decisionTree = DecisionTreeClassifier().fit(classificaiton_X,classificaiton_y)
randomForest = RandomForestClassifier().fit(classificaiton_X,classificaiton_y)
gradientBoosting = GradientBoostingClassifier().fit(classificaiton_X,classificaiton_y)
gaussianNB = GaussianNB().fit(classificaiton_X,classificaiton_y)

# model = [logistic, kneighbors, decisionTree, randomForest, gradientBoosting, gaussianNB]
# clf = GridSearchCV(svc, parameters)
# best parameter 나오면 그걸로 test

# cluster
outlier_index=[] #for save outlier index
# make outlier dataframe
outlier={}
data_outlier=DataFrame(outlier)


# scaler에 따라 데이터가 많이 변경되기에 가장 좋은 모델을 찾기 위해 scaling을 minmax, standard로 모두 진행하고 비교할 것
for scaler in[MinMax_scaler,Standard_scaler] :
    print(scaler)
    X_scaled = scaler.fit_transform(clustering_X)
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
    for k in range(2) :
        #dbscan
        if k == 0:
            print("DBSCAN")
            # For find best result
            for eps_D in [0.05, 0.1, 0.5]:
                for min_samples_D in [20, 50, 100]:
                    for algorithm_DB in ["kd_tree", "brute"]:
                        for metric_DB in [1, 2]:
                            print("eps",eps_D, "min_samples",min_samples_D, "algorithm",algorithm_DB, "p",metric_DB)
                            df_principal = pd.DataFrame(X_principal)
                            df_principal.columns = ['P1', 'P2']
                            db_default = DBSCAN(eps=eps_D, min_samples=min_samples_D, algorithm=algorithm_DB,
                                                p=metric_DB).fit(df_principal)

                            # if labels value is -1, this value is outlier
                            labels = db_default.labels_
                            j=0
                            for i in labels:
                                #print("i=",i,"j=",j)
                                if i == -1:
                                    outlier_index.append(j)
                                j=j+1
                            print(outlier_index)

                            # make outlier dataframe
                            if not outlier_index:
                                print("Not exist outlier_index")
                            else:
                                for i in outlier_index:
                                    data_outlier = data_outlier.append(df.loc[i])
                                print(data_outlier)
                                # best classification model


        else :
            print('-----------------------')
            print("EM")
            # For find best result
            for n_components_EM in [2, 3, 4, 5, 6]:
                for max_iter_EM in [50, 100, 200, 300]:
                    print("n_components",n_components_EM, "max_iter",max_iter_EM)
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
