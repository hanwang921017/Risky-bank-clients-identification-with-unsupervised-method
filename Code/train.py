import pandas as pd
from sklearn import mixture
from sklearn.decomposition import PCA


df=pd.read_csv('result.csv')




data = df.values
# defalt n_components = 100, 100 features
pca=PCA(n_components=100)
newData=pca.fit_transform(data)
# split 80% train 20% test
train=newData[0:31723]
test=newData[31723:]

covariance_types = ["full", "tied", "diag", "spherical"]
result = {}
for k in range(1,5):
    for ct in covariance_types:
        gmm=mixture.GaussianMixture(n_components=k,covariance_type=ct,max_iter=1000).fit(train)
        avg_socre_train = gmm.score(train)
        avg_score_test = gmm.score(test)
        result[k] = { ct: {'train': avg_socre_train, 'test': avg_score_test} }

import json
result_string = json.dumps(result)
print(result_string)
f = open("result.txt","w")
f.write(result_string)
f.close()
