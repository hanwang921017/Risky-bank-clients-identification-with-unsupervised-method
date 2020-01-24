import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import mixture
from sklearn.cluster import Birch

from sklearn.metrics import davies_bouldin_score


X = np.load('after_PCA_data.npy')

# kmeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# AffinityPropagation
clustering1 = AffinityPropagation().fit(X)

# MeanShift
clustering2 = MeanShift(bandwidth=2).fit(X)

# SpectralClustering
clustering3 = SpectralClustering(n_clusters=2,assign_labels="discretize",random_state=0).fit(X)

# AgglomerativeClustering
clustering4 = AgglomerativeClustering().fit(X)

# DBSCAN
clustering5= DBSCAN().fit(X)

# Birch
brc = Birch(branching_factor=50, n_clusters=2, threshold=0.5,compute_labels=True).fit(X) 

kmeans_labels = kmeans.labels_
AffinityPropagation_labels = clustering1.labels_
MeanShift_labels = clustering2.labels_
SpectralClustering_labels = clustering3.labels_
AgglomerativeClustering_labels = clustering4.labels_
DBSCAN_labels = clustering5.labels_
Birch_labels = brc.labels_

print(Birch_labels)

#davies_bouldin_score(X, labels) 
kmeans_sc = davies_bouldin_score(X, kmeans_labels) 
AffinityPropagation_sc = davies_bouldin_score(X, AffinityPropagation_labels) 
# MeanShift_sc = davies_bouldin_score(X, MeanShift_labels)
SpectralClustering_sc = davies_bouldin_score(X, SpectralClustering_labels)
AgglomerativeClustering_sc = davies_bouldin_score(X, AgglomerativeClustering_labels)
DBSCAN_sc = davies_bouldin_score(X, DBSCAN_labels)
Birch_sc = davies_bouldin_score(X, Birch_labels)

print("kmeans",Birch_sc)