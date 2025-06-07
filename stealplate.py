from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

iris = load_iris()
X = iris.data

kmeans = KMeans(n_clusters=3, random_state=43)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
plt.figure(figsize=(8,6))
plt.scatter(X_reduced[:, 0], X_reduced[:,1], c=labels, cmap='viridis', label='Data point')
plt.scatter(pca.transform(centroids)[:,0], pca.transform(centroids)[:,1], s=200, c='red', marker='X', label='Centroids')
plt.title("K Means Clustering on Iris Dataset(PCA-reduced)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.show()
print('actual data:', X)
# X_reduced = [ [1,2,3], [4,5,6], [7,8,9] ]
# X_reduced = [[-2.68412563 , 0.31939725],
#  [-2.71414169, -0.17700123],
#  [-2.88899057 ,-0.14494943],
#  [-2.74534286 ,-0.31829898]]
print('centroids:', centroids)
print('pca centr:', pca.transform(centroids))
print('pca data:', X_reduced)
print('pca data1:', X_reduced[:,0])
print('pca data2:', X_reduced[:,1])

# print('labels:',labels)
