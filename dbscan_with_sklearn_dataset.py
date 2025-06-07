from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris, load_digits, load_wine, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
X_scaled = StandardScaler().fit_transform(X)
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(6,5))
plt.scatter(X_pca[:,0], X_pca[:,1], cmap='tab10', c=labels, label='DB Clusters', s=20)
plt.title('DB Scan Clustering')
plt.legend()
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()
print('Original:', X)
print('Scaled:', X_scaled)
print('Labels', labels)

