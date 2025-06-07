from sklearn.datasets import load_iris, make_blobs, load_wine, load_digits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def run_kmeans_with_elbow(X, dataset_name, max_k=10):
    inertias = []
    ks = range(1, max_k+1)
    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    plt.figure(figsize=(6,4))
    plt.plot(ks, inertias, 'bo-')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia')
    plt.title(f'Elbow Method for {dataset_name}')
    plt.grid(True)
    plt.show()

iris = load_iris().data
digits = load_digits().data
wine = load_wine().data
blobs, _ = make_blobs(n_samples=300,centers=3,n_features=3, random_state=42, )
run_kmeans_with_elbow(iris, 'Iris')
run_kmeans_with_elbow(digits, 'Digits')
run_kmeans_with_elbow(wine, 'Wine')
run_kmeans_with_elbow(blobs, 'Blobs')
# X = iris.data
# X, y = blobs
# kmeans = KMeans(n_clusters=4, random_state=42)
# kmeans.fit(X)
# labels = kmeans.labels_
# centroids = kmeans.cluster_centers_

# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)
# centroid_pca = pca.transform(centroids)
# print('Original:', X)
# print('Ori y:',y)
# print("Label", labels)
# print('Centroids', centroids)
# plt.figure(figsize=(6,10))
# plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, label='Data points', cmap='viridis')
# plt.scatter(centroid_pca[:, 0], centroid_pca[:,1], s=200, c='red', marker='X' ,label='Centroids')
# plt.xlabel('PCA 1')
# plt.ylabel('PCA 2')
# plt.title('Sample')
# plt.legend()
# plt.show()