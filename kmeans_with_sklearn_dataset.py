from sklearn.datasets import load_iris, make_blobs, load_wine, load_digits
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
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

    k=3 if dataset_name != 'Digits' else 10
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    centroids = pca.transform(kmeans.cluster_centers_)
    metric = silhouette_score(X, labels, metric='euclidean', random_state=42)
    print(f'Sihouette Score for {dataset_name} is {metric}')
    plt.figure(figsize=(6,5))
    plt.scatter(X_pca[:,0], X_pca[:,1], cmap='tab10', c=labels, label=f'{dataset_name} points', )
    plt.scatter(centroids[:,0], centroids[:,1], label='Centroids', c='red', marker='X', s=200)
    plt.title(f'{dataset_name} Kmeans Clustering')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    plt.show()

iris = load_iris().data
digits = load_digits().data
wine = load_wine().data
blobs, _ = make_blobs(n_samples=300,centers=3,n_features=3, random_state=42,)

run_kmeans_with_elbow(iris, 'Iris')
run_kmeans_with_elbow(digits, 'Digits')
run_kmeans_with_elbow(wine, 'Wine')
run_kmeans_with_elbow(blobs, 'Blobs')