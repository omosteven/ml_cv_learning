import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.svm as svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

cat_folder = "images/cats"
dog_folder = "images/dogs"
image_size = (64, 64)

X, y = [], []

def load_images(folder, label):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, image_size, cv2.INTER_NEAREST)
        X.append(img.flatten())
        y.append(label)

def scale_images():
    load_images(cat_folder, 0)
    load_images(dog_folder, 1)
    global X
    global y
    X = np.array(X)
    y = np.array(y)
    X = X / 255.0

def plot_images():
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(10,10))
    plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], color='blue', label ='CATS')
    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1,1], color='red', label='DOGS')
    plt.xlabel("Principal Component 1")
    plt.xlabel("Principal Component 2")
    plt.title("PCA Visualization of Cats vs Dogs")
    plt.legend()
    plt.show()

def apply_logistic_regression():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=32)
    # 32
    global model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Linear Classifier Accuracy: {accuracy:.4f}")
    print(y_pred)

def plot_boundary():
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    x_min, x_max = X_pca[:, 0].min() -1, X_pca[:, 0].max() +1
    y_min, y_max = X_pca[:, 1].min()-1, X_pca[:, 1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_original = pca.inverse_transform(grid_points)

    Z = model.predict(grid_points_original)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 10))
    plt.contour(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], color='blue', label='CATS')
    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], color='red', label='DOGS')
    plt.xlabel("Principal Component 1")
    plt.xlabel("Principal Component 2")
    plt.title("Decision Boundary of Linear Classifier")
    plt.legend()
    plt.show()

def classifer_images():
    scale_images()
    apply_logistic_regression()
    plot_boundary()
    # plot_images()
    # print('X:', X)
    # print('y:', y)

classifer_images()