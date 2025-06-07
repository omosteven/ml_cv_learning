# good for unsupervised dataset
from minisom import MiniSom
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import numpy as np

iris = load_iris()
X = MinMaxScaler().fit_transform(iris.data)
som = MiniSom(x=7, y=7, input_len=4, sigma=1.0, learning_rate=0.5)
som.train_random(data=X, num_iteration=100)

win_map = som.win_map(X)
print('Min Max', X)
print("SOM clusters:", win_map.keys())
# print('SOM vals:', win_map.values())