import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import time
import seaborn as sns
import pandas as pd
import numpy as np

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = [
    LinearRegression(),
    Ridge(alpha=1.0),
    Lasso(alpha=0.1),
    DecisionTreeRegressor(max_depth=4),
    RandomForestRegressor(n_estimators=100),
    SVR()
]

model_names = ['Linear Regression','Ridge', 'Lasso', 'Decision Tree', 'Random Forest', 'SVR']
# lr = LinearRegression()
for i,model in enumerate(models):
    start_time = time.time()
    model_name = model_names[i]
    if model_name == 'SVR':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    total_time = time.time() - start_time
    print(f"Model Name: {model_name} MSE: {mse}, R2 Score: {r2}, Time: {total_time}")

# ---check for dataset linearity
X = pd.DataFrame(X, columns=diabetes.feature_names)
correlations = X.corrwith(pd.Series(y))
print('Correlation:', correlations.sort_values(ascending=False)) #closer to +1 or -1 means strong linearity

for col in X.columns:
    sns.scatterplot(x=X[col], y=y)
    plt.title(f"{col} vs  Target")
    plt.show()