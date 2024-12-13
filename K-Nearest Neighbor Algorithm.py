import numpy as np  # For numerical computations
import pandas as pd  # For handling datasets
import matplotlib.pyplot as plt  # For visualizations

data = pd.read_csv(r"D:\FSDS Material\Dataset\Non Linear emp_sal.csv")

# Extract independent variable (x) and dependent variable (y)
x = data.iloc[:, 1:2].values 
y = data.iloc[:, 2].values


from sklearn.neighbors import KNeighborsRegressor
knn_reg_model = KNeighborsRegressor(n_neighbors=2,weights='uniform',p=2)
knn_reg_model.fit(x,y)

knn_reg_pred = knn_reg_model.predict([[6.5]])
print(knn_reg_pred)