import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN

data = pd.read_csv("iris.csv")
x = data.loc[:, "petal.length"]
y = data.loc[:, "petal.width"]
lab = data.loc[:, "variety"]
print(data)
print(data['variety'])
print(type(data))
plt.scatter(x[lab == "Setosa"], y[lab == "Setosa"], color='purple', label='Setosa')
plt.scatter(x[lab == "Verginica"], y[lab == "Verginica"], color='pink', label='Verginica')
plt.scatter(x[lab == "Versicolor"], y[lab == "Versicolor"], color='blue', label='Versicolor')
plt.scatter(2.5, 0.75, color='yellow', label='Setosa')
plt.legend()
plt.show()
d = list(zip(x, y))
model = KNN(n_neighbors=3)  # Utilisation du paramètre correct n_neighbors
model.fit(d, lab)

