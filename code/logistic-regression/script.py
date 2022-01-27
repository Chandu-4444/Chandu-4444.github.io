from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import data
from sklearn import datasets
import matplotlib.pyplot as plt


def generate_data(n=100, n_features=1):
    data = datasets.make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2,
                                        random_state=1, n_clusters_per_class=1)
    # data = datasets.make_moons(n_samples=n, noise=0.1)
    plt.scatter(data[0][:, 0], data[0][:, 1], c=data[1])
    # plt.savefig("img/posts/linear-regression/generated_data.svg")
    plt.show()
    return data


def plot_line(X, y, intercept, slope):
    plt.scatter(X, y)
    plt.plot(X, slope * X + intercept, "r")
    plt.savefig("img/posts/linear-regression/generated_best_fit.svg")
    plt.show()


data = generate_data()

model = LinearRegression()
X = data[0]
y = data[1]
model.fit(X, y)
model.predict(X)
print(model.score(X, y))
