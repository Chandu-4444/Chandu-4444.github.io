import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import data
from sklearn import datasets
import matplotlib.pyplot as plt


def generate_data(n=100, n_features=1, noise=10):
    data = datasets.make_regression(n, n_features=n_features, noise=noise)
    plt.scatter(data[0], data[1])
    # plt.savefig("img/posts/linear-regression/generated_data.svg")
    plt.show()
    return data


def plot_line(X, y, intercept, slope):
    plt.scatter(X, y)
    plt.plot(X, slope * X + intercept, "r")
    plt.savefig("img/posts/linear-regression/generated_best_fit.svg")
    plt.show()


data = generate_data()

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(data[0], data[1])  # X = data[0], y = data[1]
plot_line(data[0], data[1], model.intercept_, model.coef_)

