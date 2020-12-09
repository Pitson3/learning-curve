import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_boundary(classifier, features, targets, title, x1_n_values=100,
                  x2_n_values=100):

    condition = features.shape[1] == 2
    if not condition:
        print(f"__n_features should be 2d to visualize")
        print(f"self.__n_features = {features.shape[1]}")
        return

    fig = plt.figure(figsize=(6, 5))

    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height])

    scale_factor = 0.25
    x1_start, x1_stop = features['x1'].min(), features['x1'].max()
    x1_start += scale_factor * x1_start
    x1_stop += scale_factor * x1_stop

    x2_start, x2_stop = features['x2'].min(), features['x2'].max()

    x2_start += scale_factor * x2_start
    x2_stop += scale_factor * x2_stop

    x1_vals = np.linspace(x1_start, x1_stop, x1_n_values)
    x2_vals = np.linspace(x2_start, x2_stop, x2_n_values)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    m = x1_n_values
    n = x2_n_values
    Z = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            x = np.array([X1[i, j], X2[i, j]])
            Z[i, j] = classifier.predict(np.array([x]))

    contour = plt.contour(X1, X2, Z, levels=[0])
    # plt.clabel(contour, colors='k', fmt='%2.1f', fontsize=12)

    ax.set_title('Boundary line')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')

    # add points
    plot_points(features, targets, title + " Descition boundary")
    plt.show()


def plot_points(features, targets, title):
    admitted = features[targets.to_numpy() == 1]
    rejected = features[targets.to_numpy() == -1]

    plt.scatter(admitted['x1'], admitted['x2'],
                s=25, color='cyan', edgecolor='k')
    plt.scatter(rejected['x1'], rejected['x2'],
                s=25, color='red', edgecolor='k')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)
