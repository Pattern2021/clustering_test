import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import dirname, join
from giffer import Gif

path = dirname(__file__)
gif_name = join(path, "k_means_test.gif")
gif = Gif(path)

S = 20

def make_clusters(n_clusters, N):
    """ makes n clusters with N points to each cluster """
    sigma = np.array([[1, 0], [0, 1]])
    clusters = []
    for _ in range(n_clusters):
        mean = np.array([np.random.randint(-10, 10), np.random.randint(-10, 10)])
        cluster = np.random.multivariate_normal(mean, sigma, N)
        clusters.append(cluster)
    clusters = np.array(clusters).reshape(N * n_clusters, 2)
    return clusters

clusters = make_clusters(n_clusters=S, N=1000)

def k_means(data, k):
    epoch = 0

    # initial guesses (starting points)
    means = data[np.random.randint(data.shape[0], size=k)]
    run = True
    prev_cluster = 0
    while run:
        all_clusters = [[] for _ in range(k)]
        
        # iterate over all vectors
        for i, vecs in enumerate(data):
            dist_to_means = np.linalg.norm(means - vecs, axis=1)
            nearest_mean = np.argmin(dist_to_means)
            all_clusters[nearest_mean].append(i)
        
        if np.all(all_clusters == prev_cluster):
            run = False

        means = []
        for cluster in all_clusters:
            cluster_data = data[cluster]
            mean = np.mean(cluster_data, axis=0)
            means.append(mean)
        prev_cluster = all_clusters

        plt.clf()
        for cluster in all_clusters:
            plt.scatter(*data[cluster].T)
        # plt.pause(0.1)
        gif.frame()
        epoch += 1
    gif.save(gif_name)

k_means(clusters, k=S)
