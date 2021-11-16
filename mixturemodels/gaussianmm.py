import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from os.path import dirname, join
from giffer import Gif


class GaussianMixtureModel:
    """ Gaussian Mixture Model """
    def __init__(self, k, data):
        self.k = k
        self.data = data
        self.pi = np.ones(self.k) / self.k
        self.classification = np.zeros((self.data.shape[0], self.k))
        self.indices = np.arange(self.data.shape[0])
        self.classification = self.classification[self.data]

    def _calc_gamma(self):
        return 


def main():
    np.random.seed(4)
    N = 50
    
    scatter = .01

    m1, m2 = np.random.uniform(0, 2, size=(2, 2))
    covariance = np.array([[scatter, 0], [0, scatter]])

    cls1 = np.random.multivariate_normal(m1, covariance, N)
    cls2 = np.random.multivariate_normal(m2, covariance, N)
    
    data = np.concatenate((cls1, cls2))

    classifier = GaussianMixtureModel(2, data)
    print(classifier.classification)

if __name__ == "__main__":
    main()