import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from os.path import dirname, join
from giffer import Gif

class Kmeans:
    """
    Kmeans object which tries to find k clusters on given unlabelled dataset.

    Parameters:
        data: array_like
            Unlabelled dataset, with shape (n, l) with n samples of l dimensions.
        k: int
            Amount of clusters the algorithm creates.
    Returns:
        out: array_like
            List containing classified samples in nested array
    """
    def __init__(self, data, k, fig, ax):
        self.data = data
        self.k = k
        self.fig = fig
        self.ax = ax
        self._find_clusters()

    def _find_clusters(self):
        """
        Method which finds clusters given dataset. Method is automatically called when initializing Kmeans object.
        """
        # indices of samples
        data_ind = np.arange(self.data.shape[0])

        # The initial random selected samples which are the first means.
        mean = np.random.choice(data_ind, size=self.k, replace=False)

        # finds which points are closest to each mean and stores their class label
        nearest_mean = np.argmin(cdist(self.data, self.data[mean]), axis=1)

        unchanged = False
        rotation = 120
        while not unchanged:
            prev_means = nearest_mean

            # separates clusters in separate lists.
            all_clusters = [self.data[nearest_mean == nearest] for nearest in np.unique(nearest_mean)]

            # mean for each cluster, this line is what does the "movement" of the clusters.
            mean = [np.mean(cluster_data, axis=0) for cluster_data in all_clusters]

            # finds distance from each datapoint to each mean.
            dist_to_means = cdist(self.data, mean)

            # set a new nearest mean, then iterate again.
            nearest_mean = np.argmin(dist_to_means, axis=1)
            
            # check for convergence
            if np.all(prev_means == nearest_mean):
                unchanged = True

            self.ax.cla()
            for cluster in all_clusters:
                self.ax.scatter(*cluster.T)

            for _ in range(5):
                rotation += 2
                self.ax.view_init(30, rotation)
                self.ax.set_title("Clusters")
                gif.frame(frame_duration=1)

        self.ax.set_title("Final clusters")
        gif.frame(frame_duration=10)
        gif.save(gifname)
        
        return all_clusters

def main():

    # specify path
    path = dirname(__file__)

    # make global due to this being just a test example
    global gifname, gif
    
    # name of gif with full path
    gifname = join(path, "kmeans_test.gif")
    
    # initialize Gif object
    gif = Gif(path)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.view_init(30, 120)

    # parameters for testing data
    N_means = 10
    N_samples = 500
    N_dim = 3
    spread = 2

    # create somewhat random means of different clusters
    means = [np.random.uniform(-10, 10, size=(N_dim,)) for _ in range(N_means)]
    # sigma = np.array([[spread, 0], [0, spread]])
    sigma = np.array([[spread, 0, 0], [0, spread, 0], [0, 0, spread]])

    # init with zeros array
    clusters = np.zeros((N_samples, N_dim))
    for mean in means:
        new_cluster = np.random.multivariate_normal(mean, sigma, N_samples)
        clusters = np.concatenate((clusters, new_cluster))

        # plot ground truth clusters for comparison
        ax.scatter(*new_cluster.T)
        ax.set_title("Ground truth")

    # add ground truth as a frame with duration 10.
    gif.frame(frame_duration=20)

    # Remove zeros array from clusters
    clusters = np.delete(clusters, np.arange(N_samples), axis=0)
    
    # classify by kmeans algorithm
    classifier = Kmeans(clusters, 7, fig, ax)

if __name__ == "__main__":
    main()