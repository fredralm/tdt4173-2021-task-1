import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:
    
    def __init__(self, n_clusters):
        # NOTE: Feel free add any hyperparameters 
        # (with defaults) as you see fit
        self.n_clusters = n_clusters
        
    def fit(self, X):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """

        if isinstance(X, np.ndarray):
            X_array = X
        else:
            X_array = X.to_numpy()
        m_samples, n_features = X_array.shape
        # Normalize data
        for i in range(n_features):
            X_min = np.amin(X_array[:,i])
            X_max = np.amax(X_array[:,i])
            for j in range(m_samples):
                X_array[j][i] = (X_array[j][i] - X_min) / (X_max - X_min)

        # Run algorithm with different initializations and choose lowest distortion
        min_dist = np.inf
        for i in range(100):
            # Initialize random centroids
            centroids = X_array[np.random.choice(len(X_array), self.n_clusters, replace=False)]

            # Iteratively update centroids until no more changes
            assignments = np.zeros(m_samples, dtype=int)
            prev_assignments = np.ones(m_samples, dtype=int)
            while assignments.all() != prev_assignments.all():
                prev_assignments = assignments
                euclidean_distances = cross_euclidean_distance(X_array, centroids)
                distances_squared = euclidean_distances*euclidean_distances
                # Assign points
                for i, d in enumerate(distances_squared):
                    assignments[i] = np.argmin(d)
                # Update centroids
                centroids = np.zeros((self.n_clusters, n_features))
                for i in range(self.n_clusters):
                    point_count = 0
                    for j in range(m_samples):
                        if assignments[j] == i:
                            centroids[i] += X_array[j]
                            point_count += 1
                    if point_count != 0:
                        centroids[i] = centroids[i]/point_count
            dist = euclidean_distortion(X, assignments)
            if min_dist > dist:
                min_dist = dist
                self.centroids = centroids
        
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        if isinstance(X, np.ndarray):
            X_array = X
        else:
            X_array = X.to_numpy()
        m_samples, n_features = X_array.shape
        # Normalize data
        for i in range(n_features):
            X_min = np.amin(X_array[:,i])
            X_max = np.amax(X_array[:,i])
            for j in range(m_samples):
                X_array[j][i] = (X_array[j][i] - X_min) / (X_max - X_min)
        euclidean_distances = cross_euclidean_distance(X_array, self.centroids)
        distances_squared = euclidean_distances*euclidean_distances
        assignments = np.zeros(X_array.shape[0], dtype=int)
        # Assign points
        for i, d in enumerate(distances_squared):
            assignments[i] = np.argmin(d)

        return assignments
    
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return self.centroids
    
    
    
    
# --- Some utility functions 


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    for c in np.unique(z):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += ((Xc - mu) ** 2).sum()
        
    return distortion

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    Compute Euclidean distance between two sets of points 
    
    Args:
        x (array<m,d>): float tensor with pairs of 
            n-dimensional points. 
        y (array<n,d>): float tensor with pairs of 
            n-dimensional points. Uses y=x if y is not given.
            
    Returns:
        A float array of shape <m,n> with the euclidean distances
        from all the points in x to all the points in y
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))
