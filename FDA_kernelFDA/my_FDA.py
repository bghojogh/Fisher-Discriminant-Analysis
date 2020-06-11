import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv
from sklearn.metrics.pairwise import pairwise_kernels

# ----- python fast kernel matrix:
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_kernels.html
# https://stackoverflow.com/questions/7391779/fast-kernel-matrix-computation-python
# https://stats.stackexchange.com/questions/15798/how-to-calculate-a-gaussian-kernel-effectively-in-numpy
# https://stackoverflow.com/questions/36324561/fast-way-to-calculate-kernel-matrix-python?rq=1

# ----- python fast scatter matrix:
# https://stackoverflow.com/questions/31145918/fast-weighted-scatter-matrix-calculation

class My_FDA:

    def __init__(self, n_components=None, kernel=None):
        self.n_components = n_components
        self.U = None
        self.X_train = None
        if kernel is not None:
            self.kernel = kernel
        else:
            self.kernel = 'linear'

    def fit_transform(self, X, y):
        # X: columns are sample, rows are features
        self.fit(X=X, y=y)
        X_transformed = self.transform(X=X, y=y)
        return X_transformed

    def fit(self, X, y):
        # X: columns are sample, rows are features
        self.X_train = X
        # ------ Separate classes:
        X_separated_classes = self._separate_samples_of_classes(X=X, y=y)
        y = np.asarray(y)
        y = y.reshape((1, -1))
        n_samples = X.shape[1]
        n_dimensions = X.shape[0]
        labels_of_classes = list(set(y.ravel()))
        n_classes = len(labels_of_classes)
        # ------ S_B:
        mean_of_total = X.mean(axis=1)
        mean_of_total = mean_of_total.reshape((-1, 1))
        S_B = np.zeros((n_dimensions, n_dimensions))
        for class_index in range(n_classes):
            X_class = X_separated_classes[class_index]
            n_samples_of_class = X_class.shape[1]
            mean_of_class = X_class.mean(axis=1)
            mean_of_class = mean_of_class.reshape((-1, 1))
            temp = mean_of_class - mean_of_total
            S_B = S_B + (n_samples_of_class * temp.dot(temp.T))
        # ------ M_c and M:
        S_W = np.zeros((n_dimensions, n_dimensions))
        for class_index in range(n_classes):
            X_class = X_separated_classes[class_index]
            n_samples_of_class = X_class.shape[1]
            mean_of_class = X_class.mean(axis=1)
            mean_of_class = mean_of_class.reshape((-1, 1))
            for sample_index in range(n_samples_of_class):
                sample_of_class = X_class[:, sample_index]
                sample_of_class = sample_of_class.reshape((-1, 1))
                temp = sample_of_class - mean_of_class
                S_W = S_W + temp.dot(temp.T)
        # ------ Fisher directions:
        epsilon = 0.00001  #--> to prevent singularity of matrix N
        eig_val, eig_vec = LA.eigh(inv(S_W + epsilon*np.eye(S_W.shape[0])).dot(S_B))
        idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]
        if self.n_components is not None:
            U = eig_vec[:, :self.n_components]
        else:
            U = eig_vec[:, :n_classes-1]
        self.U = U

    def transform(self, X, y):
        # X: columns are sample, rows are features
        # X_transformed: columns are sample, rows are features
        X_transformed = (self.U.T).dot(X)
        return X_transformed

    def get_projection_directions(self):
        return self.U

    def reconstruct(self, X, scaler=None, using_howMany_projection_directions=None):
        # X: rows are features and columns are samples
        if using_howMany_projection_directions != None:
            U = self.U[:, 0:using_howMany_projection_directions]
        else:
            U = self.U
        X_transformed = (U.T).dot(X)
        X_reconstructed = U.dot(X_transformed)
        return X_reconstructed

    def transform_outOfSample_all_together(self, X, using_howMany_projection_directions=None):
        # X: rows are features and columns are samples
        X_transformed = (self.U.T).dot(X)
        return X_transformed

    def _separate_samples_of_classes(self, X, y):
        # X --> rows: features, columns: samples
        # X_separated_classes --> rows: features, columns: samples
        X = X.T
        y = np.asarray(y)
        y = y.reshape((-1, 1))
        yX = np.column_stack((y, X))
        yX = yX[yX[:, 0].argsort()]  # sort array (asscending) with regards to nth column --> https://gist.github.com/stevenvo/e3dad127598842459b68
        y = yX[:, 0]
        X = yX[:, 1:]
        labels_of_classes = list(set(y))
        number_of_classes = len(labels_of_classes)
        dimension_of_data = X.shape[1]
        X_separated_classes = [np.empty((0, dimension_of_data))] * number_of_classes
        class_index = 0
        index_start_new_class = 0
        n_samples = X.shape[0]
        for sample_index in range(1, n_samples):
            if y[sample_index] != y[sample_index - 1] or sample_index == n_samples-1:
                X_separated_classes[class_index] = np.vstack([X_separated_classes[class_index], X[index_start_new_class:sample_index, :]])
                index_start_new_class = sample_index
                class_index = class_index + 1
        for class_index in range(number_of_classes):
            X_class = X_separated_classes[class_index]
            X_separated_classes[class_index] = X_class.T
        return X_separated_classes