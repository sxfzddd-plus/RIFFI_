import sys
sys.path.append("./models")
from models.RIF import RIF_model, RTree
import numpy as np
import math


def EucliDist(A, B):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(A, B)]))


def unit_vector_from_c_p_to_x(c_p, x):
    if not np.array_equal(c_p, x):
        direction = x - c_p
        norm = np.linalg.norm(direction)
        return direction / norm
    else:
        return np.zeros_like(x)


def vector_from_c_p_to_x(c_p, x):
    if not np.array_equal(c_p, x):
        return x - c_p
    else:
        return np.zeros_like(x)


class RIFFI_tree(RTree):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.importances = []
        self.sum_normals = []

    def make_importance(self, X, depth_based):
        importance_list = []
        normal_vector_list = []

        for x in X:
            importance = np.zeros(len(x))
            sum_normal = np.zeros(len(x))
            node_id = 0
            c_p = self.nodes[node_id]["cp"]
            r = self.nodes[node_id]["r"]
            N = self.nodes[node_id]["numerosity"]
            depth = 0

            while c_p is not None:
                inside = EucliDist(x, c_p) < r
                prev_id = node_id
                direction_vector = vector_from_c_p_to_x(x, c_p)
                unit_vec = unit_vector_from_c_p_to_x(x, c_p)

                if inside:
                    node_id = self.left_son[node_id]
                    sum_normal += np.abs(unit_vec)

                    if depth_based:
                        factor = (1 - (self.nodes[node_id]["numerosity"] + 1) / N) / (1 + depth)
                        #singular_importance = direction_vector * factor
                        singular_importance = abs(direction_vector) * factor
                        importance += singular_importance
                        self.nodes[prev_id].setdefault("left_importance_depth", singular_importance)
                        self.nodes[prev_id].setdefault("depth", depth)
                    else:
                        factor = (1 - (self.nodes[node_id]["numerosity"] + 1) / N)
                        #singular_importance = direction_vector * factor
                        singular_importance = direction_vector * factor
                        importance += singular_importance
                        self.nodes[prev_id].setdefault("left_importance", singular_importance)
                        self.nodes[prev_id].setdefault("depth", depth)
                else:
                    node_id = self.right_son[node_id]
                    sum_normal += np.abs(unit_vec)

                    if depth_based:
                        factor = (1 - (self.nodes[node_id]["numerosity"] + 1) / N) / (1 + depth)
                        #singular_importance = direction_vector * factor
                        singular_importance = direction_vector * factor
                        importance += singular_importance
                        self.nodes[prev_id].setdefault("right_importance_depth", singular_importance)
                        self.nodes[prev_id].setdefault("depth", depth)
                    else:
                        factor = (1 - (self.nodes[node_id]["numerosity"] + 1) / N)
                        #singular_importance = direction_vector * factor
                        singular_importance = direction_vector * factor
                        importance += singular_importance
                        self.nodes[prev_id].setdefault("right_importance", singular_importance)
                        self.nodes[prev_id].setdefault("depth", depth)

                depth += 1
                c_p = self.nodes[node_id]["cp"]
                r = self.nodes[node_id]["r"]
                N = self.nodes[node_id]["numerosity"]

            importance_list.append(importance)
            normal_vector_list.append(sum_normal)

        return np.array(importance_list), np.array(normal_vector_list)


class RIFFI_original(RIF_model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sum_importances_matrix = None
        self.sum_normal_vectors_matrix = None
        self.plus = kwargs.get('plus')

    def fit(self, X):
        if not self.dims:
            self.dims = X.shape[1]
        if not self.min_sample:
            self.min_sample = 1
        if not self.max_depth:
            self.max_depth = np.inf

        self.forest = [
            RIFFI_tree(self.dims, self.min_sample, self.max_depth,
                       self.plus, self.l_mean_weight, self.r_mean_weight)
            for _ in range(self.n_trees)
        ]
        self.subsets = []

        for tree in self.forest:
            if not self.subsample_size or self.subsample_size > X.shape[0]:
                tree.make_tree(X, 0, 0, 'first')
            else:
                idx = np.random.choice(X.shape[0], self.subsample_size, replace=False)
                tree.make_tree(X[idx, :], 0, 0, 'first')
                self.subsets.append(idx)

    def Importances(self, X, calculate, overwrite, depth_based):
        if self.sum_importances_matrix is None or calculate:
            sum_importances = np.zeros_like(X, dtype='float64')
            sum_normals = np.zeros_like(X, dtype='float64')

            for tree in self.forest:
                importances, normals = tree.make_importance(X, depth_based)
                sum_importances += importances / self.n_trees
                sum_normals += normals / self.n_trees

            if overwrite:
                self.sum_importances_matrix = sum_importances
                self.sum_normal_vectors_matrix = sum_normals

            return sum_importances, sum_normals
        else:
            return self.sum_importances_matrix, self.sum_normal_vectors_matrix

    def Global_importance(self, X, calculate, overwrite, depth_based=False):
        anomaly_scores = self.Anomaly_Score(X)
        top_outliers = np.argpartition(anomaly_scores, -int(0.1 * len(X)))[-int(0.1 * len(X)):]

        importances, normals = self.Importances(X, calculate, overwrite, depth_based)

        outlier_mean_importance = np.mean(importances[top_outliers], axis=0)
        inlier_mean_importance = np.mean(np.delete(importances, top_outliers, axis=0), axis=0)

        return (outlier_mean_importance / len(X)) / inlier_mean_importance - 1

    def Local_importances(self, X, calculate, overwrite, depth_based=False):
        importances, _ = self.Importances(X, calculate, overwrite, depth_based)
        return importances
