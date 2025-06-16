import numpy as np
import math
import sys
sys.path.append('../')
from utils_.utils_ import make_rand_vector, c_factor




def eucliDist(A, B):
    return np.sqrt(np.sum((A - B) ** 2, axis=1))

def EucliDist(A, B):
    return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)]))


class RIF_model:
    def __init__(self, n_trees, l_mean_weight, r_mean_weight, max_depth=None, min_sample=None, dims=None, subsample_size=None, plus=1):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_sample = min_sample
        self.dims = dims
        self.subsample_size = subsample_size
        self.forest = None
        self.plus = plus
        self.l_mean_weight = l_mean_weight
        self.r_mean_weight = r_mean_weight

    def fit(self, X):
        if not self.dims:
            self.dims = X.shape[1]
        if not self.min_sample:
            self.min_sample = 1
        if not self.max_depth:
            self.max_depth = int(np.ceil(np.log2(self.subsample_size)))

        self.forest = [RTree(self.dims, self.min_sample, self.max_depth, self.plus, self.l_mean_weight, self.r_mean_weight) for i in range(self.n_trees)]
        for x in self.forest:
            if not self.subsample_size:
                x.make_tree(X, 0, 0, 'first')
            else:
                X_sub = X[np.random.choice(X.shape[0], self.subsample_size, replace=False), :]
                x.make_tree(X_sub, 0, 0, 'first')

    def Anomaly_Score(self, X, algorithm=1):
        mean_path = np.zeros(len(X))
        if algorithm == 1:
            for i in self.forest:
                mean_path += i.compute_paths(X)
        elif algorithm == 0:
            for i in self.forest:
                mean_path += i.compute_paths2(X, 0)

        mean_path = mean_path / len(self.forest)
        c = c_factor(len(X))
        return 2 ** (-mean_path / c)

    def _predict(self, X, p):
        An_score = self.Anomaly_Score(X)
        y_hat = An_score > sorted(An_score, reverse=True)[int(p * len(An_score))]
        return y_hat

    def evaluate(self, X, y, p):
        An_score = self.Anomaly_Score(X)
        m = np.c_[An_score, y]
        m = m[(-m[:, 0]).argsort()]
        return np.sum(m[:int(p * len(X)), 1]) / int(p * len(X))


class RTree:
    def __init__(self, dims, min_sample, max_depth, plus, l_mean_weight, r_mean_weight):
        self.dims = dims
        self.min_sample = min_sample
        self.max_depth = max_depth
        self.depth = 0
        self.right_son = [0]
        self.left_son = [0]
        self.nodes = {}
        self.plus = plus
        self.l_mean_weight = l_mean_weight
        self.r_mean_weight = r_mean_weight

    def make_tree(self, X, id, depth, type):
        if X.shape[0] <= self.min_sample or depth >= self.max_depth:
            self.nodes[id] = {"cp": None, "r": None, "type": type, "numerosity": len(X)}
        else:
            c_p = X[np.random.choice(len(X))]
            dis_X = np.abs(X - c_p)
            mins = dis_X.min(axis=0)
            maxs = dis_X.max(axis=0)
            o = np.zeros((X.shape[1], 1))
            r_max = np.linalg.norm(maxs - o)
            r_min = np.linalg.norm(mins - o)
            if type == 'first' or type == 'right':
                r = ((r_max + r_min) / 2) * self.r_mean_weight
            elif type == 'left':
                r = ((r_max + r_min) / 2) * self.l_mean_weight

            dis = np.linalg.norm(X - c_p, axis=1)
            lefts = dis < r

            self.nodes[id] = {"cp": c_p, "r": r, "type": type, "numerosity": len(X)}

            idsx = len(self.nodes)
            self.left_son[id] = int(idsx)
            self.right_son.append(0)
            self.left_son.append(0)
            self.make_tree(X[lefts], idsx, depth + 1, type='left')

            iddx = len(self.nodes)
            self.right_son.append(0)
            self.left_son.append(0)
            self.right_son[id] = int(iddx)
            self.make_tree(X[~lefts], iddx, depth + 1, type='right')

    def compute_paths2(self, X, id, true_vec=None):
        if id == 0:
            true_vec = np.ones(len(X))
        c_p = self.nodes[id]["cp"]
        r = self.nodes[id]["r"]

        if c_p is None:
            return true_vec * 1
        else:
            val = np.array(np.linalg.norm(X[true_vec == 1], c_p, axis=1) < r)
            lefts = true_vec.copy()
            rights = true_vec.copy()
            lefts[true_vec == 1] = val
            rights[true_vec == 1] = np.logical_not(val)
            return true_vec * 1 + self.compute_paths2(X, int(self.left_son[id]), true_vec=lefts) + self.compute_paths2(X, int(self.right_son[id]), true_vec=rights)

    def compute_paths(self, X):
        paths = []
        for x in X:
            id = 0
            k = 1
            c_p = self.nodes[id]["cp"]
            r = self.nodes[id]["r"]
            while c_p is not None:
                val = np.linalg.norm(x - c_p) < r
                if val:
                    id = self.left_son[id]
                else:
                    id = self.right_son[id]
                c_p = self.nodes[id]["cp"]
                r = self.nodes[id]["r"]
                k += 1
            paths.append(k)
        return paths