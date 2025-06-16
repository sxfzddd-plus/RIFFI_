import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import StratifiedShuffleSplit as SSS
from numba import jit
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


@jit(nopython=True)
def c_factor(n):
    """
    Average path length of unsuccessful search in a binary search tree given n points
    """
    return 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1.) / (n * 1.0))


def make_rand_vector(degrees_of_freedom, dimensions):
    """
    Generate a random unit vector with specified degrees of freedom.
    """
    if dimensions < degrees_of_freedom:
        raise ValueError("Degrees of freedom exceed data dimensions")

    random_values = np.random.normal(loc=0.0, scale=1.0, size=degrees_of_freedom)
    indices = np.random.choice(range(dimensions), degrees_of_freedom, replace=False)
    vec = np.zeros(dimensions)
    vec[indices] = random_values
    return vec / np.linalg.norm(vec)


def mean_confidence_interval_importances(importances, confidence=0.95):
    """
    Calculate mean and confidence interval of importance scores
    """
    intervals = []
    for i in range(importances.shape[0]):
        data = importances[i, :]
        n = len(data)
        mean = np.mean(data)
        sem = scipy.stats.sem(data)
        margin = sem * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        intervals.append((mean, mean - margin, mean + margin))
    return intervals


def extract_order(df):
    df_sorted = df.sort_values(by=[0]).reset_index()
    df_sorted.rename(columns={"index": "feature"}, inplace=True)
    df_sorted.drop(columns=0, inplace=True)
    df_sorted = df_sorted.squeeze()
    df_sorted.index = (df_sorted.index + 1) * np.linspace(0, 1, len(df_sorted))
    return df_sorted.sort_values().index


class MatFileDataset:
    def __init__(self):
        self.X = None
        self.y = None
        self.shape = None

    def load(self, filepath: str):
        from scipy.io import loadmat
        try:
            mat = loadmat(filepath)
        except NotImplementedError:
            import mat73
            mat = mat73.loadmat(filepath)

        self.X = mat['X']
        self.y = mat['y'].reshape(-1, 1)
        self.shape = self.X.shape
        self.perc_anomalies = float(sum(self.y) / len(self.y))
        self.n_outliers = sum(self.y)


def drop_duplicates(X, y):
    df = pd.DataFrame(np.c_[X, y])
    df_unique = df.drop_duplicates().to_numpy()
    return df_unique[:, :-1], df_unique[:, -1]


def dataset(name, path="../data/"):
    filepath = path + name + ".mat"

    if filepath.endswith(".mat"):
        loader = MatFileDataset()
        loader.load(filepath)
        X, y = loader.X, loader.y
    elif filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
        X = df.drop(columns=['Target'])
        y = df['Target'].values
    else:
        raise ValueError("Unsupported file format")

    X, y = drop_duplicates(X, y)
    print(name, "\n")
    print_dataset_resume(X, y)
    return X, y


def csv_dataset(name, path="../data/"):
    filepath = path + name + ".csv"
    df = pd.read_csv(filepath, index_col=0)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    X = df[df.columns[df.columns != 'Target']]
    y = df['Target'].values
    X, y = drop_duplicates(X, y)
    print(name, "\n")
    print_dataset_resume(X, y)
    return X, y


def print_dataset_resume(X, y):
    n_samples = X.shape[0]
    outlier_ratio = np.mean(y)
    n_features = X.shape[1]
    n_outliers = int(np.sum(y))
    print(
        f"[Number of samples = {n_samples}]\n[Outlier ratio = {outlier_ratio:.4f}]\n[Number of features = {n_features}]\n[Number of outliers = {n_outliers}]")


def downsample(X, y):
    if len(X) > 2500:
        print("Downsampled to 2500")
        sss = SSS(n_splits=1, test_size=1 - 2500 / len(X))
        index = list(sss.split(X, y))[0][0]
        X, y = X[index, :], y[index]
        print(X.shape)
    return X, y


def partition_data(X, y):
    return X[y == 0, :], X[y == 1, :]
