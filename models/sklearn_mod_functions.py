from sklearn.ensemble._iforest import _average_path_length
from sklearn.utils.validation import _num_samples
from sklearn.utils import gen_batches, get_chunk_n_rows
import numpy as np

# The functions below have been adapted from the sklearn source code

def decision_function_single_tree(iforest, tree_idx, X):
    """
    Computes the decision function for samples X using a single tree from the isolation forest.

    Parameters
    ----------
    iforest : IsolationForest object
        The fitted Isolation Forest model.
    tree_idx : int
        Index of the tree to use from the forest.
    X : array-like of shape (n_samples, n_features)
        Input data.

    Returns
    -------
    scores : ndarray of shape (n_samples,)
        Anomaly scores for each sample.
    """
    return _score_samples(iforest, tree_idx, X) - iforest.offset_


def _score_samples(iforest, tree_idx, X):
    if iforest.n_features_in_ != X.shape[1]:
        raise ValueError(
            f"Number of features of the model must match the input. "
            f"Model n_features is {iforest.n_features_in_} and "
            f"input n_features is {X.shape[1]}."
        )
    return -_compute_chunked_score_samples(iforest, tree_idx, X)


def _compute_chunked_score_samples(iforest, tree_idx, X):
    n_samples = _num_samples(X)
    use_subsampled_features = iforest._max_features != X.shape[1]

    chunk_size = get_chunk_n_rows(
        row_bytes=16 * iforest._max_features,
        max_n_rows=n_samples
    )
    batch_slices = gen_batches(n_samples, chunk_size)

    scores = np.zeros(n_samples, order="f")
    for sl in batch_slices:
        scores[sl] = _compute_score_samples_single_tree(
            iforest, tree_idx, X[sl], use_subsampled_features
        )
    return scores


def _compute_score_samples_single_tree(iforest, tree_idx, X, use_subsampled_features):
    n_samples = X.shape[0]
    depths = np.zeros(n_samples, order="f")

    tree = iforest.estimators_[tree_idx]
    features = iforest.estimators_features_[tree_idx]

    X_input = X[:, features] if use_subsampled_features else X
    leaf_indices = tree.apply(X_input)
    node_paths = tree.decision_path(X_input)

    samples_per_leaf = tree.tree_.n_node_samples[leaf_indices]
    path_lengths = np.ravel(node_paths.sum(axis=1)) + _average_path_length(samples_per_leaf) - 1.0

    scores = 2 ** (-path_lengths / _average_path_length([iforest.max_samples_]))
    return scores
