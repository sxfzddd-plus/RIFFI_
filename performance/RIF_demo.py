import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from RIFFI_.models.RIF import RIF_model
import random
from ucimlrepo import fetch_ucirepo

def softmax(x):
    x = np.array(x)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def generate_circle_points(n, radius, cx=0, cy=0):
    points = []
    for _ in range(n):
        while True:
            x, y = random.uniform(-radius, radius), random.uniform(-radius, radius)
            if x**2 + y**2 <= radius**2:
                points.append((x + cx, y + cy))
                break
    return np.array(points)

def label_circle_anomaly(X, radius):
    return np.array([1 if x[0]**2 + x[1]**2 >= radius**2 else 0 for x in X])

def label_double_blob_anomaly(X, radius, center1, center2):
    return np.array([
        1 if np.linalg.norm(x - center1) >= radius and np.linalg.norm(x - center2) >= radius else 0
        for x in X
    ])

def label_sinusoid_anomaly(X, margin, xmin, xmax):
    return np.array([
        1 if x[0] <= xmin or x[0] >= xmax or abs(x[1]) >= np.sin(x[0] * 8 * np.pi) + margin else 0
        for x in X
    ])

def evaluate_model(X_train, X_eval, y_eval, lw, rw,subsample=256):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_eval = scaler.transform(X_eval)

    model = RIF_model(n_trees=200, l_mean_weight=lw, r_mean_weight=rw, subsample_size=subsample)
    model.fit(X_train)
    scores = model.Anomaly_Score(X_eval)
    probs = softmax(scores)

    fpr, tpr, _ = roc_curve(y_eval, probs, pos_label=1)
    precision, recall, _ = precision_recall_curve(y_eval, probs)
    return auc(fpr, tpr), average_precision_score(y_eval, probs)

# ==== experiment ====
def experiment_single_blob():
    print("\n[single_blob]")
    X = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 500)
    X[0] = [3.3, 3.3]
    test_X = generate_circle_points(200, 10)
    test_y = label_circle_anomaly(test_X, 4)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    aucs, prcs = [], []

    for train_idx, _ in kf.split(X):
        auc_, prc_ = evaluate_model(X[train_idx], test_X, test_y, lw=0.6, rw=1.5)
        aucs.append(auc_)
        prcs.append(prc_)

    print("AUC: %.2f (+/- %.2f)" % (np.mean(aucs), 2*np.std(aucs)))
    print("PRC: %.2f (+/- %.2f)" % (np.mean(prcs), 2*np.std(prcs)))

def experiment_double_blob():
    print("\n[Double Blob]")
    np.random.seed(2)
    blob1 = np.random.multivariate_normal([35, 0], [[1, 0], [0, 1]], 250)
    blob2 = np.random.multivariate_normal([0, 35], [[1, 0], [0, 1]], 250)
    X = np.vstack((blob1, blob2))

    test_X = generate_circle_points(2000, 37, 17.5, 17.5)
    test_y = label_double_blob_anomaly(test_X, 4, np.array([35, 0]), np.array([0, 35]))

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    aucs, prcs = [], []

    for train_idx, _ in kf.split(X):
        auc_, prc_ = evaluate_model(X[train_idx], test_X, test_y, lw=0.3, rw=1.3)
        aucs.append(auc_)
        prcs.append(prc_)

    print("AUC: %.2f (+/- %.2f)" % (np.mean(aucs), 2*np.std(aucs)))
    print("PRC: %.2f (+/- %.2f)" % (np.mean(prcs), 2*np.std(prcs)))

def experiment_sinusoid():
    print("\n[Sinusoid]")
    N = 1000
    x = np.random.rand(N) * 8 * np.pi
    y = np.sin(x) + np.random.randn(N) / 4.
    X = np.column_stack((x, y))

    test_X = generate_circle_points(500, 17, 13, 0)
    test_y = label_sinusoid_anomaly(test_X, 4, 0, 28)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    aucs, prcs = [], []

    for train_idx, _ in kf.split(X):

        auc_, prc_ = evaluate_model(X[train_idx], test_X, test_y, lw=1.2, rw=1.9)
        aucs.append(auc_)
        prcs.append(prc_)

    print("AUC: %.2f (+/- %.2f)" % (np.mean(aucs), 2*np.std(aucs)))
    print("PRC: %.2f (+/- %.2f)" % (np.mean(prcs), 2*np.std(prcs)))

def experiment_glass_local():
    print("\n[UCI Glass]")
    glass_identification = fetch_ucirepo(id=42)

    X = glass_identification.data.features
    y = glass_identification.data.targets

    X = X.values
    y = y.values.reshape(len(y))

    normal_X = X[y != 7]
    anomaly_X = X[y == 7]
    normal_y = np.zeros(len(normal_X))
    anomaly_y = np.ones(len(anomaly_X))


    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    aucs, prcs = [], []

    for train_idx, test_idx in kf.split(normal_X):
        X_train = normal_X[train_idx]
        X_test = normal_X[test_idx]
        y_test = normal_y[test_idx]

        X_eval = np.vstack((X_test, anomaly_X))
        y_eval = np.concatenate((y_test, anomaly_y))

        auc_, prc_ = evaluate_model(X_train, X_eval, y_eval, lw=0.6, rw=0.3,subsample=32)
        aucs.append(auc_)
        prcs.append(prc_)

    print("AUC: %.2f (+/- %.2f)" % (np.mean(aucs), 2*np.std(aucs)))
    print("PRC: %.2f (+/- %.2f)" % (np.mean(prcs), 2*np.std(prcs)))

# ==== 主函数入口 ====
def main():
    experiment_single_blob()
    experiment_double_blob()
    experiment_sinusoid()
    experiment_glass_local()

if __name__ == '__main__':
    main()
