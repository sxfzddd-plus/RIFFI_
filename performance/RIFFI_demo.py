import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import warnings
sys.path.append('../')
from utils_.utils_ import *
from models.RIFFI_ import *
from ucimlrepo import fetch_ucirepo

sns.set()
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


def plot_feature_bar(global_importances, name):
    filename = f'Feat_bar_plot_{name}'
    patterns = [None, "/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]

    imp_vals = global_importances['Importances']
    feat_imp = pd.DataFrame({
        'Global Importance': np.round(imp_vals, 3),
        'Feature': global_importances['feat_order'],
        'std': global_importances['std']
    })

    if len(feat_imp) > 15:
        feat_imp = feat_imp.iloc[-15:].reset_index(drop=True)

    plt.style.use('default')
    plt.rcParams['axes.facecolor'] = '#F2F2F2'
    plt.rcParams['axes.axisbelow'] = True

    ax1 = feat_imp.plot(
        y='Global Importance', x='Feature', kind="barh",
        xerr='std', capsize=5, alpha=1, legend=False
    )

    ax1.grid(alpha=0.7)
    ax2 = ax1.twinx()
    values = [f"{v} ± {np.round(feat_imp['std'][i], 2)}" for i, v in enumerate(feat_imp['Global Importance'])]

    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(range(feat_imp.shape[0]))
    ax2.set_yticklabels(values)
    ax2.grid(alpha=0)

    plt.axvline(x=0, color=".5")
    ax1.set_xlabel('Importance Score', fontsize=20)
    ax1.set_ylabel('Features', fontsize=20)
    plt.xlim(np.min(imp_vals) - 0.2 * np.min(imp_vals))
    plt.subplots_adjust(left=0.3)
    plt.savefig(f'../pickle_file/{filename}_feat_bar.png', bbox_inches='tight')


def plot_importance_distribution(importances, name, dim, top_k):
    label_list = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
    if 'GFI_' not in name:
        name = 'LFI_' + name

    color = plt.cm.get_cmap('tab20', 20).colors
    importances_matrix = np.array([
        np.array(pd.Series(row).sort_values(ascending=False).index) for row in importances
    ])

    bar_data = [[(list(importances_matrix[:, j]).count(i) / len(importances_matrix)) * 100
                 for i in range(dim)] for j in range(dim)]
    bars = pd.DataFrame(bar_data)

    tick_labels = [
        rf'${i}^{{st}}$' if i == 1 else
        rf'${i}^{{nd}}$' if i == 2 else
        rf'${i}^{{rd}}$' if i == 3 else
        rf'${i}^{{th}}$' for i in range(1, top_k + 1)
    ]

    plt.figure(figsize=(16, 10))
    font = {'family': 'serif', 'serif': 'Times New Roman', 'weight': 'normal', 'size': 60}
    plt.rc('font', **font)
    plt.gcf().set_facecolor('white')
    plt.gca().set_facecolor('white')

    for i in range(dim):
        plt.bar(range(top_k), bars.T.iloc[i, :top_k].values,
                bottom=bars.T.iloc[:i, :top_k].sum().values,
                color=color[i % 20], edgecolor='white', width=0.85, label=label_list[i])

    plt.xlabel("Rank", fontsize=25)
    plt.xticks(range(top_k), tick_labels, fontsize=37)
    plt.ylabel("Percentage count", fontsize=25)
    plt.yticks(range(10, 101, 10), [f"{x}%" for x in range(10, 101, 10)], fontsize=37)
    plt.legend(bbox_to_anchor=(1.05, 0.95), loc="upper left", fontsize=37)
    plt.savefig(f'importances_bars.png', bbox_inches='tight')
    plt.show()
    plt.close()


def compute_imps_shap(model, X_train, X_test, sample_size):
    model.fit(X_train)
    explainer = shap.TreeExplainer(model)
    imps = np.zeros((sample_size, X_train.shape[1]))

    for i in tqdm(range(sample_size)):
        shap_values = explainer.shap_values(X_test[i:i + 1])
        imps[i, :] = shap_values[0]

    mean_imp = np.mean(imps, axis=0)
    std_imp = np.std(imps, axis=0)
    mean_imp_val = np.sort(mean_imp)
    feat_order = mean_imp.argsort()

    return imps, {'Importances': mean_imp_val, 'feat_order': feat_order, 'std': std_imp[feat_order]}


def compute_imps_loc(model, X_train, X_test, sample_size):
    model.fit(X_train)
    imps = model.Local_importances(X_test, calculate=True, overwrite=False, depth_based=False)
    imps = np.abs(imps)
    mean_imp = np.mean(imps, axis=0)
    std_imp = np.std(imps, axis=0)
    mean_imp_val = np.sort(mean_imp)
    feat_order = mean_imp.argsort()

    return imps, {'Importances': mean_imp_val, 'feat_order': feat_order, 'std': std_imp[feat_order]}


def interpretation_plots(name, model_class):
    target = 'UCL'
    glass_identification = fetch_ucirepo(id=42)

    X = glass_identification.data.features
    y = glass_identification.data.targets

    X_anomaly = X[np.isin(y, 7)]

    X_train = X[~np.isin(y, 7)]

    if model_class == RIFFI_original:
        name = 'loc_minmax_' + name + 'GFI_RIFFI_original_'
        model = RIFFI_original(128, max_depth=100, l_mean_weight=1.5, r_mean_weight=0.9,
                                subsample_size=128, plus=1)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train_scaled = scaler.fit_transform(X_train)
        X_anomaly_scaled = scaler.transform(X_anomaly)

        imps, plt_data = compute_imps_loc(model, X_train_scaled, X_anomaly_scaled, len(X_anomaly_scaled))
        plot_importance_distribution(imps, name, X.shape[1], X.shape[1])


interpretation_plots('glass', RIFFI_original)
