"""Типы траекторий вовлечённости (k-means, GMM) и скрытые состояния (HMM).

Матрица переходов HMM между состояниями «низкая / средняя / высокая вовлечённость»
переиспользуется в статье 3 как часть модели переходов MDP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from .data import KEY


def trajectory_matrix(idx: pd.DataFrame, weeks: int) -> pd.DataFrame:
    """Студенты × недели 0..weeks−1 значений E(t).

    После отчисления E = 0, у неотчисленных пропуски заполняются последним значением.
    """
    sub = idx[idx["week"] < weeks]
    mat = sub.pivot_table(index=KEY, columns="week", values="E", observed=True)
    mat = mat.reindex(columns=range(weeks))
    withdrawn = sub.groupby(KEY, observed=True)["withdraw_week"].first().reindex(mat.index).notna()
    filled = mat.ffill(axis=1)
    filled[withdrawn] = mat[withdrawn].fillna(0.0)
    return filled.fillna(0.0)


def cluster_trajectories(mat: pd.DataFrame, k_range, sample: int, seed: int) -> dict:
    x = mat.to_numpy()
    n = len(x)
    sample = min(sample, n)
    rows = []
    for k in k_range:
        if k >= n:
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)
        sil = silhouette_score(x, km.labels_, sample_size=sample, random_state=seed)
        gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=seed).fit(x)
        rows.append({"k": k, "kmeans_inertia": km.inertia_, "silhouette": sil, "gmm_bic": gmm.bic(x)})
    sel = pd.DataFrame(rows)
    best_k = int(sel.loc[sel["silhouette"].idxmax(), "k"])

    km = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit(x)
    # нумерация кластеров по возрастанию средней вовлечённости
    order = np.argsort(km.cluster_centers_.mean(axis=1))
    remap = {old: new for new, old in enumerate(order)}
    labels = pd.Series([remap[l] for l in km.labels_], index=mat.index, name="cluster")
    centers = pd.DataFrame(km.cluster_centers_[order], columns=mat.columns)
    return {"selection": sel, "best_k": best_k, "labels": labels, "centers": centers}


def cluster_profiles(labels: pd.Series, student_info: pd.DataFrame) -> pd.DataFrame:
    df = labels.reset_index().merge(student_info[KEY + ["final_result"]], on=KEY, how="left")
    tab = pd.crosstab(df["cluster"], df["final_result"], normalize="index").round(3)
    tab["n_students"] = df.groupby("cluster").size()
    return tab


def fit_hmm(idx: pd.DataFrame, n_states_range, max_sequences: int, n_iter: int, seed: int) -> dict:
    """Гауссовская HMM на последовательностях (B(t), C(t)); число состояний — по BIC."""
    rng = np.random.default_rng(seed)
    idx = idx.sort_values(KEY + ["week"])
    groups = idx.groupby(KEY, sort=False, observed=True).indices
    keys = list(groups)
    pick = rng.choice(len(keys), size=min(max_sequences, len(keys)), replace=False)

    feats = ["B", "C"]
    x_all = idx[feats].to_numpy()
    seqs = [groups[keys[i]] for i in pick]
    x_fit = np.concatenate([x_all[s] for s in seqs])
    lengths = [len(s) for s in seqs]

    rows, models = [], {}
    for n in n_states_range:
        m = GaussianHMM(n_components=n, covariance_type="diag", n_iter=n_iter, random_state=seed)
        m.fit(x_fit, lengths)
        rows.append({"n_states": n, "log_likelihood": m.score(x_fit, lengths), "bic": m.bic(x_fit, lengths)})
        models[n] = m
    sel = pd.DataFrame(rows)
    best_n = int(sel.loc[sel["bic"].idxmin(), "n_states"])
    m = models[best_n]

    # состояния по возрастанию средней (B + C): 0 — самая низкая вовлечённость
    order = np.argsort(m.means_.sum(axis=1))
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))

    all_lengths = [len(v) for v in groups.values()]
    x_ordered = np.concatenate([x_all[v] for v in groups.values()])
    states = remap[m.predict(x_ordered, all_lengths)]
    row_index = np.concatenate(list(groups.values()))
    state_col = pd.Series(states, index=idx.index[row_index], name="hmm_state")

    trans = pd.DataFrame(m.transmat_[np.ix_(order, order)])
    means = pd.DataFrame(m.means_[order], columns=feats)
    return {"selection": sel, "best_n": best_n, "transmat": trans, "means": means, "states": state_col}
