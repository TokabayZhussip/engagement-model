import numpy as np
import pytest

from engagement.data import load_oulad
from engagement.features import BEHAVIORAL, COGNITIVE, build_weekly_features
from engagement.index import build_engagement_index, cronbach_alpha, entropy_weights, validate_index
from engagement.predict import fit_tabular_models, make_drop_dataset, split_by_presentation
from engagement.states import cluster_trajectories, describe_clusters, fit_hmm, trajectory_matrix
from scripts.make_synthetic_oulad import generate


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    raw = tmp_path_factory.mktemp("oulad")
    generate(raw, n_students=40, seed=1)
    return load_oulad(raw)


@pytest.fixture(scope="module")
def index(data):
    panel = build_weekly_features(data, 38, ["quiz", "forumng"])
    idx, w = build_engagement_index(panel)
    return panel, idx, w


def test_entropy_weights_sum_to_one():
    rng = np.random.default_rng(0)
    x = rng.random((200, 4))
    x[:, 0] = 0.5  # константный индикатор не несёт информации
    w = entropy_weights(x)
    assert np.isclose(w.sum(), 1.0)
    assert w[0] < w[1:].min()


def test_cronbach_alpha_perfect_items():
    v = np.linspace(0, 1, 50)
    assert np.isclose(cronbach_alpha(np.column_stack([v, v, v])), 1.0)


def test_clicks_preserved(data, index):
    panel, _, _ = index
    sv = data["student_vle"]
    assert panel["clicks"].sum() <= sv.loc[sv["date"] >= 0, "sum_click"].sum()
    assert panel["clicks"].sum() > 0


def test_index_range_and_weights(index):
    _, idx, w = index
    for col in ["B", "C", "E"]:
        assert idx[col].between(0, 1).all()
    assert np.isclose(sum(w["behavioral"].values()), 1)
    assert np.isclose(sum(w["cognitive"].values()), 1)
    assert len(w["behavioral"]) == len(BEHAVIORAL) and len(w["cognitive"]) == len(COGNITIVE)


def test_index_related_to_outcome(data, index):
    _, idx, _ = index
    v = validate_index(idx, data["student_info"])
    assert v["spearman_rho_E_vs_outcome"] > 0  # в синтетике вовлечённость связана с итогом


def test_states(index):
    _, idx, _ = index
    mat = trajectory_matrix(idx, 15)
    cl = cluster_trajectories(mat, [2, 3], 1000, 0)
    assert cl["labels"].nunique() == cl["best_k"]
    h = fit_hmm(idx, [2, 3], 200, 20, 0, train_mask=idx["code_presentation"] != "2014J")
    assert np.allclose(h["transmat"].sum(axis=1), 1)
    assert len(h["states"]) == len(idx)
    f = h["filtered"]
    pcols = [c for c in f.columns if c.startswith("hmm_p")]
    assert np.allclose(f[pcols].sum(axis=1), 1)
    assert len(f) == len(idx)


def test_cluster_k_final(index):
    _, idx, _ = index
    mat = trajectory_matrix(idx, 15)
    cl = cluster_trajectories(mat, [2, 3, 4], 1000, 0, k_final=3)
    assert cl["best_k"] == 3 and cl["labels"].nunique() == 3
    desc = describe_clusters(cl["centers"])
    assert len(desc) == 3 and desc["type"].notna().all()


@pytest.mark.parametrize("label", ["hmm", "ratio"])
def test_prediction(index, label):
    _, idx, _ = index
    h = fit_hmm(idx, [3], 200, 20, 0)
    idx = idx.join(h["filtered"])
    ds = make_drop_dataset(idx, 2, 3, 0.5, 0.2, label)
    assert set(ds["y"].unique()) <= {0, 1}
    if label == "hmm":
        assert (ds["hmm_fstate"] >= 1).all()
    tr, va, te = split_by_presentation(ds, ["2014J"], 0.1, 0)
    assert not (tr & te).any() and not (va & te).any()
    res = fit_tabular_models(ds, tr, va, te, 0)
    assert 0.5 <= res["gbm"]["metrics"]["roc_auc"] <= 1.0
