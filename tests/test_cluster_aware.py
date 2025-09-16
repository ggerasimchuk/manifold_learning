import numpy as np
import pandas as pd

from tools.manifold import embed_umap_euclid, transform_prefix_to_Z
from tools.clustering import approximate_membership
from tools.forecast import (
    knn_forecast_cluster_restricted,
    knn_forecast_cluster_weighted,
)

from tools.manifold import embed_umap_euclid, transform_prefix_to_Z
from tools.clustering import approximate_membership
from tools.forecast import (
    knn_forecast_cluster_restricted,
    knn_forecast_cluster_weighted,
)


def test_transform_prefix_to_Z_padding():
    rng = np.random.default_rng(0)
    tensor_channels = ("r_oil_norm", "wc")
    X = rng.normal(size=(6, 8, len(tensor_channels)))
    Z_train, umap_model = embed_umap_euclid(
        X,
        tensor_channels=tensor_channels,
        channels=tensor_channels,
        n_neighbors=3,
        min_dist=0.1,
        random_state=42,
    )

    prefix = X[0, :4, :]
    proj = transform_prefix_to_Z(umap_model, prefix, tensor_channels, tensor_channels)

    padded = np.zeros_like(X[0])
    padded[:4] = prefix
    flat = padded.transpose(1, 0).reshape(1, -1)
    manual = umap_model.transform(flat)

    assert proj.shape == (1, Z_train.shape[1])
    assert np.allclose(proj, manual)


def test_approximate_membership_shapes():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(30, 2))
    import hdbscan

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True)
    clusterer.fit(data)

    new_points = rng.normal(size=(4, 2))
    res = approximate_membership(clusterer, new_points)
    assert set(res.keys()) == {"labels", "prob"}
    assert res["labels"].shape == (4,)
    assert res["prob"].shape == (4,)


def test_knn_forecast_cluster_restricted_prefers_cluster():
    T = 6
    T_pref = 3
    Y_full = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1.1, 1.1, 1.1, 1.0, 0.9, 0.8],
            [1.05, 1.0, 0.95, np.nan, np.nan, np.nan],
            [0.8, 0.8, 0.8, 0.7, 0.6, 0.5],
        ],
        dtype=float,
    )
    X_pref = Y_full[:, :T_pref]

    df_map = pd.DataFrame(
        {
            "well": ["w0", "w1", "w2", "w3"],
            "cluster": [0, 0, 0, 1],
            "prob": [0.9, 0.8, 0.85, 0.6],
            "x": [0.0, 0.5, 0.2, -0.5],
            "y": [0.0, 0.1, -0.1, 0.4],
            "anomaly_score": [0.1, 0.2, 0.1, 0.3],
        }
    )

    preds, info = knn_forecast_cluster_restricted(
        X_pref,
        Y_full.copy(),
        T_pref=T_pref,
        df_map=df_map,
        train_indices=np.array([0, 1, 3]),
        K=2,
        min_incluster=1,
        mix_global_frac=0.5,
    )

    neighbors = info["selection"][2]["neighbors"]
    assert neighbors == [0, 1]
    assert np.allclose(preds[2], np.nanmedian(np.vstack([Y_full[0, T_pref:], Y_full[1, T_pref:]]), axis=0))


def test_knn_forecast_cluster_weighted_weights():
    T = 6
    T_pref = 3
    Y_full = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [0.9, 0.95, 1.0, 0.9, 0.8, 0.7],
            [1.05, 1.0, 0.95, np.nan, np.nan, np.nan],
            [0.5, 0.6, 0.7, 0.7, 0.6, 0.5],
        ],
        dtype=float,
    )
    X_pref = Y_full[:, :T_pref]
    Z = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.1, 0.0],
            [-1.0, -0.5],
        ],
        dtype=float,
    )
    df_map = pd.DataFrame(
        {
            "well": ["w0", "w1", "w2", "w3"],
            "cluster": [0, 1, 0, 1],
            "prob": [0.9, 0.7, 0.85, 0.6],
            "x": Z[:, 0],
            "y": Z[:, 1],
            "anomaly_score": [0.1, 0.2, 0.1, 0.3],
        }
    )

    preds, info = knn_forecast_cluster_weighted(
        X_pref,
        Y_full.copy(),
        T_pref=T_pref,
        Z=Z,
        df_map=df_map,
        K=2,
        alpha_map=0.0,
        w_cluster_match=1.0,
    )

    weights = info["selection"][2]["weights"]
    assert len(weights) == 2
    assert weights[0] > weights[1]
    assert preds.shape[1] == T - T_pref
