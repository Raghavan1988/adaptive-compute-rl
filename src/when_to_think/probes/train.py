"""Train + evaluate the value-of-compute probe with strict split discipline (M3).

The whole point of M3 is Question 2: *can frozen hidden states predict the value of
continuing better than simple baselines?* Answering it honestly requires that the
probe never touch the test split during fitting or model selection (AGENTS.md §4.2):

- Standardizer statistics and probe weights are fit on TRAIN only.
- The layer and regularization strength are chosen by VAL performance only.
- The TEST split is scored exactly once, for the single selected model, and reported
  separately for the probe, the input-only baseline, and the prior.

Both targets are evaluated end-to-end and reported side by side:
``value_of_compute`` (regression) and ``fixes_incorrect`` (binary). Results are a plain
dict written to JSON, so every headline number is reproducible from the result files
(AGENTS.md §18). We describe a positive result as *decodability*, never as evidence the
model "knows" its own value of compute (CLAUDE.md).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from when_to_think.config import ProbeConfig
from when_to_think.probes import baselines as bl
from when_to_think.probes import metrics as M
from when_to_think.probes.dataset import ProbeDataset, build_probe_dataset
from when_to_think.probes.models import LogisticProbe, RidgeProbe, StandardScaler
from when_to_think.representations.reader import HiddenStateReader

# Targets evaluated. "value_of_compute" is the Δ-accuracy regression; "fixes_incorrect"
# the binary "continuing turns a wrong answer right" classification.
TARGETS = ("value_of_compute", "fixes_incorrect")


def _y_for(ds: ProbeDataset, target: str) -> np.ndarray:
    return ds.y_value if target == "value_of_compute" else ds.y_binary.astype(float)


def _selection_score(target: str, val_metrics: dict) -> float:
    """Val criterion used for model selection (higher is better).

    Regression: R² (fall back to −MSE when the val target has no variance). Binary:
    AUROC (fall back to −Brier when only one class is present in val).
    """
    if target == "value_of_compute":
        r2 = val_metrics.get("r2", math.nan)
        return r2 if not math.isnan(r2) else -val_metrics["mse"]
    auroc = val_metrics.get("auroc", math.nan)
    return auroc if not math.isnan(auroc) else -val_metrics["brier"]


def _fit_predict(
    target: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    alpha: float,
    cfg: ProbeConfig,
    standardize: bool,
) -> np.ndarray:
    """Fit a probe (train only) and return predictions on ``X_eval``."""
    scaler = StandardScaler().fit(X_train) if standardize else None
    Xtr = scaler.transform(X_train) if scaler else X_train
    Xev = scaler.transform(X_eval) if scaler else X_eval
    if target == "value_of_compute":
        model = RidgeProbe(alpha=alpha).fit(Xtr, y_train)
        return model.predict(Xev)
    model = LogisticProbe(
        alpha=alpha, lr=cfg.logreg_lr, max_iter=cfg.logreg_max_iter
    ).fit(Xtr, y_train)
    return model.predict_proba(Xev)


def _metrics(target: str, y_true: np.ndarray, pred: np.ndarray) -> dict:
    if target == "value_of_compute":
        return M.regression_metrics(y_true, pred)
    return M.binary_metrics(y_true, pred)


def _evaluate_feature_matrix(
    target: str,
    X: np.ndarray,
    y: np.ndarray,
    tr: np.ndarray,
    va: np.ndarray,
    te: np.ndarray,
    alphas: list[float],
    cfg: ProbeConfig,
    standardize: bool,
) -> dict[str, Any]:
    """Select an alpha on val, then score test once — for one feature matrix.

    Shared by the hidden-state probe and the input-only baseline so the two are scored
    with identical selection discipline on identical instances.
    """
    grid = []
    best_alpha, best_score = alphas[0], -math.inf
    for alpha in alphas:
        val_pred = _fit_predict(target, X[tr], y[tr], X[va], alpha, cfg, standardize)
        vm = _metrics(target, y[va], val_pred)
        score = _selection_score(target, vm)
        grid.append({"alpha": alpha, "val": vm, "selection_score": score})
        if score > best_score:
            best_score, best_alpha = score, alpha

    test_pred = _fit_predict(target, X[tr], y[tr], X[te], best_alpha, cfg, standardize)
    val_pred = _fit_predict(target, X[tr], y[tr], X[va], best_alpha, cfg, standardize)
    return {
        "selected_alpha": best_alpha,
        "val_selection_score": best_score,
        "grid": grid,
        "val": _metrics(target, y[va], val_pred),
        "test": _metrics(target, y[te], test_pred),
        "test_predictions": test_pred,
    }


def _decodable(target: str, probe_test: dict, baseline_test: dict) -> tuple[bool, float]:
    """Does the hidden-state probe beat the input-only baseline on test?

    Regression: higher R². Binary: higher AUROC. Returns (verdict, margin). This is a
    *decodability* claim (the value of compute is more readable from internal state than
    from the input alone), never a mechanistic one.
    """
    key = "r2" if target == "value_of_compute" else "auroc"
    p, b = probe_test.get(key, math.nan), baseline_test.get(key, math.nan)
    if math.isnan(p) or math.isnan(b):
        return False, math.nan
    return bool(p > b), float(p - b)


def _per_budget_breakdown(
    target: str, ds: ProbeDataset, tr: np.ndarray, te: np.ndarray, pred_test: np.ndarray,
) -> list[dict]:
    """Test metrics split by stop-budget (the decision-point / reasoning-step analysis)."""
    y = _y_for(ds, target)
    test_budgets = ds.budgets[te]
    out = []
    for b in sorted(set(test_budgets.tolist())):
        mask = test_budgets == b
        if not mask.any():
            continue
        out.append({"stop_budget": int(b), **_metrics(target, y[te][mask], pred_test[mask])})
    return out


def train_probe_for_target(
    target: str,
    rows: list[dict[str, Any]],
    reader: HiddenStateReader,
    cfg: ProbeConfig,
) -> dict[str, Any]:
    """Full M3 pipeline for one target: layer/alpha selection on val, test scored once."""
    # Layer sweep — build one dataset per candidate layer (instances are identical
    # across layers; only the feature vector changes).
    datasets = {layer: build_probe_dataset(
        rows, reader, layer=layer,
        continue_mode=cfg.continue_mode, correct_threshold=cfg.correct_threshold,
    ) for layer in cfg.layers}

    any_ds = next(iter(datasets.values()))
    for split in ("train", "val", "test"):
        if not any_ds.split_mask(split).any():
            raise ValueError(
                f"No instances in the '{split}' split. The probe needs train/val/test "
                "coverage — run the fixed-budget sweep with --splits train,val,test."
            )
    tr = any_ds.split_mask("train")
    va = any_ds.split_mask("val")
    te = any_ds.split_mask("test")

    # Select (layer, alpha) by val, per layer, then pick the best layer by val score.
    alphas = cfg.ridge_alphas if target == "value_of_compute" else cfg.logreg_alphas
    layer_results: dict[int, dict] = {}
    best_layer, best_score = cfg.layers[0], -math.inf
    for layer, ds in datasets.items():
        y = _y_for(ds, target)
        res = _evaluate_feature_matrix(
            target, ds.X, y, tr, va, te, alphas, cfg, cfg.standardize
        )
        layer_results[layer] = res
        if res["val_selection_score"] > best_score:
            best_score, best_layer = res["val_selection_score"], layer

    best_ds = datasets[best_layer]
    best = layer_results[best_layer]
    y_best = _y_for(best_ds, target)

    # Input-only difficulty baseline on the SAME instances (fair comparison, same
    # target/selection discipline; only the feature matrix differs).
    X_input, input_feature_names = bl.input_difficulty_features(best_ds.meta)
    input_res = _evaluate_feature_matrix(
        target, X_input, y_best, tr, va, te, alphas, cfg, standardize=True
    )

    # Prior / base-rate baseline: constant train statistic, scored on test.
    if target == "value_of_compute":
        prior_pred = bl.prior_regression_prediction(y_best[tr], int(te.sum()))
    else:
        prior_pred = bl.prior_binary_prediction(y_best[tr], int(te.sum()))
    prior_test = _metrics(target, y_best[te], prior_pred)

    decodable, margin = _decodable(target, best["test"], input_res["test"])

    # Per-example test predictions (machine-readable; plots/tables derive from these).
    test_meta = [m for m, keep in zip(best_ds.meta, te.tolist(), strict=True) if keep]
    predictions = [
        {**m, "probe_prediction": float(p), "input_baseline_prediction": float(ib)}
        for m, p, ib in zip(
            test_meta, best["test_predictions"], input_res["test_predictions"], strict=True
        )
    ]

    return {
        "target": target,
        "definition": (
            "delta = P(correct|continue) - P(correct|stop)"
            if target == "value_of_compute"
            else "1 iff P(correct|stop) < threshold <= P(correct|continue)"
        ),
        "continue_mode": cfg.continue_mode,
        "correct_threshold": cfg.correct_threshold,
        "n_instances": {
            "train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum()),
        },
        "candidate_layers": list(cfg.layers),
        "selected_layer": best_layer,
        "selected_alpha": best["selected_alpha"],
        "layerwise_val": {
            str(layer): {
                "val": layer_results[layer]["val"],
                "selection_score": layer_results[layer]["val_selection_score"],
                "selected_alpha": layer_results[layer]["selected_alpha"],
            }
            for layer in cfg.layers
        },
        "hidden_state_probe": {"val": best["val"], "test": best["test"]},
        "input_only_baseline": {
            "features": input_feature_names,
            "val": input_res["val"],
            "test": input_res["test"],
        },
        "prior_baseline": {"test": prior_test},
        "per_stop_budget_test": _per_budget_breakdown(
            target, best_ds, tr, te, np.asarray(best["test_predictions"])
        ),
        # Question 2 verdict, framed as decodability (not mechanism).
        "hidden_state_beats_input_baseline": decodable,
        "decodability_margin": margin,
        "predictions": predictions,
    }


def train_probe(
    rows: list[dict[str, Any]],
    reader: HiddenStateReader,
    cfg: ProbeConfig,
) -> dict[str, Any]:
    """Run M3 for every target and assemble the machine-readable result dict."""
    per_target = {t: train_probe_for_target(t, rows, reader, cfg) for t in TARGETS}
    return {
        "config": {
            "layers": list(cfg.layers),
            "continue_mode": cfg.continue_mode,
            "correct_threshold": cfg.correct_threshold,
            "ridge_alphas": list(cfg.ridge_alphas),
            "logreg_alphas": list(cfg.logreg_alphas),
            "standardize": cfg.standardize,
        },
        "stored_layers": reader.layers,
        "targets": per_target,
        # Honest scope note: entropy / verbalized-confidence baselines are not yet
        # collected (see probes/baselines.py), so Q2 is answered vs the input-only
        # baseline for now.
        "baselines_evaluated": ["input_only_difficulty", "prior"],
        "baselines_not_yet_available": ["entropy", "verbalized_confidence"],
    }
