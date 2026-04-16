"""
Compare model architectures on the enhanced dataset (v2).

Models tested:
  1. Ridge Regression (linear baseline)
  2. LightGBM (baseline regression)
  3. LightGBM (tuned regression)
  4. LightGBM (LambdaRank -- ranking objective)
  5. XGBoost (regression)
  6. Neural Network (MLP with BatchNorm)

For each: R^2, RMSE, MAE, rank-1 accuracy, rank confusion.
"""
import sys
import time
import io
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from data_prep.domain_config import get_domain_config
from models.evaluation_split import build_eval_split

EXCLUDE_COLS = {
    "driver_id",
    "route_idx",
    "service_month",
    "service_date",
    "service_window_pos",
    "expected_revenue",
    "driver_cost",
    "expected_profit",
}
TARGET = "expected_profit"


def flush():
    sys.stdout.flush()
    sys.stderr.flush()


def load_data(dataset_path: Path):
    print("  Loading dataset...", end=" ")
    flush()
    df = pd.read_parquet(dataset_path)
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    print(f"OK  ({len(df):,} rows, {len(feat_cols)} features)")
    flush()
    return df, feat_cols


def split_data(df, feat_cols):
    print("  Splitting train/val...", end=" ")
    flush()
    X = df[feat_cols].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)
    groups = df["driver_id"].values
    split = build_eval_split(df)
    train_idx, val_idx = split.train_idx, split.val_idx
    print(f"OK ({split.train_label} -> {split.val_label})")
    flush()
    return X, y, groups, train_idx, val_idx, split


def rank_accuracy(df, val_idx, y_pred):
    """Top-1 rank accuracy: did we pick the route with highest true profit?"""
    val_df = df.iloc[val_idx][["driver_id", "expected_profit"]].copy()
    val_df["pred"] = y_pred

    correct, total = 0, 0
    grouped = list(val_df.groupby("driver_id"))
    for _, grp in tqdm(grouped, desc="    Rank eval", unit="drv", ncols=90, leave=False):
        if len(grp) < 2:
            continue
        total += 1
        if grp["expected_profit"].idxmax() == grp["pred"].idxmax():
            correct += 1
    return correct / total if total > 0 else 0, correct, total


def evaluate(name, y_true, y_pred, df, val_idx):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    rank_acc, rc, rt = rank_accuracy(df, val_idx, y_pred)
    print(f"    R2={r2:.4f}  RMSE=${rmse:.2f}  MAE=${mae:.2f}  Corr={corr:.4f}")
    print(f"    Rank-1 accuracy: {rank_acc:.1%} ({rc:,}/{rt:,})")
    flush()
    return {"model": name, "r2": r2, "rmse": rmse, "mae": mae, "corr": corr,
            "rank_acc": rank_acc, "rank_correct": rc, "rank_total": rt}


def _lgb_progress(period=100):
    """LightGBM callback that prints progress every `period` rounds."""
    import lightgbm as lgb
    def _callback(env):
        if (env.iteration + 1) % period == 0 or env.iteration == 0:
            val_score = env.evaluation_result_list[0][2] if env.evaluation_result_list else "?"
            tqdm.write(f"      round {env.iteration + 1:>5d}  val={val_score}")
    _callback.order = 10
    return _callback


def train_lgb_baseline(X, y, train_idx, val_idx, feat_cols):
    import lightgbm as lgb
    params = {
        "objective": "regression", "metric": "rmse",
        "learning_rate": 0.05, "num_leaves": 63, "max_depth": 8,
        "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "verbose": -1,
    }
    dtrain = lgb.Dataset(X[train_idx], label=y[train_idx], feature_name=feat_cols, free_raw_data=False)
    dval = lgb.Dataset(X[val_idx], label=y[val_idx], feature_name=feat_cols, reference=dtrain, free_raw_data=False)
    model = lgb.train(params, dtrain, num_boost_round=1000,
                      valid_sets=[dval], valid_names=["val"],
                      callbacks=[lgb.early_stopping(50, verbose=False), _lgb_progress(200)])
    return model, model.predict(X[val_idx])


def train_lgb_tuned(X, y, train_idx, val_idx, feat_cols):
    import lightgbm as lgb
    params = {
        "objective": "regression", "metric": "rmse",
        "learning_rate": 0.03, "num_leaves": 127, "max_depth": 10,
        "min_child_samples": 30, "subsample": 0.85, "colsample_bytree": 0.7,
        "reg_alpha": 0.05, "reg_lambda": 0.5,
        "min_split_gain": 0.01, "verbose": -1,
    }
    dtrain = lgb.Dataset(X[train_idx], label=y[train_idx], feature_name=feat_cols, free_raw_data=False)
    dval = lgb.Dataset(X[val_idx], label=y[val_idx], feature_name=feat_cols, reference=dtrain, free_raw_data=False)
    model = lgb.train(params, dtrain, num_boost_round=2000,
                      valid_sets=[dval], valid_names=["val"],
                      callbacks=[lgb.early_stopping(80, verbose=False), _lgb_progress(200)])
    return model, model.predict(X[val_idx])


def _profit_to_relevance(y, groups):
    """Convert profit labels to per-group relevance grades (0-4) for LambdaRank."""
    relevance = np.zeros(len(y), dtype=np.int32)
    offset = 0
    for g in groups:
        chunk = y[offset:offset + g]
        if g == 1:
            relevance[offset] = 2
        else:
            order = np.argsort(np.argsort(chunk))
            rel = np.round(order / max(g - 1, 1) * 4).astype(np.int32)
            relevance[offset:offset + g] = rel
        offset += g
    return relevance


def train_lgb_lambdarank(X, y, df, train_idx, val_idx, feat_cols):
    """LightGBM LambdaRank: directly optimizes pairwise route ordering."""
    import lightgbm as lgb

    print("    Preparing ranking groups...")
    flush()
    tr_df = df.iloc[train_idx].reset_index(drop=True)
    va_df = df.iloc[val_idx].reset_index(drop=True)

    va_df["_orig_idx"] = np.arange(len(va_df))
    tr_sorted = tr_df.sort_values("driver_id").reset_index(drop=True)
    va_sorted = va_df.sort_values("driver_id").reset_index(drop=True)

    X_tr = tr_sorted[feat_cols].values.astype(np.float32)
    y_tr = tr_sorted[TARGET].values.astype(np.float32)
    X_va = va_sorted[feat_cols].values.astype(np.float32)
    y_va = va_sorted[TARGET].values.astype(np.float32)

    tr_groups = tr_sorted.groupby("driver_id", sort=False).size().values
    va_groups = va_sorted.groupby("driver_id", sort=False).size().values

    print(f"    Train groups: {len(tr_groups):,}  Val groups: {len(va_groups):,}")
    flush()

    tr_rel = _profit_to_relevance(y_tr, tr_groups)
    va_rel = _profit_to_relevance(y_va, va_groups)

    dtrain = lgb.Dataset(X_tr, label=tr_rel, group=tr_groups,
                         feature_name=feat_cols, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=va_rel, group=va_groups,
                       feature_name=feat_cols, reference=dtrain, free_raw_data=False)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [1, 3],
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": 10,
        "min_child_samples": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.05,
        "reg_lambda": 0.5,
        "label_gain": [0, 1, 2, 3, 4],
        "verbose": -1,
    }

    print("    Training LambdaRank...")
    flush()
    model = lgb.train(params, dtrain, num_boost_round=1500,
                      valid_sets=[dval], valid_names=["val"],
                      callbacks=[lgb.early_stopping(60, verbose=False), _lgb_progress(200)])

    raw_scores_sorted = model.predict(X_va)

    orig_order = va_sorted["_orig_idx"].values
    raw_scores = np.empty(len(raw_scores_sorted), dtype=np.float32)
    raw_scores[orig_order] = raw_scores_sorted
    return model, raw_scores


def train_xgboost(X, y, train_idx, val_idx, feat_cols):
    import xgboost as xgb
    params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.03, "max_depth": 8, "min_child_weight": 30,
        "subsample": 0.85, "colsample_bytree": 0.7,
        "reg_alpha": 0.05, "reg_lambda": 0.5,
        "tree_method": "hist", "verbosity": 0,
    }
    dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx], feature_names=feat_cols)
    dval = xgb.DMatrix(X[val_idx], label=y[val_idx], feature_names=feat_cols)
    model = xgb.train(params, dtrain, num_boost_round=2000,
                      evals=[(dval, "val")],
                      early_stopping_rounds=80, verbose_eval=200)
    return model, model.predict(dval)


def train_mlp(X, y, train_idx, val_idx, feat_cols):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
    except ImportError:
        print("    PyTorch not installed, skipping MLP")
        return None, None

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[train_idx])
    X_va = scaler.transform(X[val_idx])

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y[train_idx], dtype=torch.float32).unsqueeze(1)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    y_va_t = torch.tensor(y[val_idx], dtype=torch.float32).unsqueeze(1)

    n_feat = X_tr.shape[1]
    model = nn.Sequential(
        nn.Linear(n_feat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.15),
        nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(64, 1),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_ds, batch_size=2048, shuffle=True)

    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = 25

    pbar = tqdm(range(200), desc="    MLP epochs", unit="ep", ncols=90)
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_va_t)
            val_loss = criterion(val_pred, y_va_t).item()
        scheduler.step(val_loss)

        pbar.set_postfix({"val_loss": f"{val_loss:.4f}", "best": f"{best_val_loss:.4f}",
                          "pat": f"{patience_counter}/{max_patience}"})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                pbar.close()
                print(f"    Early stop at epoch {epoch + 1}")
                break
    else:
        pbar.close()

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_va_t).squeeze().numpy()

    flush()
    return (model, scaler), y_pred


def train_ridge(X, y, train_idx, val_idx, feat_cols):
    from sklearn.linear_model import Ridge
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[train_idx])
    X_va = scaler.transform(X[val_idx])
    model = Ridge(alpha=1.0)
    model.fit(X_tr, y[train_idx])
    return (model, scaler), model.predict(X_va)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare model families on the domain-specific training dataset")
    parser.add_argument("--domain", type=str, default="yellow", choices=["yellow", "green"])
    parser.add_argument("--dataset", type=str, default="", help="Optional explicit dataset path")
    args = parser.parse_args()

    domain_config = get_domain_config(args.domain)
    dataset_path = Path(args.dataset) if args.dataset else domain_config.training_dataset_path()
    model_dir = ROOT / "models" if args.domain == "yellow" else domain_config.models_dir
    results_dir = ROOT / "results" if args.domain == "yellow" else domain_config.results_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=== Model Comparison (Enhanced Features v2) ===\n")
    flush()
    df, feat_cols = load_data(dataset_path)
    X, y, groups, train_idx, val_idx, split = split_data(df, feat_cols)

    print(f"  Train: {len(train_idx):,} rows  Val: {len(val_idx):,} rows")
    n_train_drivers = len(set(groups[train_idx]))
    n_val_drivers = len(set(groups[val_idx]))
    print(f"  Train drivers: {n_train_drivers:,}  Val drivers: {n_val_drivers:,}")
    print(f"  Split name: {split.split_name}")
    flush()

    models_to_train = [
        ("Ridge (linear)",      lambda: train_ridge(X, y, train_idx, val_idx, feat_cols)),
        ("LightGBM (baseline)", lambda: train_lgb_baseline(X, y, train_idx, val_idx, feat_cols)),
        ("LightGBM (tuned)",    lambda: train_lgb_tuned(X, y, train_idx, val_idx, feat_cols)),
        ("LGB LambdaRank",      lambda: train_lgb_lambdarank(X, y, df, train_idx, val_idx, feat_cols)),
        ("XGBoost",             lambda: train_xgboost(X, y, train_idx, val_idx, feat_cols)),
        ("MLP (Neural Net)",    lambda: train_mlp(X, y, train_idx, val_idx, feat_cols)),
    ]

    results = []
    trained_models = {}

    for i, (name, train_fn) in enumerate(models_to_train, 1):
        print(f"\n{'='*60}")
        print(f"  [{i}/{len(models_to_train)}] Training: {name}")
        print(f"{'='*60}")
        flush()
        t0 = time.time()
        try:
            model_obj, y_pred = train_fn()
        except ImportError as e:
            print(f"    SKIPPED: {e}")
            flush()
            continue
        except Exception as e:
            print(f"    ERROR: {e}")
            flush()
            continue

        if y_pred is None:
            continue

        elapsed = time.time() - t0
        print(f"  Training time: {elapsed:.1f}s")
        flush()

        if hasattr(model_obj, "best_iteration"):
            print(f"  Best iteration: {model_obj.best_iteration}")

        result = evaluate(name, y[val_idx], y_pred, df, val_idx)
        results.append(result)
        trained_models[name] = model_obj

    # Summary table
    print(f"\n{'='*78}")
    print("SUMMARY")
    print(f"{'='*78}")
    print(f"{'Model':25s} {'R2':>8s} {'RMSE':>8s} {'MAE':>8s} {'Corr':>8s} {'Rank-1':>8s}")
    print("-" * 78)
    for r in results:
        print(f"{r['model']:25s} {r['r2']:8.4f} ${r['rmse']:7.2f} ${r['mae']:7.2f} "
              f"{r['corr']:8.4f} {r['rank_acc']:7.1%}")
    flush()

    best_rank = max(results, key=lambda r: r["rank_acc"])
    best_r2 = max(results, key=lambda r: r["r2"])
    print(f"\nBest by rank accuracy: {best_rank['model']} ({best_rank['rank_acc']:.1%})")
    print(f"Best by R2:           {best_r2['model']} (R2={best_r2['r2']:.4f})")
    flush()

    # Feature importance for LightGBM models
    for label in ["LightGBM (tuned)", "LGB LambdaRank"]:
        model = trained_models.get(label)
        if model is None:
            continue
        print(f"\n=== Feature Importance: {label} ===")
        imp = pd.DataFrame({
            "feature": feat_cols,
            "importance": model.feature_importance(importance_type="gain"),
        }).sort_values("importance", ascending=False)
        for _, row in imp.iterrows():
            bar = "#" * int(row["importance"] / imp["importance"].max() * 40)
            print(f"  {row['feature']:35s} {row['importance']:>12.0f}  {bar}")
    flush()

    # Save models
    lgb_tuned = trained_models.get("LightGBM (tuned)")
    lgb_rank = trained_models.get("LGB LambdaRank")

    if lgb_tuned:
        joblib.dump(lgb_tuned, model_dir / "profit_model_v2.pkl")
        imp_reg = pd.DataFrame({
            "feature": feat_cols,
            "importance": lgb_tuned.feature_importance(importance_type="gain"),
        }).sort_values("importance", ascending=False)
        imp_reg.to_csv(model_dir / "feature_importance_v2.csv", index=False)
        print(f"\n  Regression model saved:  {model_dir / 'profit_model_v2.pkl'}")

    if lgb_rank:
        joblib.dump(lgb_rank, model_dir / "profit_model_v2_rank.pkl")
        imp_rank = pd.DataFrame({
            "feature": feat_cols,
            "importance": lgb_rank.feature_importance(importance_type="gain"),
        }).sort_values("importance", ascending=False)
        imp_rank.to_csv(model_dir / "feature_importance_v2_rank.csv", index=False)
        print(f"  LambdaRank model saved:  {model_dir / 'profit_model_v2_rank.pkl'}")

    pd.DataFrame(results).to_csv(results_dir / "model_comparison.csv", index=False)
    print(f"  Comparison saved: {results_dir / 'model_comparison.csv'}")
    flush()


if __name__ == "__main__":
    main()
