"""
General-purpose XGBoost training with optional distributed mode via Ray Train.

Single-node mode (default, --num-workers 1):
  Standard xgb.train() with all CPU cores via OpenMP. Best when data fits in
  memory on one machine — no communication overhead.

Distributed mode (--num-workers > 1):
  Uses Ray Train DataParallelTrainer. Each worker gets a data shard, builds a
  local DMatrix, and calls xgb.train(). Ray handles Rabit coordinator setup
  so workers communicate gradients/histograms automatically. No xgboost_ray
  dependency — works with current XGBoost 2.x/3.x.

Supports binary classification, multiclass classification, and regression.

Usage:
    # Single-node binary classification (default)
    python train_xgboost_dist.py --input features/ --no-header

    # Distributed with 4 workers
    python train_xgboost_dist.py --input features/ --no-header --num-workers 4

    # Regression
    python train_xgboost_dist.py --input features/ --no-header \
        --objective reg:squarederror --eval-metric rmse

    # Multiclass
    python train_xgboost_dist.py --input features/ --no-header \
        --objective multi:softprob --num-class 5 --num-workers 4

    # GPU
    python train_xgboost_dist.py --input features/ --no-header --use-gpu
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import ray
import ray.data
import xgboost as xgb
import sklearn.metrics as skm


TARGET_COL = "target"

# Objective → task type mapping
_BINARY_OBJECTIVES = {"binary:logistic", "binary:logitraw", "binary:hinge"}
_MULTI_OBJECTIVES = {"multi:softmax", "multi:softprob"}
_RANK_OBJECTIVES = {"rank:pairwise", "rank:ndcg", "rank:map"}

# Metrics where higher is better (for checkpoint selection)
_MAXIMIZE_METRICS = {"auc", "map", "ndcg", "aucpr", "pre"}


def _infer_task_type(objective):
    """Return 'binary', 'multiclass', or 'regression' from the objective string."""
    if objective in _BINARY_OBJECTIVES:
        return "binary"
    if objective in _MULTI_OBJECTIVES:
        return "multiclass"
    if objective in _RANK_OBJECTIVES:
        return "ranking"
    return "regression"


# ---------------------------------------------------------------------------
# Task-specific evaluation
# ---------------------------------------------------------------------------

def _evaluate_binary(y_val, pred_proba, threshold):
    """Evaluate binary classification: AUC, precision, recall, F1, confusion matrix."""
    pred_labels = (pred_proba >= threshold).astype(np.int32)

    auc_roc = float(skm.roc_auc_score(y_val, pred_proba))
    auc_pr = float(skm.average_precision_score(y_val, pred_proba))
    acc = float(skm.accuracy_score(y_val, pred_labels))
    prec = float(skm.precision_score(y_val, pred_labels, zero_division=0))
    rec = float(skm.recall_score(y_val, pred_labels, zero_division=0))
    f1 = float(skm.f1_score(y_val, pred_labels, zero_division=0))
    tn, fp, fn, tp = skm.confusion_matrix(y_val, pred_labels).ravel()

    print(f"\n  {'='*50}")
    print(f"  BINARY CLASSIFICATION METRICS (threshold={threshold})")
    print(f"  {'='*50}")
    print(f"  AUC-ROC:   {auc_roc:.6f}")
    print(f"  AUC-PR:    {auc_pr:.6f}")
    print(f"  Accuracy:  {acc:.6f}")
    print(f"  Precision: {prec:.6f}")
    print(f"  Recall:    {rec:.6f}")
    print(f"  F1:        {f1:.6f}")
    print(f"  {'='*50}")
    print(f"  Confusion Matrix:")
    print(f"    TN={tn:>10,}   FP={fp:>10,}")
    print(f"    FN={fn:>10,}   TP={tp:>10,}")
    print(f"  {'='*50}")

    return {
        "task_type": "binary",
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "threshold": threshold,
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
    }


def _evaluate_multiclass(y_val, pred_proba):
    """Evaluate multiclass: accuracy, macro/weighted F1, per-class report."""
    pred_labels = np.argmax(pred_proba, axis=1).astype(np.int32)

    acc = float(skm.accuracy_score(y_val, pred_labels))
    f1_macro = float(skm.f1_score(y_val, pred_labels, average="macro", zero_division=0))
    f1_weighted = float(skm.f1_score(y_val, pred_labels, average="weighted", zero_division=0))

    print(f"\n  {'='*50}")
    print(f"  MULTICLASS METRICS")
    print(f"  {'='*50}")
    print(f"  Accuracy:    {acc:.6f}")
    print(f"  F1 (macro):  {f1_macro:.6f}")
    print(f"  F1 (weight): {f1_weighted:.6f}")
    print(f"  {'='*50}")
    print(f"\n  Classification Report:")
    print(skm.classification_report(y_val, pred_labels, zero_division=0))

    return {
        "task_type": "multiclass",
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def _evaluate_regression(y_val, preds):
    """Evaluate regression: RMSE, MAE, R²."""
    rmse = float(np.sqrt(skm.mean_squared_error(y_val, preds)))
    mae = float(skm.mean_absolute_error(y_val, preds))
    r2 = float(skm.r2_score(y_val, preds))

    print(f"\n  {'='*50}")
    print(f"  REGRESSION METRICS")
    print(f"  {'='*50}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R²:   {r2:.6f}")
    print(f"  {'='*50}")

    return {
        "task_type": "regression",
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


# ---------------------------------------------------------------------------
# Data loading — returns Ray Datasets (not pandas)
# ---------------------------------------------------------------------------

def _is_parquet(path):
    """Check whether path points to Parquet data."""
    if path.endswith(".parquet"):
        return True
    if os.path.isdir(path):
        entries = os.listdir(path)[:10]
        return any(f.endswith(".parquet") for f in entries)
    return False


def _csv_kwargs(no_header):
    """Build csv read options for headerless files."""
    if no_header:
        import pyarrow.csv as pa_csv
        return {"read_options": pa_csv.ReadOptions(autogenerate_column_names=True)}
    return {}


def _read_path(path, no_header):
    """Read a file or directory into a Ray Dataset."""
    is_parquet = _is_parquet(path)
    if is_parquet:
        return ray.data.read_parquet(path)
    return ray.data.read_csv(path, **_csv_kwargs(no_header))


def load_datasets(train_path, no_header, val_path=None, val_split=0.2, seed=42):
    """Load feature data with Ray Data, return (train_ds, eval_ds, label_col).

    If val_path is provided, train and val are loaded from separate paths.
    Otherwise, val is split from train_path using val_split fraction.

    Returns Ray Datasets — conversion to pandas happens downstream in the
    training path that needs it.
    """
    is_parquet = _is_parquet(train_path)

    # ---- Separate train/val paths provided ----
    if val_path is not None:
        train_ds = _read_path(train_path, no_header)
        eval_ds = _read_path(val_path, no_header)
        print(f"  Separate inputs: train={train_path}, val={val_path}")

        col_names = train_ds.schema().names
        label_col = TARGET_COL if TARGET_COL in col_names else col_names[0]
        return train_ds, eval_ds, label_col

    # ---- Single path — split automatically ----
    csv_kw = _csv_kwargs(no_header) if not is_parquet else {}

    # File-based split for local directories with enough files
    if os.path.isdir(train_path):
        ext = ".parquet" if is_parquet else ".csv"
        files = sorted(glob.glob(os.path.join(train_path, f"*{ext}")))

        if len(files) >= 5:
            rng = np.random.RandomState(seed)
            indices = rng.permutation(len(files))
            split_idx = max(1, int(len(files) * (1 - val_split)))

            train_files = [files[i] for i in indices[:split_idx]]
            val_files = [files[i] for i in indices[split_idx:]]

            if is_parquet:
                train_ds = ray.data.read_parquet(train_files)
                eval_ds = ray.data.read_parquet(val_files)
            else:
                train_ds = ray.data.read_csv(train_files, **csv_kw)
                eval_ds = ray.data.read_csv(val_files, **csv_kw)

            print(f"  File-based split: {len(train_files)} train / "
                  f"{len(val_files)} val files")

            col_names = train_ds.schema().names
            label_col = TARGET_COL if TARGET_COL in col_names else col_names[0]
            return train_ds, eval_ds, label_col

    # Fallback: read everything and split the dataset
    if is_parquet:
        ds = ray.data.read_parquet(train_path)
    else:
        ds = ray.data.read_csv(train_path, **csv_kw)

    train_ds, eval_ds = ds.train_test_split(test_size=val_split, seed=seed)

    col_names = train_ds.schema().names
    label_col = TARGET_COL if TARGET_COL in col_names else col_names[0]
    return train_ds, eval_ds, label_col


# ---------------------------------------------------------------------------
# Single-node training path
# ---------------------------------------------------------------------------

def _train_single_node(train_ds, eval_ds, label_col, params,
                       num_rounds, early_stopping, verbose_eval):
    """Standard single-node XGBoost training via xgb.train()."""
    train_df = train_ds.to_pandas()
    eval_df = eval_ds.to_pandas()

    y_train = train_df[label_col].values
    y_eval = eval_df[label_col].values

    dtrain = xgb.DMatrix(train_df.drop(columns=[label_col]), label=y_train)
    deval = xgb.DMatrix(eval_df.drop(columns=[label_col]), label=y_eval)

    del train_df, eval_df

    nthread = params.get("nthread", os.cpu_count())
    device = "GPU (cuda)" if params.get("device") == "cuda" else f"CPU ({nthread} threads)"
    print(f"  Mode:       single-node")
    print(f"  Device:     {device}")
    print(f"  Rounds:     {num_rounds} (early stop: {early_stopping})")
    print(f"  Max depth:  {params['max_depth']}  LR: {params['learning_rate']}")
    print()

    evals_result = {}
    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        evals=[(dtrain, "train"), (deval, "eval")],
        evals_result=evals_result,
        num_boost_round=num_rounds,
        early_stopping_rounds=early_stopping,
        verbose_eval=verbose_eval,
    )

    return bst, evals_result, y_train, y_eval, deval


# ---------------------------------------------------------------------------
# Distributed training path — DataParallelTrainer
# ---------------------------------------------------------------------------

class _ShardIter(xgb.DataIter):
    """Feeds Ray Data shard batches into QuantileDMatrix without full materialization.

    XGBoost makes exactly 2 passes over the iterator:
      Pass 1 — sketch (quantile summary)
      Pass 2 — compressed histogram data
    Each batch is loaded, consumed, and freed, so peak memory ≈ 1 batch.
    """

    def __init__(self, shard, label_col, batch_size=65536):
        self._shard = shard
        self._label_col = label_col
        self._batch_size = batch_size
        self._it = None
        super().__init__(cache_prefix=None)

    def next(self, input_data):
        if self._it is None:
            self._it = iter(self._shard.iter_batches(
                batch_format="pandas", batch_size=self._batch_size,
            ))
        try:
            batch = next(self._it)
            input_data(
                data=batch.drop(columns=[self._label_col]),
                label=batch[self._label_col],
            )
            return 1
        except StopIteration:
            return 0

    def reset(self):
        self._it = None


def _distributed_train_func(config):
    """Training function executed on each DataParallelTrainer worker.

    Each worker:
      1. Receives an auto-sharded slice of the training data
      2. Builds a QuantileDMatrix via streaming iterator (no full materialization)
      3. Builds eval DMatrix from numpy chunks (smaller than train)
      4. Calls native xgb.train()
      5. RayTrainReportCallback reports metrics + checkpoints back to Ray Train
      6. Ray handles Rabit coordination so workers share gradients
    """
    import os
    import numpy as np
    import xgboost as xgb
    import ray.train
    from ray.train.xgboost import RayTrainReportCallback

    if config["params"].get("device") == "cuda":
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_SHM_DISABLE", "1")

    train_shard = ray.train.get_dataset_shard("train")
    eval_shard = ray.train.get_dataset_shard("eval")

    label_col = config["label_col"]

    # Train: QuantileDMatrix from streaming iterator — peak mem ≈ 1 batch
    dtrain = xgb.QuantileDMatrix(
        _ShardIter(train_shard, label_col), max_bin=256,
    )

    # Eval: accumulate pandas chunks then build standard DMatrix.
    # Using pandas (not numpy) preserves feature names, which XGBoost 2.x+
    # requires to match between train (QuantileDMatrix) and eval DMatrix.
    import pandas as pd
    eval_frames = []
    for batch in eval_shard.iter_batches(batch_format="pandas", batch_size=65536):
        eval_frames.append(batch)
    eval_df = pd.concat(eval_frames)
    del eval_frames
    y_eval = eval_df[label_col].values
    deval = xgb.DMatrix(eval_df.drop(columns=[label_col]), label=y_eval)
    del eval_df, y_eval

    bst = xgb.train(
        config["params"],
        dtrain=dtrain,
        evals=[(deval, "eval")],
        num_boost_round=config["num_rounds"],
        early_stopping_rounds=config["early_stopping"],
        verbose_eval=config.get("verbose_eval", 10),
        callbacks=[RayTrainReportCallback()],
    )


def _train_distributed(train_ds, eval_ds, label_col, params,
                       num_rounds, early_stopping, verbose_eval,
                       num_workers, cpus_per_worker, use_gpu=False,
                       storage_path=None):
    """Distributed XGBoost training via Ray Train V2 XGBoostTrainer."""
    from ray.train import ScalingConfig, RunConfig, CheckpointConfig, DataConfig
    from ray.train.xgboost import XGBoostTrainer

    # Determine checkpoint scoring — keep best model by primary eval metric
    primary_metric = params["eval_metric"][0]
    score_attr = f"eval-{primary_metric}"
    score_order = "max" if primary_metric in _MAXIMIZE_METRICS else "min"

    config = {
        "label_col": label_col,
        "params": {**params, "nthread": cpus_per_worker},
        "num_rounds": num_rounds,
        "early_stopping": early_stopping,
        "verbose_eval": verbose_eval,
    }

    # GPU: use_gpu=True lets Ray assign 1 GPU + 1 CPU per worker automatically.
    # Do NOT set resources_per_worker — it overrides the GPU allocation.
    # CPU: explicitly allocate CPUs per worker for OpenMP parallelism.
    if use_gpu:
        scaling = ScalingConfig(
            num_workers=num_workers,
            use_gpu=True,
        )
    else:
        scaling = ScalingConfig(
            num_workers=num_workers,
            resources_per_worker={"CPU": cpus_per_worker},
        )

    # Multi-node requires shared storage (S3 or NFS) for checkpoints.
    # Single-node can use local filesystem (default).
    run_config_kwargs = {
        "name": "xgboost-distributed",
        "checkpoint_config": CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute=score_attr,
            checkpoint_score_order=score_order,
        ),
    }
    if storage_path:
        run_config_kwargs["storage_path"] = storage_path

    trainer = XGBoostTrainer(
        _distributed_train_func,
        train_loop_config=config,
        scaling_config=scaling,
        datasets={"train": train_ds, "eval": eval_ds},
        dataset_config=DataConfig(
            datasets_to_split=["train", "eval"],
        ),
        run_config=RunConfig(**run_config_kwargs),
    )

    print(f"  Mode:          distributed (XGBoostTrainer)")
    print(f"  Workers:       {num_workers}")
    if use_gpu:
        print(f"  GPUs/worker:   1")
        print(f"  Device:        GPU (cuda)")
    else:
        print(f"  CPUs/worker:   {cpus_per_worker}")
        print(f"  Total CPUs:    {num_workers * cpus_per_worker}")
        print(f"  Device:        CPU")
    print(f"  Rounds:        {num_rounds} (early stop: {early_stopping})")
    print(f"  Max depth:     {params['max_depth']}  LR: {params['learning_rate']}")
    print(f"  Checkpoint on: {score_attr} ({score_order})")
    print()

    result = trainer.fit()

    # Extract trained model from best checkpoint
    bst = xgb.Booster()
    with result.checkpoint.as_directory() as checkpoint_dir:
        # RayTrainReportCallback saves as model.ubj
        model_file = os.path.join(checkpoint_dir, "model.ubj")
        if not os.path.exists(model_file):
            # Fallback: find any model file in the checkpoint
            for f in os.listdir(checkpoint_dir):
                if f.endswith((".ubj", ".json", ".bin")):
                    model_file = os.path.join(checkpoint_dir, f)
                    break
        bst.load_model(model_file)

    return bst, result.metrics


# ---------------------------------------------------------------------------
# Main training orchestrator
# ---------------------------------------------------------------------------

def run_training(
    input_path: str,
    val_path: str = None,
    model_dir: str = "model",
    no_header: bool = False,
    val_split: float = 0.2,
    use_gpu: bool = False,
    objective: str = "binary:logistic",
    eval_metric: list = None,
    tree_method: str = "hist",
    num_class: int = None,
    num_rounds: int = 500,
    early_stopping: int = 20,
    max_depth: int = 7,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: int = 5,
    threshold: float = 0.5,
    verbose_eval: int = 10,
    seed: int = 42,
    num_workers: int = 1,
    storage_path: str = None,
    class_weight_sample_rows: int = -1,
) -> dict:
    """Train XGBoost and return evaluation metrics.

    If val_path is provided, uses it as the validation set directly.
    Otherwise, splits input_path into train/val using val_split fraction.

    When num_workers=1, uses single-node xgb.train() (OpenMP multi-threaded).
    When num_workers>1, uses DataParallelTrainer for distributed training.
    """
    total_start = time.time()
    task_type = _infer_task_type(objective)
    distributed = num_workers > 1

    if eval_metric is None:
        if task_type == "binary":
            eval_metric = ["logloss", "auc"]
        elif task_type == "multiclass":
            eval_metric = ["mlogloss"]
        else:
            eval_metric = ["rmse"]

    # ---- Load data ----
    print("[1/4] Loading data (Ray Data parallel I/O)...")
    step_start = time.time()
    train_ds, eval_ds, label_col = load_datasets(
        input_path, no_header, val_path=val_path,
        val_split=val_split, seed=seed,
    )

    n_features = len(train_ds.schema().names) - 1

    # Row counts: defer to post-training arrays (distributed) or in-memory
    # data (single-node). Avoids an expensive full-dataset count() pass.
    train_count = None
    eval_count = None

    print(f"  Features: {n_features}")
    print(f"  Label:    {label_col}")
    print(f"  Loaded in {time.time() - step_start:.1f}s")

    # ---- Prepare params ----
    mode_str = "distributed" if distributed else "single-node"
    print(f"\n[2/4] Preparing training ({task_type}, {mode_str})...")

    # Class weight (binary only) — sample-based to avoid full materialization.
    # Default (-1) streams up to 10M rows; user can override via --class-weight-sample-rows.
    scale_pos_weight = None
    if task_type == "binary":
        _sample_pos, _sample_neg = 0, 0
        _SAMPLE_LIMIT = 10_000_000 if class_weight_sample_rows < 0 else class_weight_sample_rows
        for batch in train_ds.select_columns([label_col]).iter_batches(
            batch_format="numpy", batch_size=65536,
        ):
            col = batch[label_col]
            _sample_pos += int((col == 1).sum())
            _sample_neg += int((col == 0).sum())
            if _sample_pos + _sample_neg >= _SAMPLE_LIMIT:
                break
        scale_pos_weight = _sample_neg / max(_sample_pos, 1)
        print(f"  Class balance (sampled {_sample_pos + _sample_neg:,} rows): "
              f"{_sample_neg:,} neg / {_sample_pos:,} pos (weight={scale_pos_weight:.2f})")

    nthread = os.cpu_count()
    params = {
        "objective": objective,
        "eval_metric": eval_metric,
        "tree_method": tree_method,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "min_child_weight": min_child_weight,
        "nthread": nthread,
        "seed": seed,
    }

    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight

    if task_type == "multiclass":
        if num_class is None:
            # Stream batches to find max label — avoids full .to_pandas()
            _label_max = 0
            for batch in train_ds.select_columns([label_col]).iter_batches(
                batch_format="numpy", batch_size=65536,
            ):
                _label_max = max(_label_max, int(batch[label_col].max()))
            num_class = _label_max + 1
        params["num_class"] = num_class

    if use_gpu:
        params["device"] = "cuda"

    # ---- Train ----
    print(f"\n[3/4] Training XGBoost...")
    step_start = time.time()

    if distributed:
        cluster_res = ray.cluster_resources()
        total_cpus = int(cluster_res.get("CPU", nthread))

        if use_gpu:
            total_gpus = int(cluster_res.get("GPU", 0))
            if total_gpus == 0:
                raise RuntimeError(
                    "--use-gpu was set but Ray cluster reports 0 GPUs. "
                    "Check that GPU instances are provisioned and CUDA is available."
                )
            if num_workers > total_gpus:
                print(f"  Warning: num_workers={num_workers} > available GPUs={total_gpus}, "
                      f"capping to {total_gpus}")
                num_workers = total_gpus
            # GPU workers: XGBoost computes on GPU, CPUs only for data prep.
            # ScalingConfig(use_gpu=True) assigns 1 CPU + 1 GPU per worker by default.
            cpus_per_worker = 1
        else:
            # Reserve 20% of CPUs for Ray Data I/O and preprocessing.
            # Ray docs: "Allocate N CPUs per worker, leaving resources for Ray Data."
            usable_cpus = max(num_workers, int(total_cpus * 0.80))
            cpus_per_worker = max(1, usable_cpus // num_workers)
            reserved = total_cpus - (num_workers * cpus_per_worker)
            print(f"  CPU budget:    {total_cpus} total, {num_workers * cpus_per_worker} "
                  f"for training, {reserved} reserved for Ray Data")

        bst, dist_metrics = _train_distributed(
            train_ds, eval_ds, label_col, params,
            num_rounds, early_stopping, verbose_eval,
            num_workers, cpus_per_worker, use_gpu=use_gpu,
            storage_path=storage_path,
        )
        # Model loaded from checkpoint — use all trees it contains
        best_round = bst.num_boosted_rounds() - 1
    else:
        bst, evals_result, y_train_arr, y_eval_arr, deval_single = (
            _train_single_node(
                train_ds, eval_ds, label_col, params,
                num_rounds, early_stopping, verbose_eval,
            )
        )
        best_round = bst.best_iteration

    train_time = time.time() - step_start

    # Best metric value
    primary_metric = eval_metric[0]
    if not distributed:
        best_metric_val = evals_result["eval"][primary_metric][best_round]
    else:
        best_metric_val = dist_metrics.get(f"eval-{primary_metric}")

    print(f"\n  Training done in {train_time:.1f}s")
    print(f"  Best round:         {best_round}")
    if best_metric_val is not None:
        print(f"  Best val {primary_metric}: {best_metric_val:.6f}")

    # ---- Evaluate on full validation set ----
    print(f"\n[4/4] Evaluating on validation set...")
    step_start = time.time()

    if distributed:
        # Batched prediction: iterate eval set in chunks to avoid full
        # materialization. Each batch builds a temporary DMatrix, predicts,
        # and frees — peak memory ≈ 1 batch + accumulated pred arrays.
        all_preds, all_labels = [], []
        for batch in eval_ds.iter_batches(batch_format="pandas", batch_size=65536):
            y_batch = batch[label_col].values
            X_batch = batch.drop(columns=[label_col])
            d_batch = xgb.DMatrix(X_batch)
            all_preds.append(bst.predict(d_batch, iteration_range=(0, best_round + 1)))
            all_labels.append(y_batch)
            del d_batch, X_batch
        y_eval = np.concatenate(all_labels)
        preds = np.concatenate(all_preds)
        del all_preds, all_labels
        eval_count = len(y_eval)
    else:
        y_eval = y_eval_arr
        deval = deval_single
        preds = bst.predict(deval, iteration_range=(0, best_round + 1))
        train_count = len(y_train_arr)
        eval_count = len(y_eval)

    if task_type == "binary":
        eval_metrics = _evaluate_binary(y_eval, preds, threshold)
    elif task_type == "multiclass":
        eval_metrics = _evaluate_multiclass(y_eval, preds)
    else:
        eval_metrics = _evaluate_regression(y_eval, preds)

    print(f"  Evaluated in {time.time() - step_start:.1f}s")

    # ---- Save artifacts ----
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.ubj")
    bst.save_model(model_path)
    bst.save_model(os.path.join(model_dir, "model.json"))

    # Feature importance
    importance = bst.get_score(importance_type="gain")
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    with open(os.path.join(model_dir, "feature_importance.json"), "w") as f:
        json.dump(importance, f, indent=2)

    # Metrics
    metrics = {
        **eval_metrics,
        "best_round": int(best_round),
        "train_rows": train_count if train_count is not None else "unknown",
        "val_rows": eval_count,
        "n_features": n_features,
        "distributed": distributed,
        "num_workers": num_workers if distributed else 1,
        "training_time_s": round(train_time, 1),
        "total_time_s": round(time.time() - total_start, 1),
    }
    if best_metric_val is not None:
        metrics[f"best_val_{primary_metric}"] = float(best_metric_val)

    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Params (for reproducibility)
    saved_params = {
        **params,
        "num_boost_round": num_rounds,
        "early_stopping_rounds": early_stopping,
        "actual_rounds": best_round + 1,
        "use_gpu": use_gpu,
        "num_workers": num_workers,
    }
    with open(os.path.join(model_dir, "params.json"), "w") as f:
        json.dump(saved_params, f, indent=2)

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE ({task_type}, {mode_str})")
    print(f"{'='*60}")
    print(f"  Total time:  {total_time:.1f}s")
    print(f"  Model:       {model_path}")
    print(f"  Best round:  {best_round}")
    if task_type == "binary":
        print(f"  AUC-ROC:     {eval_metrics['auc_roc']:.6f}")
    elif task_type == "multiclass":
        print(f"  Accuracy:    {eval_metrics['accuracy']:.6f}")
        print(f"  F1 (macro):  {eval_metrics['f1_macro']:.6f}")
    else:
        print(f"  RMSE:        {eval_metrics['rmse']:.6f}")
        print(f"  R²:          {eval_metrics['r2']:.6f}")
    print(f"{'='*60}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="XGBoost training with optional distributed mode (Ray Train)"
    )
    # Data
    p.add_argument("--input", required=True,
                   help="Training data path (directory or file)")
    p.add_argument("--val-input", default=None,
                   help="Validation data path. If omitted, splits from --input using --val-split")
    p.add_argument("--model-dir", default="model",
                   help="Output dir for model artifacts (default: model/)")
    p.add_argument("--no-header", action="store_true",
                   help="Input CSV has no header (target is first column)")
    p.add_argument("--val-split", type=float, default=0.2,
                   help="Validation fraction when --val-input not provided (default: 0.2)")

    # XGBoost objective / task
    p.add_argument("--objective", default="binary:logistic",
                   help="XGBoost objective (default: binary:logistic)")
    p.add_argument("--eval-metric", nargs="+", default=None,
                   help="Eval metric(s) (default: auto per task type)")
    p.add_argument("--tree-method", default="hist",
                   help="Tree method (default: hist)")
    p.add_argument("--num-class", type=int, default=None,
                   help="Number of classes (for multiclass objectives)")

    # Hyperparameters
    p.add_argument("--num-rounds", type=int, default=500,
                   help="Max boosting rounds (default: 500)")
    p.add_argument("--early-stopping", type=int, default=20,
                   help="Early stopping patience (default: 20)")
    p.add_argument("--max-depth", type=int, default=7,
                   help="Max tree depth (default: 7)")
    p.add_argument("--learning-rate", type=float, default=0.1,
                   help="Learning rate (default: 0.1)")
    p.add_argument("--subsample", type=float, default=0.8,
                   help="Row subsample per tree (default: 0.8)")
    p.add_argument("--colsample-bytree", type=float, default=0.8,
                   help="Column subsample per tree (default: 0.8)")
    p.add_argument("--min-child-weight", type=int, default=5,
                   help="Min child weight (default: 5)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Binary classification threshold (default: 0.5)")

    # Infrastructure
    p.add_argument("--num-workers", type=int, default=1,
                   help="Number of training workers. 1=single-node, >1=distributed (default: 1)")
    p.add_argument("--use-gpu", action="store_true",
                   help="Use GPU training")
    p.add_argument("--verbose-eval", type=int, default=10,
                   help="Print metrics every N rounds (default: 10)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--ray-address", type=str, default=None,
                   help="Ray cluster address (default: local)")
    p.add_argument("--class-weight-sample-rows", type=int, default=-1,
                   help="Rows to sample for class weight estimation. -1=auto (10M)")

    return p.parse_args()


def main():
    args = parse_args()

    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init()
    print(f"Ray resources: {ray.cluster_resources()}\n")

    run_training(
        input_path=args.input,
        val_path=args.val_input,
        model_dir=args.model_dir,
        no_header=args.no_header,
        val_split=args.val_split,
        use_gpu=args.use_gpu,
        objective=args.objective,
        eval_metric=args.eval_metric,
        tree_method=args.tree_method,
        num_class=args.num_class,
        num_rounds=args.num_rounds,
        early_stopping=args.early_stopping,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        threshold=args.threshold,
        verbose_eval=args.verbose_eval,
        seed=args.seed,
        num_workers=args.num_workers,
        class_weight_sample_rows=args.class_weight_sample_rows,
    )

    ray.shutdown()


if __name__ == "__main__":
    main()
