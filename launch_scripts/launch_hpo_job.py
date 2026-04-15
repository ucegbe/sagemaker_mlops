"""
Launch multi-algorithm hyperparameter tuning job for XGBoost and LightGBM.

Both algorithms run as Ray-based training inside a PyTorch container on SageMaker.
The tuner explores hyperparameter space for each algorithm independently and
ranks all trials by the objective metric to find the best model.

Usage:
    # Default: 20 total jobs, 4 parallel, Bayesian strategy, binary classification
    python launch_hpo_job.py

    # More exploration
    python launch_hpo_job.py --max-jobs 40 --max-parallel-jobs 5

    # Regression task (minimize RMSE)
    python launch_hpo_job.py --objective-metric rmse --objective-type Minimize \
        --xgb-objective reg:squarederror --lgbm-objective regression

    # Multiclass
    python launch_hpo_job.py --objective-metric accuracy --objective-type Maximize \
        --xgb-objective multi:softmax --xgb-num-class 5 \
        --lgbm-objective multiclass --lgbm-num-class 5

    # Random search, wait for completion
    python launch_hpo_job.py --strategy Random --wait
"""

import argparse
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import (
    HyperparameterTuner,
    ContinuousParameter,
    IntegerParameter,
)


S3_INPUT = "s3://sagemaker-us-east-1-734584155256/omf-loan-fake/training-data1/train"
S3_VAL_INPUT = "s3://sagemaker-us-east-1-734584155256/omf-loan-fake/training-data1/val"
S3_OUTPUT_BASE = "s3://sagemaker-us-east-1-734584155256/omf-loan-fake/model/"

# Regex building block for a floating-point number
_NUM = r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"


def parse_args():
    p = argparse.ArgumentParser(
        description="Launch multi-algorithm HPO job (XGBoost + LightGBM)"
    )
    # Infrastructure
    p.add_argument("--instance-type", default="ml.m5.4xlarge",
                   help="Instance type for all training jobs (default: ml.m5.4xlarge)")
    p.add_argument("--s3-input", default=S3_INPUT,
                   help=f"S3 training data path (default: {S3_INPUT})")
    p.add_argument("--s3-val-input", default=S3_VAL_INPUT,
                   help=f"S3 validation data path (default: {S3_VAL_INPUT})")
    p.add_argument("--s3-output", default=S3_OUTPUT_BASE,
                   help=f"S3 model output base (default: {S3_OUTPUT_BASE})")
    p.add_argument("--wait", action="store_true",
                   help="Wait for HPO job to complete")

    # HPO strategy
    p.add_argument("--max-jobs", type=int, default=20,
                   help="Maximum total training jobs across both algorithms (default: 20)")
    p.add_argument("--max-parallel-jobs", type=int, default=4,
                   help="Maximum concurrent training jobs (default: 4)")
    p.add_argument("--strategy", default="Bayesian",
                   choices=["Bayesian", "Random"],
                   help="HPO search strategy (default: Bayesian)")
    p.add_argument("--objective-metric", default="auc-roc",
                   help="Objective metric name to optimize (default: auc-roc)")
    p.add_argument("--objective-type", default="Maximize",
                   choices=["Maximize", "Minimize"],
                   help="Optimization direction (default: Maximize)")

    # Shared training settings (static, not tuned)
    p.add_argument("--num-rounds", type=int, default=500,
                   help="Max boosting rounds for both algorithms (default: 500)")
    p.add_argument("--early-stopping", type=int, default=20,
                   help="Early stopping patience for both algorithms (default: 20)")

    # Per-algorithm objective overrides
    p.add_argument("--xgb-objective", default="binary:logistic",
                   help="XGBoost objective (default: binary:logistic)")
    p.add_argument("--xgb-num-class", type=int, default=None,
                   help="XGBoost num_class (for multiclass objectives)")
    p.add_argument("--lgbm-objective", default="binary",
                   help="LightGBM objective (default: binary)")
    p.add_argument("--lgbm-num-class", type=int, default=None,
                   help="LightGBM num_class (for multiclass objectives)")
    return p.parse_args()


def _build_xgb_metrics():
    """Metric definitions for XGBoost covering binary, multiclass, and regression.

    XGBoost per-round format: eval-METRIC:VALUE  (e.g. eval-auc:0.681)
    Final evaluation format:  METRIC:   VALUE     (e.g. AUC-ROC:   0.681)
    """
    return [
        # Final evaluation metrics (printed by our training script)
        # Binary
        {"Name": "auc-roc",    "Regex": rf"AUC-ROC:\s+{_NUM}"},
        {"Name": "auc-pr",     "Regex": rf"AUC-PR:\s+{_NUM}"},
        {"Name": "f1",         "Regex": rf"F1:\s+{_NUM}"},
        # Multiclass
        {"Name": "accuracy",   "Regex": rf"Accuracy:\s+{_NUM}"},
        {"Name": "f1-macro",   "Regex": rf"F1 \(macro\):\s+{_NUM}"},
        {"Name": "f1-weighted", "Regex": rf"F1 \(weight\):\s+{_NUM}"},
        # Regression
        {"Name": "rmse",       "Regex": rf"RMSE:\s+{_NUM}"},
        {"Name": "mae",        "Regex": rf"MAE:\s+{_NUM}"},
        {"Name": "r2",         "Regex": rf"R..?:\s+{_NUM}"},  # matches R² or R2
        # XGBoost per-round metrics (from xgb.train verbose output)
        {"Name": "xgb:eval-auc",      "Regex": rf"eval-auc:{_NUM}"},
        {"Name": "xgb:eval-logloss",  "Regex": rf"eval-logloss:{_NUM}"},
        {"Name": "xgb:eval-mlogloss", "Regex": rf"eval-mlogloss:{_NUM}"},
        {"Name": "xgb:eval-rmse",     "Regex": rf"eval-rmse:{_NUM}"},
        {"Name": "xgb:eval-error",    "Regex": rf"eval-error:{_NUM}"},
        {"Name": "xgb:eval-merror",   "Regex": rf"eval-merror:{_NUM}"},
    ]


def _build_lgbm_metrics():
    """Metric definitions for LightGBM covering binary, multiclass, and regression.

    LightGBM per-round format: eval's METRIC: VALUE  (e.g. eval's auc: 0.681)
    Final evaluation format:   METRIC:   VALUE        (e.g. AUC-ROC:   0.681)
    """
    return [
        # Final evaluation metrics (printed by our training script — same format as XGBoost)
        # Binary
        {"Name": "auc-roc",    "Regex": rf"AUC-ROC:\s+{_NUM}"},
        {"Name": "auc-pr",     "Regex": rf"AUC-PR:\s+{_NUM}"},
        {"Name": "f1",         "Regex": rf"F1:\s+{_NUM}"},
        # Multiclass
        {"Name": "accuracy",   "Regex": rf"Accuracy:\s+{_NUM}"},
        {"Name": "f1-macro",   "Regex": rf"F1 \(macro\):\s+{_NUM}"},
        {"Name": "f1-weighted", "Regex": rf"F1 \(weight\):\s+{_NUM}"},
        # Regression
        {"Name": "rmse",       "Regex": rf"RMSE:\s+{_NUM}"},
        {"Name": "mae",        "Regex": rf"MAE:\s+{_NUM}"},
        {"Name": "r2",         "Regex": rf"R2:\s+{_NUM}"},
        # LightGBM per-round metrics (from lgb.log_evaluation callback)
        {"Name": "lgbm:eval-auc",         "Regex": rf"eval's auc: {_NUM}"},
        {"Name": "lgbm:eval-logloss",     "Regex": rf"eval's binary_logloss: {_NUM}"},
        {"Name": "lgbm:eval-mlogloss",    "Regex": rf"eval's multi_logloss: {_NUM}"},
        {"Name": "lgbm:eval-rmse",        "Regex": rf"eval's rmse: {_NUM}"},
        {"Name": "lgbm:eval-l2",          "Regex": rf"eval's l2: {_NUM}"},
        {"Name": "lgbm:eval-multi-error", "Regex": rf"eval's multi_error: {_NUM}"},
    ]


def main():
    args = parse_args()
    session = sagemaker.Session()
    role = sagemaker.get_execution_role()

    print(f"Role:             {role}")
    print(f"Region:           {session.boto_region_name}")
    print(f"Instance type:    {args.instance_type}")
    print(f"Strategy:         {args.strategy}")
    print(f"Max jobs:         {args.max_jobs}")
    print(f"Max parallel:     {args.max_parallel_jobs}")
    print(f"Objective:        {args.objective_type} {args.objective_metric}")
    print(f"Boosting rounds:  {args.num_rounds} (early stop: {args.early_stopping})")
    print(f"Train input:      {args.s3_input}")
    print(f"Val input:        {args.s3_val_input}")
    print(f"Output:           {args.s3_output}")
    print()

    # ---- XGBoost estimator ----
    xgb_hyperparameters = {
        "objective": args.xgb_objective,
        "tree-method": "hist",
        "num-rounds": args.num_rounds,
        "early-stopping": args.early_stopping,
        "num-workers": 0,
        "no-header": "",
    }
    if args.xgb_num_class is not None:
        xgb_hyperparameters["num-class"] = args.xgb_num_class

    xgb_estimator = PyTorch(
        entry_point="sagemaker_train.py",
        source_dir="src_dist_training",
        role=role,
        instance_count=1,
        instance_type=args.instance_type,
        framework_version="2.5.1",
        py_version="py311",
        output_path=args.s3_output,
        sagemaker_session=session,
        base_job_name="loan-hpo-xgboost",
        hyperparameters=xgb_hyperparameters,
        metric_definitions=_build_xgb_metrics(),
    )

    xgb_hp_ranges = {
        "learning-rate": ContinuousParameter(0.01, 0.3),
        "max-depth": IntegerParameter(3, 10),
        "subsample": ContinuousParameter(0.5, 1.0),
        "colsample-bytree": ContinuousParameter(0.5, 1.0),
        "min-child-weight": IntegerParameter(1, 10),
    }

    # ---- LightGBM estimator ----
    lgbm_hyperparameters = {
        "objective": args.lgbm_objective,
        "num-rounds": args.num_rounds,
        "early-stopping": args.early_stopping,
        "num-workers": 0,
        "no-header": "",
    }
    if args.lgbm_num_class is not None:
        lgbm_hyperparameters["num-class"] = args.lgbm_num_class

    lgbm_estimator = PyTorch(
        entry_point="sagemaker_train.py",
        source_dir="src_dist_training_lgbm",
        role=role,
        instance_count=1,
        instance_type=args.instance_type,
        framework_version="2.5.1",
        py_version="py311",
        output_path=args.s3_output,
        sagemaker_session=session,
        base_job_name="loan-hpo-lightgbm",
        hyperparameters=lgbm_hyperparameters,
        metric_definitions=_build_lgbm_metrics(),
    )

    lgbm_hp_ranges = {
        "learning-rate": ContinuousParameter(0.01, 0.3),
        "max-depth": IntegerParameter(3, 10),
        "num-leaves": IntegerParameter(15, 255),
        "subsample": ContinuousParameter(0.5, 1.0),
        "colsample-bytree": ContinuousParameter(0.5, 1.0),
        "min-child-samples": IntegerParameter(5, 50),
    }

    # ---- Create multi-algorithm tuner ----
    print("Algorithms:")
    print(f"  XGBoost  ({args.xgb_objective}):  5 tunable HPs")
    print(f"  LightGBM ({args.lgbm_objective}): 6 tunable HPs")
    print()

    tuner = HyperparameterTuner.create(
        estimator_dict={
            "xgboost": xgb_estimator,
            "lightgbm": lgbm_estimator,
        },
        objective_metric_name_dict={
            "xgboost": args.objective_metric,
            "lightgbm": args.objective_metric,
        },
        hyperparameter_ranges_dict={
            "xgboost": xgb_hp_ranges,
            "lightgbm": lgbm_hp_ranges,
        },
        metric_definitions_dict={
            "xgboost": _build_xgb_metrics(),
            "lightgbm": _build_lgbm_metrics(),
        },
        objective_type=args.objective_type,
        strategy=args.strategy,
        max_jobs=args.max_jobs,
        max_parallel_jobs=args.max_parallel_jobs,
        base_tuning_job_name="loan-hpo-xgb-lgbm",
    )

    # ---- Input channels (same data for both algorithms) ----
    channels = {"training": args.s3_input}
    if args.s3_val_input:
        channels["validation"] = args.s3_val_input

    inputs = {
        "xgboost": channels,
        "lightgbm": channels,
    }

    print("Launching HPO job...")
    tuner.fit(
        inputs=inputs,
        include_cls_metadata={},
        wait=args.wait,
        logs=args.wait,
    )

    tuning_job_name = tuner.latest_tuning_job.name
    if args.wait:
        print(f"\nHPO job complete: {tuning_job_name}")
    else:
        print(f"\nHPO job submitted: {tuning_job_name}")
    print(f"Monitor at: https://{session.boto_region_name}.console.aws.amazon.com/"
          f"sagemaker/home?region={session.boto_region_name}"
          f"#/hyper-tuning-jobs/{tuning_job_name}")


if __name__ == "__main__":
    main()
