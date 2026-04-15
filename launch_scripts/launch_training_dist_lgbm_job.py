"""
Launch LightGBM training (single-node or distributed) as a SageMaker Training job.

Usage:
    # Single-node (default)
    python launch_training_dist_lgbm_job.py

    # Distributed with 4 workers on one instance
    python launch_training_dist_lgbm_job.py --num-workers 4

    # Regression task
    python launch_training_dist_lgbm_job.py --objective regression

    # Wait for completion
    python launch_training_dist_lgbm_job.py --wait
"""

import argparse
import sagemaker
from sagemaker.pytorch import PyTorch


S3_INPUT = "s3://sagemaker-us-east-1-734584155256/omf-loan-fake/training-data1/train"

# Regex building block for a floating-point number
_NUM = r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"

# CloudWatch metric definitions for LightGBM — covers binary, multiclass, regression
LGBM_METRIC_DEFINITIONS = [
    # Final evaluation metrics (printed by training script)
    {"Name": "auc-roc",    "Regex": rf"AUC-ROC:\s+{_NUM}"},
    {"Name": "auc-pr",     "Regex": rf"AUC-PR:\s+{_NUM}"},
    {"Name": "f1",         "Regex": rf"F1:\s+{_NUM}"},
    {"Name": "accuracy",   "Regex": rf"Accuracy:\s+{_NUM}"},
    {"Name": "f1-macro",   "Regex": rf"F1 \(macro\):\s+{_NUM}"},
    {"Name": "f1-weighted", "Regex": rf"F1 \(weight\):\s+{_NUM}"},
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
S3_VAL_INPUT = "s3://sagemaker-us-east-1-734584155256/omf-loan-fake/training-data1/val"
S3_OUTPUT_BASE = "s3://sagemaker-us-east-1-734584155256/omf-loan-fake/model/"


def parse_args():
    p = argparse.ArgumentParser(
        description="Launch SageMaker LightGBM training job (distributed)"
    )
    p.add_argument("--instance-type", default="ml.m5.4xlarge",
                   help="Instance type (default: ml.m5.4xlarge)")
    p.add_argument("--instance-count", type=int, default=1,
                   help="Number of instances (default: 1)")
    p.add_argument("--s3-input", default=S3_INPUT,
                   help=f"S3 training data path (default: {S3_INPUT})")
    p.add_argument("--s3-val-input", default=S3_VAL_INPUT,
                   help=f"S3 validation data path (default: {S3_VAL_INPUT})")
    p.add_argument("--s3-output", default=S3_OUTPUT_BASE,
                   help=f"S3 model output (default: {S3_OUTPUT_BASE})")
    p.add_argument("--wait", action="store_true",
                   help="Wait for job to complete")
    p.add_argument("--keep-alive", type=int, default=0,
                   help="Keep instance warm pool alive for N seconds after job (default: 0 = disabled)")

    # Training mode
    p.add_argument("--num-workers", type=int, default=0,
                   help="Number of training workers. 0=auto-detect from cluster topology (default: 0)")
    p.add_argument("--no-header", action="store_true", default=True,
                   help="Input CSV has no header (default: True)")

    # LightGBM task / objective
    p.add_argument("--objective", default="binary",
                   help="LightGBM objective (default: binary)")
    p.add_argument("--num-class", type=int, default=None,
                   help="Number of classes (for multiclass objectives)")

    # Hyperparameters
    p.add_argument("--num-rounds", type=int, default=500)
    p.add_argument("--early-stopping", type=int, default=20)
    p.add_argument("--max-depth", type=int, default=7)
    p.add_argument("--num-leaves", type=int, default=127)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--min-child-samples", type=int, default=20)

    # Ray object store tuning
    p.add_argument("--object-store-memory", type=float, default=-1.0,
                   help="Ray object store size in bytes. -1.0=auto (40%% of system RAM)")
    p.add_argument("--object-store-fraction", type=float, default=-1.0,
                   help="Ray object store as fraction of system RAM. -1.0=auto")

    # Class weight sampling
    p.add_argument("--class-weight-sample-rows", type=int, default=-1,
                   help="Rows to sample for class weight estimation. -1=auto (10M)")
    return p.parse_args()


def main():
    args = parse_args()
    session = sagemaker.Session()
    role = sagemaker.get_execution_role()

    print(f"Role:           {role}")
    print(f"Region:         {session.boto_region_name}")
    print(f"Instance type:  {args.instance_type}")
    print(f"Instance count: {args.instance_count}")
    print(f"Workers:        {args.num_workers}")
    print(f"Train input:    {args.s3_input}")
    print(f"Val input:      {args.s3_val_input or '(split from train)'}")
    print(f"Output:         {args.s3_output}")
    print()

    hyperparameters = {
        "objective": args.objective,
        **({"num-class": args.num_class} if args.num_class else {}),
        "num-workers": args.num_workers,
        "num-rounds": args.num_rounds,
        "early-stopping": args.early_stopping,
        "max-depth": args.max_depth,
        "num-leaves": args.num_leaves,
        "learning-rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample-bytree": args.colsample_bytree,
        "min-child-samples": args.min_child_samples,
    }
    if args.no_header:
        hyperparameters["no-header"] = ""  # SageMaker passes as --no-header flag

    # Object store tuning — pass through to sagemaker_train.py
    if args.object_store_memory > 0:
        hyperparameters["object-store-memory"] = args.object_store_memory
    if args.object_store_fraction > 0:
        hyperparameters["object-store-fraction"] = args.object_store_fraction

    # Class weight sampling — pass through to sagemaker_train.py
    if args.class_weight_sample_rows >= 0:
        hyperparameters["class-weight-sample-rows"] = args.class_weight_sample_rows

    # Multi-node requires shared S3 storage for Ray Train checkpoints
    if args.instance_count > 1:
        checkpoint_s3 = f"s3://{session.default_bucket()}/ray-checkpoints-lgbm"
        hyperparameters["checkpoint-s3-uri"] = checkpoint_s3
        print(f"Checkpoint:     {checkpoint_s3} (multi-node shared storage)")

    environment = {}

    estimator_kwargs = dict(
        entry_point="sagemaker_train.py",
        source_dir="src_dist_training_lgbm",
        role=role,
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        framework_version="2.5.1",
        py_version="py311",
        output_path=args.s3_output,
        sagemaker_session=session,
        base_job_name="loan-lgbm-dist-training",
        hyperparameters=hyperparameters,
        environment=environment,
        metric_definitions=LGBM_METRIC_DEFINITIONS,
    )
    if args.keep_alive > 0:
        estimator_kwargs["keep_alive_period_in_seconds"] = args.keep_alive
        print(f"Keep-alive:     {args.keep_alive}s warm pool")

    estimator = PyTorch(**estimator_kwargs)

    # Build input channels — "validation" is optional
    channels = {"training": args.s3_input}
    if args.s3_val_input:
        channels["validation"] = args.s3_val_input

    print("Launching training job...")
    estimator.fit(
        channels,
        wait=args.wait,
        logs=args.wait,
    )

    if args.wait:
        print(f"\nJob complete. Model at: {args.s3_output}")
    else:
        job_name = estimator.latest_training_job.name
        print(f"\nJob submitted: {job_name}")
        print(f"Monitor at: https://{session.boto_region_name}.console.aws.amazon.com/"
              f"sagemaker/home?region={session.boto_region_name}"
              f"#/jobs/{job_name}")


if __name__ == "__main__":
    main()
