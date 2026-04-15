"""
Config-driven SageMaker Pipeline: Feature Engineering -> Training/Tuning.

Chains a feature engineering processing step with training/tuning steps.
FE output (train/val split) feeds directly into the training input.

Pipeline graph:

    FeatureEngineering (ProcessingStep)
        |
    CheckXgbTraining  --> TrainXGBoost         (algo in [xgboost,both] AND tuning=false)
    CheckLgbmTraining --> TrainLightGBM        (algo in [lightgbm,both] AND tuning=false)
    CheckXgbTuning    --> TuneXGBoost          (algo == xgboost AND tuning=true)
    CheckLgbmTuning   --> TuneLightGBM         (algo == lightgbm AND tuning=true)
    CheckBothTuning   --> TuneBothAlgorithms   (algo == both AND tuning=true)

Usage:
    # Create/update pipeline
    python pipeline_fe_training.py

    # Start with defaults (FE -> train both algorithms)
    python pipeline_fe_training.py --start

    # FE -> XGBoost training only
    python pipeline_fe_training.py --start --params AlgorithmChoice=xgboost

    # FE -> multi-algorithm HPO tuning (both algorithms in one tuner)
    python pipeline_fe_training.py --start --params RunTuning=true

    # FE -> single-algorithm HPO tuning (XGBoost only)
    python pipeline_fe_training.py --start --params RunTuning=true AlgorithmChoice=xgboost

    # Custom FE input and training hyperparameters
    python pipeline_fe_training.py --start --params FeRawInputS3Uri=s3://bucket/data.csv XgbLearningRate=0.05
"""

import argparse
import os

import yaml
import sagemaker
from sagemaker.pytorch import PyTorch, PyTorchProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.tuner import (
    HyperparameterTuner,
    ContinuousParameter,
    IntegerParameter,
)
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.workflow.steps import TrainingStep, TuningStep, ProcessingStep
from sagemaker.workflow.conditions import ConditionEquals, ConditionIn
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import Join
from sagemaker.workflow.pipeline import Pipeline


DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "pipeline_config.yaml"
)


# ── CloudWatch metric definitions ────────────────────────────────────────────

_NUM = r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"

XGB_METRIC_DEFINITIONS = [
    {"Name": "auc-roc",     "Regex": rf"AUC-ROC:\s+{_NUM}"},
    {"Name": "auc-pr",      "Regex": rf"AUC-PR:\s+{_NUM}"},
    {"Name": "f1",          "Regex": rf"F1:\s+{_NUM}"},
    {"Name": "accuracy",    "Regex": rf"Accuracy:\s+{_NUM}"},
    {"Name": "f1-macro",    "Regex": rf"F1 \(macro\):\s+{_NUM}"},
    {"Name": "f1-weighted", "Regex": rf"F1 \(weight\):\s+{_NUM}"},
    {"Name": "rmse",        "Regex": rf"RMSE:\s+{_NUM}"},
    {"Name": "mae",         "Regex": rf"MAE:\s+{_NUM}"},
    {"Name": "r2",          "Regex": rf"R..?:\s+{_NUM}"},
    {"Name": "xgb:eval-auc",      "Regex": rf"eval-auc:{_NUM}"},
    {"Name": "xgb:eval-logloss",  "Regex": rf"eval-logloss:{_NUM}"},
    {"Name": "xgb:eval-mlogloss", "Regex": rf"eval-mlogloss:{_NUM}"},
    {"Name": "xgb:eval-rmse",     "Regex": rf"eval-rmse:{_NUM}"},
    {"Name": "xgb:eval-error",    "Regex": rf"eval-error:{_NUM}"},
    {"Name": "xgb:eval-merror",   "Regex": rf"eval-merror:{_NUM}"},
]

LGBM_METRIC_DEFINITIONS = [
    {"Name": "auc-roc",     "Regex": rf"AUC-ROC:\s+{_NUM}"},
    {"Name": "auc-pr",      "Regex": rf"AUC-PR:\s+{_NUM}"},
    {"Name": "f1",          "Regex": rf"F1:\s+{_NUM}"},
    {"Name": "accuracy",    "Regex": rf"Accuracy:\s+{_NUM}"},
    {"Name": "f1-macro",    "Regex": rf"F1 \(macro\):\s+{_NUM}"},
    {"Name": "f1-weighted", "Regex": rf"F1 \(weight\):\s+{_NUM}"},
    {"Name": "rmse",        "Regex": rf"RMSE:\s+{_NUM}"},
    {"Name": "mae",         "Regex": rf"MAE:\s+{_NUM}"},
    {"Name": "r2",          "Regex": rf"R2:\s+{_NUM}"},
    {"Name": "lgbm:eval-auc",         "Regex": rf"eval's auc: {_NUM}"},
    {"Name": "lgbm:eval-logloss",     "Regex": rf"eval's binary_logloss: {_NUM}"},
    {"Name": "lgbm:eval-mlogloss",    "Regex": rf"eval's multi_logloss: {_NUM}"},
    {"Name": "lgbm:eval-rmse",        "Regex": rf"eval's rmse: {_NUM}"},
    {"Name": "lgbm:eval-l2",          "Regex": rf"eval's l2: {_NUM}"},
    {"Name": "lgbm:eval-multi-error", "Regex": rf"eval's multi_error: {_NUM}"},
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_config(path):
    """Load pipeline configuration from YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def _build_hp_ranges(ranges_config):
    """Convert config dict to SageMaker hyperparameter range objects."""
    ranges = {}
    for name, spec in ranges_config.items():
        if spec["type"] == "continuous":
            ranges[name] = ContinuousParameter(float(spec["min"]), float(spec["max"]))
        elif spec["type"] == "integer":
            ranges[name] = IntegerParameter(int(spec["min"]), int(spec["max"]))
    return ranges


# ── Pipeline construction ────────────────────────────────────────────────────

def create_pipeline(config, role):
    """Build the FE -> Training SageMaker Pipeline from config."""

    cfg_src = config["source"]
    cfg_params = config["parameters"]
    cfg_ranges = config.get("tuning_ranges", {})
    pipeline_name = config["pipeline"].get(
        "fe_training_name", "loan-fe-training-pipeline"
    )

    pipeline_session = PipelineSession()
    default_bucket = pipeline_session.default_bucket()

    # ================================================================
    # Pipeline Parameters
    # ================================================================

    # -- Execution role --
    execution_role = ParameterString(
        name="ExecutionRole", default_value=role,
    )

    # -- Pipeline control --
    algorithm_choice = ParameterString(
        name="AlgorithmChoice",
        default_value=str(cfg_params["AlgorithmChoice"]),
        enum_values=["xgboost", "lightgbm", "both"],
    )
    run_tuning = ParameterString(
        name="RunTuning",
        default_value=str(cfg_params["RunTuning"]),
        enum_values=["true", "false"],
    )

    # -- Feature Engineering parameters --
    fe_raw_input_s3 = ParameterString(
        name="FeRawInputS3Uri",
        default_value=str(cfg_params["FeRawInputS3Uri"]),
    )
    fe_output_s3 = ParameterString(
        name="FeOutputS3Uri",
        default_value=str(cfg_params["FeOutputS3Uri"]),
    )
    fe_instance_type = ParameterString(
        name="FeInstanceType",
        default_value=str(cfg_params["FeInstanceType"]),
    )
    fe_instance_count = ParameterInteger(
        name="FeInstanceCount",
        default_value=int(cfg_params["FeInstanceCount"]),
    )
    fe_volume_size = ParameterInteger(
        name="FeVolumeSize",
        default_value=int(cfg_params["FeVolumeSize"]),
    )
    fe_format = ParameterString(
        name="FeOutputFormat",
        default_value=str(cfg_params["FeOutputFormat"]),
        enum_values=["csv", "parquet"],
    )
    fe_no_header = ParameterString(
        name="FeNoHeader",
        default_value=str(cfg_params["FeNoHeader"]),
    )
    fe_batch_size = ParameterString(
        name="FeBatchSize",
        default_value=str(cfg_params["FeBatchSize"]),
    )
    fe_val_split = ParameterString(
        name="FeValSplit",
        default_value=str(cfg_params["FeValSplit"]),
    )

    # -- Training infrastructure --
    instance_type = ParameterString(
        name="InstanceType",
        default_value=str(cfg_params["InstanceType"]),
    )
    instance_count = ParameterInteger(
        name="InstanceCount",
        default_value=int(cfg_params["InstanceCount"]),
    )
    volume_size = ParameterInteger(
        name="VolumeSize",
        default_value=int(cfg_params["VolumeSize"]),
    )
    output_s3 = ParameterString(
        name="OutputS3Uri",
        default_value=str(cfg_params["OutputS3Uri"]),
    )

    # -- Shared training settings --
    num_rounds = ParameterString(
        name="NumRounds",
        default_value=str(cfg_params["NumRounds"]),
    )
    early_stopping = ParameterString(
        name="EarlyStopping",
        default_value=str(cfg_params["EarlyStopping"]),
    )

    # -- XGBoost hyperparameters --
    xgb_objective = ParameterString(
        name="XgbObjective",
        default_value=str(cfg_params["XgbObjective"]),
    )
    xgb_tree_method = ParameterString(
        name="XgbTreeMethod",
        default_value=str(cfg_params["XgbTreeMethod"]),
    )
    xgb_no_header = ParameterString(
        name="XgbNoHeader",
        default_value=str(cfg_params["XgbNoHeader"]),
    )
    xgb_lr = ParameterString(
        name="XgbLearningRate",
        default_value=str(cfg_params["XgbLearningRate"]),
    )
    xgb_max_depth = ParameterString(
        name="XgbMaxDepth",
        default_value=str(cfg_params["XgbMaxDepth"]),
    )
    xgb_subsample = ParameterString(
        name="XgbSubsample",
        default_value=str(cfg_params["XgbSubsample"]),
    )
    xgb_colsample = ParameterString(
        name="XgbColsampleBytree",
        default_value=str(cfg_params["XgbColsampleBytree"]),
    )
    xgb_min_child = ParameterString(
        name="XgbMinChildWeight",
        default_value=str(cfg_params["XgbMinChildWeight"]),
    )

    # -- LightGBM hyperparameters --
    lgbm_objective = ParameterString(
        name="LgbmObjective",
        default_value=str(cfg_params["LgbmObjective"]),
    )
    lgbm_no_header = ParameterString(
        name="LgbmNoHeader",
        default_value=str(cfg_params["LgbmNoHeader"]),
    )
    lgbm_lr = ParameterString(
        name="LgbmLearningRate",
        default_value=str(cfg_params["LgbmLearningRate"]),
    )
    lgbm_max_depth = ParameterString(
        name="LgbmMaxDepth",
        default_value=str(cfg_params["LgbmMaxDepth"]),
    )
    lgbm_num_leaves = ParameterString(
        name="LgbmNumLeaves",
        default_value=str(cfg_params["LgbmNumLeaves"]),
    )
    lgbm_subsample = ParameterString(
        name="LgbmSubsample",
        default_value=str(cfg_params["LgbmSubsample"]),
    )
    lgbm_colsample = ParameterString(
        name="LgbmColsampleBytree",
        default_value=str(cfg_params["LgbmColsampleBytree"]),
    )
    lgbm_min_child = ParameterString(
        name="LgbmMinChildSamples",
        default_value=str(cfg_params["LgbmMinChildSamples"]),
    )

    # -- HPO tuning settings --
    max_tuning_jobs = ParameterInteger(
        name="MaxTuningJobs",
        default_value=int(cfg_params["MaxTuningJobs"]),
    )
    max_parallel_jobs = ParameterInteger(
        name="MaxParallelJobs",
        default_value=int(cfg_params["MaxParallelJobs"]),
    )
    tuning_strategy = ParameterString(
        name="TuningStrategy",
        default_value=str(cfg_params["TuningStrategy"]),
        enum_values=["Bayesian", "Random"],
    )
    objective_metric = ParameterString(
        name="ObjectiveMetric",
        default_value=str(cfg_params["ObjectiveMetric"]),
    )
    objective_type = ParameterString(
        name="ObjectiveType",
        default_value=str(cfg_params["ObjectiveType"]),
        enum_values=["Maximize", "Minimize"],
    )

    # ================================================================
    # Feature Engineering Step (ProcessingStep)
    # ================================================================

    fe_src = cfg_src["feature_engineering"]
    fw_version = str(cfg_src["framework_version"])
    py_version = str(cfg_src["py_version"])

    processor = PyTorchProcessor(
        role=execution_role,
        instance_count=fe_instance_count,
        instance_type=fe_instance_type,
        volume_size_in_gb=fe_volume_size,
        framework_version=fw_version,
        py_version=py_version,
        sagemaker_session=pipeline_session,
        base_job_name="loan-feature-engineering",
    )

    step_fe = ProcessingStep(
        name="FeatureEngineering",
        step_args=processor.run(
            code=fe_src["entry_point"],
            source_dir=fe_src["source_dir"],
            inputs=[
                ProcessingInput(
                    source=fe_raw_input_s3,
                    destination="/opt/ml/processing/input/data",
                    s3_data_distribution_type="FullyReplicated",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    source="/opt/ml/processing/output/data",
                    destination=fe_output_s3,
                ),
            ],
            arguments=[
                "--output", "/opt/ml/processing/output/data",
                "--format", fe_format,
                "--batch-size", fe_batch_size,
                "--no-header", fe_no_header,
                "--val-split", fe_val_split,
            ],
        ),
    )

    # ================================================================
    # Training Input (from FE output train/val split)
    # ================================================================

    # FE produces {fe_output_s3}/train and {fe_output_s3}/val subdirectories
    train_data_uri = Join(on="/", values=[fe_output_s3, "train"])
    val_data_uri = Join(on="/", values=[fe_output_s3, "val"])

    train_input = TrainingInput(s3_data=train_data_uri)
    val_input = TrainingInput(s3_data=val_data_uri)
    channels = {"training": train_input, "validation": val_input}

    # Checkpoint S3 paths for multi-node Ray
    xgb_ckpt_s3 = f"s3://{default_bucket}/ray-checkpoints-xgb"
    lgbm_ckpt_s3 = f"s3://{default_bucket}/ray-checkpoints-lgbm"

    # Source config for training (baked at pipeline creation time)
    xgb_src = cfg_src["xgboost"]
    lgbm_src = cfg_src["lightgbm"]

    # ================================================================
    # Training Steps
    # ================================================================

    xgb_train_estimator = PyTorch(
        entry_point=xgb_src["entry_point"],
        source_dir=xgb_src["source_dir"],
        role=execution_role,
        instance_count=instance_count,
        instance_type=instance_type,
        volume_size=volume_size,
        framework_version=fw_version,
        py_version=py_version,
        output_path=output_s3,
        sagemaker_session=pipeline_session,
        base_job_name="loan-xgb-training",
        metric_definitions=XGB_METRIC_DEFINITIONS,
        hyperparameters={
            "objective": xgb_objective,
            "tree-method": xgb_tree_method,
            "no-header": xgb_no_header,
            "num-rounds": num_rounds,
            "early-stopping": early_stopping,
            "num-workers": "0",
            "learning-rate": xgb_lr,
            "max-depth": xgb_max_depth,
            "subsample": xgb_subsample,
            "colsample-bytree": xgb_colsample,
            "min-child-weight": xgb_min_child,
            "checkpoint-s3-uri": xgb_ckpt_s3,
        },
        disable_profiler=True,
        disable_output_compression=True
    )

    step_train_xgb = TrainingStep(
        name="TrainXGBoost",
        step_args=xgb_train_estimator.fit(inputs=channels),
    )

    lgbm_train_estimator = PyTorch(
        entry_point=lgbm_src["entry_point"],
        source_dir=lgbm_src["source_dir"],
        role=execution_role,
        instance_count=instance_count,
        instance_type=instance_type,
        volume_size=volume_size,
        framework_version=fw_version,
        py_version=py_version,
        output_path=output_s3,
        sagemaker_session=pipeline_session,
        base_job_name="loan-lgbm-training",
        metric_definitions=LGBM_METRIC_DEFINITIONS,
        hyperparameters={
            "objective": lgbm_objective,
            "no-header": lgbm_no_header,
            "num-rounds": num_rounds,
            "early-stopping": early_stopping,
            "num-workers": "0",
            "learning-rate": lgbm_lr,
            "max-depth": lgbm_max_depth,
            "num-leaves": lgbm_num_leaves,
            "subsample": lgbm_subsample,
            "colsample-bytree": lgbm_colsample,
            "min-child-samples": lgbm_min_child,
            "checkpoint-s3-uri": lgbm_ckpt_s3,
        },
        disable_profiler=True,
        disable_output_compression=True
    )

    step_train_lgbm = TrainingStep(
        name="TrainLightGBM",
        step_args=lgbm_train_estimator.fit(inputs=channels),
    )

    # ================================================================
    # Tuning Steps (HPO)
    # ================================================================

    xgb_tune_estimator = PyTorch(
        entry_point=xgb_src["entry_point"],
        source_dir=xgb_src["source_dir"],
        role=execution_role,
        instance_count=1,
        instance_type=instance_type,
        volume_size=volume_size,
        framework_version=fw_version,
        py_version=py_version,
        output_path=output_s3,
        sagemaker_session=pipeline_session,
        base_job_name="loan-hpo-xgboost",
        metric_definitions=XGB_METRIC_DEFINITIONS,
        hyperparameters={
            "objective": xgb_objective,
            "tree-method": xgb_tree_method,
            "no-header": xgb_no_header,
            "num-rounds": num_rounds,
            "early-stopping": early_stopping,
            "num-workers": "0",
        },
        disable_profiler=True,
        disable_output_compression=True
    )

    xgb_tuner = HyperparameterTuner(
        estimator=xgb_tune_estimator,
        objective_metric_name=objective_metric,
        hyperparameter_ranges=_build_hp_ranges(cfg_ranges.get("xgboost", {})),
        objective_type=objective_type,
        max_jobs=max_tuning_jobs,
        max_parallel_jobs=max_parallel_jobs,
        strategy=tuning_strategy,
        base_tuning_job_name="loan-hpo-xgboost",
    )

    step_tune_xgb = TuningStep(
        name="TuneXGBoost",
        step_args=xgb_tuner.fit(inputs=channels),
    )

    lgbm_tune_estimator = PyTorch(
        entry_point=lgbm_src["entry_point"],
        source_dir=lgbm_src["source_dir"],
        role=execution_role,
        instance_count=1,
        instance_type=instance_type,
        volume_size=volume_size,
        framework_version=fw_version,
        py_version=py_version,
        output_path=output_s3,
        sagemaker_session=pipeline_session,
        base_job_name="loan-hpo-lightgbm",
        metric_definitions=LGBM_METRIC_DEFINITIONS,
        hyperparameters={
            "objective": lgbm_objective,
            "no-header": lgbm_no_header,
            "num-rounds": num_rounds,
            "early-stopping": early_stopping,
            "num-workers": "0",
        },
        disable_profiler=True,
        disable_output_compression=True
    )

    lgbm_tuner = HyperparameterTuner(
        estimator=lgbm_tune_estimator,
        objective_metric_name=objective_metric,
        hyperparameter_ranges=_build_hp_ranges(cfg_ranges.get("lightgbm", {})),
        objective_type=objective_type,
        max_jobs=max_tuning_jobs,
        max_parallel_jobs=max_parallel_jobs,
        strategy=tuning_strategy,
        base_tuning_job_name="loan-hpo-lightgbm",
    )

    step_tune_lgbm = TuningStep(
        name="TuneLightGBM",
        step_args=lgbm_tuner.fit(inputs=channels),
    )

    # ================================================================
    # Multi-Algorithm Tuning Step (both algorithms in one tuner)
    # ================================================================

    both_tuner = HyperparameterTuner.create(
        estimator_dict={
            "xgboost": xgb_tune_estimator,
            "lightgbm": lgbm_tune_estimator,
        },
        objective_metric_name_dict={
            "xgboost": objective_metric,
            "lightgbm": objective_metric,
        },
        hyperparameter_ranges_dict={
            "xgboost": _build_hp_ranges(cfg_ranges.get("xgboost", {})),
            "lightgbm": _build_hp_ranges(cfg_ranges.get("lightgbm", {})),
        },
        metric_definitions_dict={
            "xgboost": XGB_METRIC_DEFINITIONS,
            "lightgbm": LGBM_METRIC_DEFINITIONS,
        },
        objective_type=objective_type,
        strategy=tuning_strategy,
        max_jobs=max_tuning_jobs,
        max_parallel_jobs=max_parallel_jobs,
        base_tuning_job_name="loan-hpo-xgb-lgbm",
    )

    both_channels = {
        "xgboost": channels,
        "lightgbm": channels,
    }

    step_tune_both = TuningStep(
        name="TuneBothAlgorithms",
        step_args=both_tuner.fit(inputs=both_channels, include_cls_metadata={}),
    )

    # ================================================================
    # Condition Steps (all depend on FE completing first)
    # ================================================================

    step_check_xgb_train = ConditionStep(
        name="CheckXgbTraining",
        depends_on=[step_fe],
        conditions=[
            ConditionIn(value=algorithm_choice, in_values=["xgboost", "both"]),
            ConditionEquals(left=run_tuning, right="false"),
        ],
        if_steps=[step_train_xgb],
        else_steps=[],
    )

    step_check_lgbm_train = ConditionStep(
        name="CheckLgbmTraining",
        depends_on=[step_fe],
        conditions=[
            ConditionIn(value=algorithm_choice, in_values=["lightgbm", "both"]),
            ConditionEquals(left=run_tuning, right="false"),
        ],
        if_steps=[step_train_lgbm],
        else_steps=[],
    )

    step_check_xgb_tune = ConditionStep(
        name="CheckXgbTuning",
        depends_on=[step_fe],
        conditions=[
            ConditionEquals(left=algorithm_choice, right="xgboost"),
            ConditionEquals(left=run_tuning, right="true"),
        ],
        if_steps=[step_tune_xgb],
        else_steps=[],
    )

    step_check_lgbm_tune = ConditionStep(
        name="CheckLgbmTuning",
        depends_on=[step_fe],
        conditions=[
            ConditionEquals(left=algorithm_choice, right="lightgbm"),
            ConditionEquals(left=run_tuning, right="true"),
        ],
        if_steps=[step_tune_lgbm],
        else_steps=[],
    )

    step_check_both_tune = ConditionStep(
        name="CheckBothTuning",
        depends_on=[step_fe],
        conditions=[
            ConditionEquals(left=algorithm_choice, right="both"),
            ConditionEquals(left=run_tuning, right="true"),
        ],
        if_steps=[step_tune_both],
        else_steps=[],
    )

    # ================================================================
    # Assemble Pipeline
    # ================================================================

    all_parameters = [
        # Role
        execution_role,
        # Control
        algorithm_choice, run_tuning,
        # Feature Engineering
        fe_raw_input_s3, fe_output_s3,
        fe_instance_type, fe_instance_count, fe_volume_size,
        fe_format, fe_no_header, fe_batch_size, fe_val_split,
        # Training infrastructure
        instance_type, instance_count, volume_size, output_s3,
        # Shared training
        num_rounds, early_stopping,
        # XGBoost HPs
        xgb_objective, xgb_tree_method, xgb_no_header,
        xgb_lr, xgb_max_depth, xgb_subsample, xgb_colsample, xgb_min_child,
        # LightGBM HPs
        lgbm_objective, lgbm_no_header,
        lgbm_lr, lgbm_max_depth, lgbm_num_leaves,
        lgbm_subsample, lgbm_colsample, lgbm_min_child,
        # HPO
        max_tuning_jobs, max_parallel_jobs, tuning_strategy,
        objective_metric, objective_type,
    ]

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=all_parameters,
        steps=[
            step_fe,
            step_check_xgb_train,
            step_check_lgbm_train,
            step_check_xgb_tune,
            step_check_lgbm_tune,
            step_check_both_tune,
        ],
        sagemaker_session=pipeline_session,
    )

    return pipeline


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Create/update and optionally start the FE -> Training pipeline"
    )
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help="Path to pipeline config YAML file")
    p.add_argument("--pipeline-name", default=None,
                   help="Override pipeline name from config")
    p.add_argument("--start", action="store_true",
                   help="Start a pipeline execution after creating/updating")
    p.add_argument("--params", nargs="*", default=[],
                   help="Parameter overrides as Key=Value pairs "
                        "(e.g., AlgorithmChoice=xgboost RunTuning=true)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading config: {args.config}")
    config = load_config(args.config)

    # Override pipeline name from CLI if provided
    if args.pipeline_name:
        config["pipeline"]["fe_training_name"] = args.pipeline_name
    pipeline_name = config["pipeline"].get(
        "fe_training_name", "loan-fe-training-pipeline"
    )

    # Resolve execution role: config > auto-detect
    cfg_role = config["pipeline"].get("role")
    if cfg_role:
        role = cfg_role
        print(f"Role:           {role} (from config)")
    else:
        role = sagemaker.get_execution_role()
        print(f"Role:           {role} (auto-detected)")

    print(f"Building pipeline: {pipeline_name}")
    pipeline = create_pipeline(config, role)

    print("Creating / updating pipeline...")
    pipeline.upsert(role_arn=role)
    print(f"Pipeline ready: {pipeline_name}")

    region = sagemaker.Session().boto_region_name
    print(f"View at: https://{region}.console.aws.amazon.com/sagemaker/home"
          f"?region={region}#/pipelines/{pipeline_name}")

    if args.start:
        # Parse Key=Value overrides
        overrides = {}
        for kv in args.params:
            if "=" not in kv:
                raise ValueError(f"Invalid parameter format '{kv}', expected Key=Value")
            key, value = kv.split("=", 1)
            overrides[key] = value

        if overrides:
            print(f"\nStarting execution with overrides: {overrides}")
        else:
            print("\nStarting execution with defaults...")

        execution = pipeline.start(parameters=overrides)
        print(f"Execution ARN: {execution.arn}")
        print(f"View at: https://{region}.console.aws.amazon.com/sagemaker/home"
              f"?region={region}#/pipelines/{pipeline_name}/executions/"
              f"{execution.arn.split('/')[-1]}")


if __name__ == "__main__":
    main()
