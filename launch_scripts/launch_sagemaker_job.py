"""
Launch the feature engineering pipeline as a SageMaker multi-node Processing job.

This script submits the job from a local machine / notebook / SageMaker Studio.
The actual feature engineering runs on the remote SageMaker instances.

Usage:
    python launch_sagemaker_job.py                          # 2 nodes, ml.m5.4xlarge
    python launch_sagemaker_job.py --instance-count 4       # 4 nodes
    python launch_sagemaker_job.py --instance-type ml.m5.12xlarge
    python launch_sagemaker_job.py --wait                   # block until complete
"""

import argparse
import sagemaker
from sagemaker.pytorch import PyTorchProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

S3_BUCKET = "mm-fsi-fix"
S3_PREFIX = "loan-fe"

# Paths
S3_INPUT = f"s3://{S3_BUCKET}/{S3_PREFIX}/raw_loan_data.csv"
S3_OUTPUT = f"s3://{S3_BUCKET}/{S3_PREFIX}/features/"


def parse_args():
    parser = argparse.ArgumentParser(description="Launch SageMaker FE job")
    parser.add_argument("--instance-count", type=int, default=2,
                        help="Number of instances (default: 2)")
    parser.add_argument("--instance-type", type=str, default="ml.m5.4xlarge",
                        help="Instance type (default: ml.m5.4xlarge)")
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv",
                        help="Output format (default: csv)")
    parser.add_argument("--no-header", action="store_true",
                        help="Omit CSV header")
    parser.add_argument("--wait", action="store_true",
                        help="Wait for job to complete")
    parser.add_argument("--s3-input", type=str, default=S3_INPUT,
                        help=f"S3 input path (default: {S3_INPUT})")
    parser.add_argument("--s3-output", type=str, default=S3_OUTPUT,
                        help=f"S3 output path (default: {S3_OUTPUT})")
    return parser.parse_args()


def main():
    args = parse_args()
    session = sagemaker.Session()
    role = sagemaker.get_execution_role()

    print(f"Role:            {role}")
    print(f"Region:          {session.boto_region_name}")
    print(f"Instance type:   {args.instance_type}")
    print(f"Instance count:  {args.instance_count}")
    print(f"Input:           {args.s3_input}")
    print(f"Output:          {args.s3_output}")
    print()

    # Use PyTorchProcessor — provides a Python environment where we can
    # pip install Ray via requirements.txt. PyTorch itself isn't used,
    # but it gives us a well-maintained container with good Python/pip support.
    processor = PyTorchProcessor(
        role=role,
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        framework_version="2.5.1",
        py_version="py311",
        sagemaker_session=session,
        base_job_name="loan-feature-engineering",
    )

    # Build script arguments
    script_args = [
        "--input", "/opt/ml/processing/input/data/raw_loan_data.csv",
        "--output", "/opt/ml/processing/output/data",
        "--format", args.format,
        "--batch-size", "65536",
    ]
    if args.no_header:
        script_args.append("--no-header")

    print("Launching processing job...")
    processor.run(
        code="sagemaker_run.py",
        source_dir="src",  # contains sagemaker_run.py, feature_engineering.py, requirements.txt
        inputs=[
            ProcessingInput(
                source=args.s3_input,
                destination="/opt/ml/processing/input/data",
                s3_data_distribution_type="FullyReplicated",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output/data",
                destination=args.s3_output,
            ),
        ],
        arguments=script_args,
        wait=args.wait,
        logs=args.wait,
    )

    if args.wait:
        print(f"\nJob complete. Output at: {args.s3_output}")
    else:
        job_name = processor.latest_job.name
        print(f"\nJob submitted: {job_name}")
        print(f"Monitor at: https://{session.boto_region_name}.console.aws.amazon.com/"
              f"sagemaker/home?region={session.boto_region_name}#/processing-jobs/{job_name}")
        print(f"\nTo wait for completion:")
        print(f"  processor.latest_job.wait()")


if __name__ == "__main__":
    main()
