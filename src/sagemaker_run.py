"""
SageMaker multi-node entry point for Ray feature engineering pipeline.

This script handles:
  1. Detecting SageMaker environment (Processing or Training job)
  2. Bootstrapping a Ray cluster across all SageMaker nodes
  3. Running the feature engineering pipeline on the head node
  4. Workers contribute CPU/memory but don't run the pipeline directly

Works with:
  - SageMaker Processing (FrameworkProcessor / ScriptProcessor)
  - SageMaker Training (Estimator with Framework containers)
  - Local mode (no SageMaker — falls back to single-node Ray)

Usage in SageMaker (V2):
    # As a Processing job
    from sagemaker.processing import FrameworkProcessor
    processor = FrameworkProcessor(
        estimator_cls=PyTorch,  # or any framework with Ray installed
        role=role,
        instance_count=4,
        instance_type="ml.m5.4xlarge",
        framework_version="2.1",
    )
    processor.run(
        code="sagemaker_run.py",
        source_dir=".",  # includes feature_engineering.py
        inputs=[ProcessingInput(source="s3://bucket/raw_data.csv", destination="/opt/ml/processing/input/data")],
        outputs=[ProcessingOutput(source="/opt/ml/processing/output/data", destination="s3://bucket/features/")],
        arguments=["--format", "csv", "--no-header"],
    )

    # As a Training job (Estimator)
    from sagemaker.pytorch import PyTorch
    estimator = PyTorch(
        entry_point="sagemaker_run.py",
        source_dir=".",
        role=role,
        instance_count=4,
        instance_type="ml.m5.4xlarge",
        framework_version="2.1",
        py_version="py310",
        hyperparameters={"format": "csv", "no-header": ""},
    )
    estimator.fit({"training": "s3://bucket/raw_data.csv"})

    # Local test (no SageMaker)
    python sagemaker_run.py --input raw_loan_data.csv --output features.parquet
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time


# ---------------------------------------------------------------------------
# SageMaker environment detection
# ---------------------------------------------------------------------------

RESOURCE_CONFIG_PATHS = [
    "/opt/ml/config/resourceconfig.json",      # Processing jobs
    "/opt/ml/input/config/resourceconfig.json", # Training jobs
]


def get_sagemaker_env():
    """Detect SageMaker environment and return config dict, or None if local.

    SageMaker Training jobs set SM_HOSTS / SM_CURRENT_HOST env vars.
    SageMaker Processing jobs do NOT — they only provide resourceconfig.json.
    We check both sources.
    """
    hosts = None
    current_host = None
    network_interface = "eth0"

    # Source 1: env vars (Training jobs)
    sm_hosts = os.environ.get("SM_HOSTS")
    sm_current = os.environ.get("SM_CURRENT_HOST")
    if sm_hosts and sm_current:
        hosts = json.loads(sm_hosts)
        current_host = sm_current
        network_interface = os.environ.get("SM_NETWORK_INTERFACE_NAME", "eth0")

    # Source 2: resourceconfig.json (Processing + Training jobs)
    if not hosts:
        for rc_path in RESOURCE_CONFIG_PATHS:
            if os.path.isfile(rc_path):
                with open(rc_path) as f:
                    rc = json.load(f)
                hosts = rc.get("hosts", [])
                current_host = rc.get("current_host")
                network_interface = rc.get("network_interface_name", "eth0")
                print(f"[SageMaker] Read resource config from {rc_path}")
                break

    if not hosts or not current_host:
        return None

    # Determine job type by checking which paths exist
    is_processing = os.path.isdir("/opt/ml/processing")
    is_training = os.path.isdir("/opt/ml/input") and not is_processing

    if is_processing:
        input_dir = "/opt/ml/processing/input/data"
        output_dir = "/opt/ml/processing/output/data"
    else:
        channel = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
        input_dir = channel
        output_dir = "/opt/ml/model"

    return {
        "hosts": hosts,
        "current_host": current_host,
        "is_head": current_host == hosts[0],
        "head_host": hosts[0],
        "num_nodes": len(hosts),
        "network_interface": network_interface,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "is_processing": is_processing,
        "is_training": is_training,
        "num_cpus_per_node": os.cpu_count(),
    }


# ---------------------------------------------------------------------------
# Ray cluster bootstrap
# ---------------------------------------------------------------------------

RAY_HEAD_PORT = 6379
RAY_DASHBOARD_PORT = 8265


def _get_host_ip(hostname):
    """Resolve hostname to IP address."""
    try:
        return socket.gethostbyname(hostname)
    except socket.gaierror:
        return hostname


def _wait_for_port(host, port, timeout=300):
    """Wait until a TCP port is accepting connections."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=5):
                return True
        except (ConnectionRefusedError, OSError, socket.timeout):
            time.sleep(1)
    raise TimeoutError(f"Port {host}:{port} not available after {timeout}s")


def _run_cmd(cmd):
    """Run a shell command and print output."""
    print(f"  $ {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        print(f"    stdout: {result.stdout.strip()}")
    if result.returncode != 0:
        print(f"    stderr: {result.stderr.strip()}")
    return result


def bootstrap_ray_cluster(sm_env):
    """
    Start a Ray cluster across SageMaker nodes.

    - Head node: starts `ray start --head`
    - Workers: wait for head, then `ray start --address=<head>`
    - Head waits until all nodes have joined
    """
    head_ip = _get_host_ip(sm_env["head_host"])
    head_addr = f"{head_ip}:{RAY_HEAD_PORT}"
    num_cpus = sm_env["num_cpus_per_node"]

    print(f"\n{'='*60}")
    print(f"RAY CLUSTER BOOTSTRAP")
    print(f"{'='*60}")
    print(f"  Nodes:          {sm_env['num_nodes']}")
    print(f"  CPUs per node:  {num_cpus}")
    print(f"  Total CPUs:     {sm_env['num_nodes'] * num_cpus}")
    print(f"  Head node:      {sm_env['head_host']} ({head_ip})")
    print(f"  Current node:   {sm_env['current_host']}")
    print(f"  Role:           {'HEAD' if sm_env['is_head'] else 'WORKER'}")
    print(f"{'='*60}\n")

    if sm_env["is_head"]:
        # Start Ray head
        print("[Head] Starting Ray head node...")
        _run_cmd(
            f"ray start --head "
            f"--port={RAY_HEAD_PORT} "
            f"--dashboard-port={RAY_DASHBOARD_PORT} "
            f"--num-cpus={num_cpus} "
            f"--disable-usage-stats"
        )

        # Wait for all workers to join
        if sm_env["num_nodes"] > 1:
            print(f"[Head] Waiting for {sm_env['num_nodes'] - 1} worker(s) to join...")
            _wait_for_workers(sm_env["num_nodes"], timeout=600)
            print(f"[Head] All {sm_env['num_nodes']} nodes are in the cluster.")
        else:
            print("[Head] Single-node cluster ready.")

    else:
        # Worker: wait for head to be available, then join
        print(f"[Worker] Waiting for head node at {head_addr}...")
        _wait_for_port(head_ip, RAY_HEAD_PORT, timeout=300)

        print("[Worker] Head is up. Joining cluster...")
        _run_cmd(
            f"ray start "
            f"--address={head_addr} "
            f"--num-cpus={num_cpus} "
            f"--disable-usage-stats"
        )
        print("[Worker] Joined. Idling until head completes pipeline.")

    return head_addr


def _wait_for_workers(expected_nodes, timeout=600):
    """Wait until the Ray cluster has the expected number of nodes."""
    import ray
    ray.init(address="auto")

    start = time.time()
    while time.time() - start < timeout:
        nodes = ray.nodes()
        alive = [n for n in nodes if n["Alive"]]
        if len(alive) >= expected_nodes:
            ray.shutdown()
            return
        print(f"  ... {len(alive)}/{expected_nodes} nodes alive "
              f"({int(time.time() - start)}s elapsed)", flush=True)
        time.sleep(5)

    ray.shutdown()
    alive_count = len([n for n in ray.nodes() if n["Alive"]])
    raise TimeoutError(
        f"Only {alive_count}/{expected_nodes} nodes joined after {timeout}s"
    )


# ---------------------------------------------------------------------------
# Resolve input/output paths
# ---------------------------------------------------------------------------

def resolve_paths(args, sm_env):
    """
    Determine input and output paths.

    Priority:
      1. Explicit CLI --input / --output (can be S3 paths)
      2. SageMaker-provided directories
      3. Local defaults
    """
    # Input path
    if args.input:
        input_path = args.input
    elif sm_env:
        # Auto-detect: find the data file in the SageMaker input dir
        input_dir = sm_env["input_dir"]
        if os.path.isdir(input_dir):
            files = [f for f in os.listdir(input_dir)
                     if f.endswith((".csv", ".parquet"))]
            if files:
                input_path = os.path.join(input_dir, files[0])
            else:
                # Directory of parquet files
                input_path = input_dir
        else:
            input_path = input_dir

    # Output path
    if args.output:
        output_path = args.output
    elif sm_env:
        output_dir = sm_env["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir
    else:
        output_path = f"features"

    return input_path, output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SageMaker multi-node Ray feature engineering"
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Input data path (local, S3, or auto-detected)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (local, S3, or auto-detected)")
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv",
                        help="Output format (default: csv)")
    parser.add_argument("--no-header", nargs="?", const="true", default="false",
                        help="Omit CSV header row")
    parser.add_argument("--batch-size", type=int, default=32768,
                        help="Rows per batch (default: 32768)")
    parser.add_argument("--val-split", type=float, default=0.0,
                        help="Validation split fraction (default: 0.0)")
    parser.add_argument("--signal-s3-uri", type=str, default=None,
                        help="S3 URI for cross-node completion signal "
                             "(e.g. s3://bucket/signals/). Required for "
                             "multi-node jobs without shared EFS.")
    return parser.parse_args()


def main():
    args = parse_args()
    sm_env = get_sagemaker_env()

    if sm_env:
        print(f"SageMaker environment detected: "
              f"{'Processing' if sm_env['is_processing'] else 'Training'} job, "
              f"{sm_env['num_nodes']} node(s)")

        signal_path = _get_signal_path(sm_env, args)

        # Bootstrap Ray cluster
        head_addr = bootstrap_ray_cluster(sm_env)

        if not sm_env["is_head"]:
            # Worker: wait for head to finish, then clean up
            try:
                print(f"[Worker] Listening for signal at {signal_path}")
                _wait_for_head_completion(signal_path)
            finally:
                _stop_ray()
            print("[Worker] Exiting.")
            return

        # Head node: connect to the cluster we bootstrapped
        import ray
        ray.init(address="auto")
        print(f"\nRay cluster resources: {ray.cluster_resources()}")

    else:
        print("No SageMaker environment detected. Running locally.")
        import ray
        ray.init()
        print(f"Ray cluster resources: {ray.cluster_resources()}")
        signal_path = None

    # Resolve paths
    input_path, output_path = resolve_paths(args, sm_env)
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")
    print()

    # Run the pipeline
    try:
        from feature_engineering import run_pipeline
        run_pipeline(
            input_path=input_path,
            output_path=output_path,
            out_format=args.format,
            batch_size=args.batch_size,
            no_header=str(args.no_header).lower() in ("true", "1", "yes", ""),
            val_split=args.val_split,
        )
    finally:
        # Always clean up Ray — even on failure
        import ray
        ray.shutdown()

        if sm_env:
            # Signal workers BEFORE stopping Ray, so they see it while
            # still connected (gives them time to wrap up)
            if signal_path:
                _signal_completion(signal_path)
            # Now stop the daemon
            _stop_ray()

# ---------------------------------------------------------------------------
# Worker synchronization
# ---------------------------------------------------------------------------

# For SageMaker Processing: /opt/ml/processing/output is often a shared
# EBS/EFS mount. But that's not guaranteed, so we support S3 too.
# For S3-based signaling, pass --signal-s3-uri s3://bucket/job-signals/

LOCAL_SIGNAL_DIR = "/tmp/ray_pipeline_signals"
COMPLETION_FILENAME = "pipeline_complete"


def _get_signal_path(sm_env, args):
    """
    Return the path where the completion signal will be written/read.

    Uses shared filesystem if available, otherwise falls back to a local
    marker (only works for single-node).
    """
    # If the user provided an S3 signal path, use it
    signal_uri = getattr(args, "signal_s3_uri", None)
    if signal_uri:
        return f"{signal_uri.rstrip('/')}/{COMPLETION_FILENAME}"

    # SageMaker Processing: output dir is typically shared via EFS/EBS
    if sm_env.get("is_processing"):
        signal_dir = sm_env["output_dir"]
        os.makedirs(signal_dir, exist_ok=True)
        return os.path.join(signal_dir, COMPLETION_FILENAME)

    # Fallback: local tmp (only reliable for single-node)
    os.makedirs(LOCAL_SIGNAL_DIR, exist_ok=True)
    return os.path.join(LOCAL_SIGNAL_DIR, COMPLETION_FILENAME)


def _signal_completion(signal_path):
    """Head signals completion by writing a marker."""
    if signal_path.startswith("s3://"):
        # Write to S3
        import boto3
        parts = signal_path.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=b"done")
    else:
        with open(signal_path, "w") as f:
            f.write("done")
    print(f"[Head] Completion signal written to {signal_path}")


def _check_signal_exists(signal_path):
    """Check if the completion signal exists."""
    if signal_path.startswith("s3://"):
        import boto3
        parts = signal_path.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        try:
            boto3.client("s3").head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False
    else:
        return os.path.isfile(signal_path)


def _wait_for_head_completion(signal_path, timeout=7200):
    """
    Workers wait for the head to finish by polling for a completion signal.

    Falls back to detecting Ray daemon shutdown if the signal mechanism fails.
    """
    start = time.time()
    while time.time() - start < timeout:
        # Primary: check for the explicit completion signal
        if _check_signal_exists(signal_path):
            print("[Worker] Completion signal detected.")
            return

        # Secondary: if Ray daemon is gone, head must have stopped
        result = subprocess.run(
            "ray status", shell=True, capture_output=True, text=True
        )
        if result.returncode != 0:
            print("[Worker] Ray daemon exited.")
            return

        time.sleep(10)

    print(f"[Worker] WARNING: Timed out after {timeout}s waiting for head.")


def _stop_ray():
    """Stop the Ray daemon processes on this node."""
    print("[Cleanup] Stopping Ray daemon...")
    subprocess.run("ray stop --force", shell=True, capture_output=True)
    print("[Cleanup] Ray stopped.")

if __name__ == "__main__":
    main()
