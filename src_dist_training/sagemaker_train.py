"""
SageMaker entry point for XGBoost training (single-node or multi-node distributed).

Single-node: ray.init() starts a local cluster.
Multi-node:  Bootstraps a Ray cluster across SageMaker instances using
             resourceconfig.json, then runs training on the head node.
             Worker nodes idle until the head completes.

SageMaker channels:
  - "training" (required): training data
  - "validation" (optional): validation data — if absent, splits from training
"""

import argparse
import json
import os
import socket
import subprocess
import time

import ray

from train_xgboost_dist import run_training


# ---------------------------------------------------------------------------
# Object store memory helpers
# ---------------------------------------------------------------------------

def _get_memory_info():
    """Return (total_ram, available_ram, shm_size) in bytes.

    - total_ram: /proc/meminfo MemTotal (physical RAM)
    - available_ram: /proc/meminfo MemAvailable (usable after kernel/driver reservations)
    - shm_size: /dev/shm total size (tmpfs backing for Ray object store)

    On GPU instances, MemTotal overstates usable RAM because CUDA drivers and
    pinned memory reserves are not reflected. MemAvailable is the safer base
    for allocation decisions.
    """
    total, available = None, None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass

    shm_size = None
    try:
        import shutil
        stat = shutil.disk_usage("/dev/shm")
        shm_size = stat.total
    except (OSError, AttributeError):
        pass

    return total, available, shm_size


def _resolve_object_store_bytes(memory_bytes, fraction, total_ram, available_ram, shm_size):
    """Determine object store size in bytes from user flags.

    Args:
        memory_bytes: Explicit size in bytes (-1.0 = auto)
        fraction: Fraction of available RAM (-1.0 = auto)
        total_ram: Physical RAM in bytes
        available_ram: Available RAM in bytes (accounts for GPU/kernel reserves)
        shm_size: /dev/shm size in bytes (object store backing tmpfs)

    Returns:
        int or None. None means let Ray decide.
    """
    # Explicit bytes takes priority — no further adjustment
    if memory_bytes > 0:
        return int(memory_bytes)

    # Use available RAM as the base (not total) — on GPU instances this can
    # be 30-40% less than MemTotal due to CUDA driver/pinned memory reserves.
    base_ram = available_ram or total_ram

    if fraction > 0:
        if base_ram:
            computed = int(base_ram * fraction)
        else:
            return None
    elif base_ram:
        # Auto: 30% of available RAM. Conservative enough to leave ample heap
        # for DMatrix/QuantileDMatrix construction on each worker.
        computed = int(base_ram * 0.30)
    else:
        return None

    # Cap at 80% of /dev/shm — the tmpfs backing the object store.
    # Exceeding /dev/shm causes Ray to fall back to disk spill.
    if shm_size:
        computed = min(computed, int(shm_size * 0.80))

    return computed


# ---------------------------------------------------------------------------
# Multi-node Ray cluster bootstrap
# ---------------------------------------------------------------------------

RAY_HEAD_PORT = 6379


def _get_sagemaker_hosts():
    """Return (hosts, current_host, network_interface) from SageMaker env."""
    # Training jobs set SM_HOSTS / SM_CURRENT_HOST
    sm_hosts = os.environ.get("SM_HOSTS")
    sm_current = os.environ.get("SM_CURRENT_HOST")
    if sm_hosts and sm_current:
        return json.loads(sm_hosts), sm_current

    # Fallback: resourceconfig.json
    for path in ["/opt/ml/input/config/resourceconfig.json",
                 "/opt/ml/config/resourceconfig.json"]:
        if os.path.isfile(path):
            with open(path) as f:
                rc = json.load(f)
            return rc["hosts"], rc["current_host"]

    return None, None


def _resolve_ip(hostname):
    try:
        return socket.gethostbyname(hostname)
    except socket.gaierror:
        return hostname


def _wait_for_port(host, port, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=5):
                return
        except (ConnectionRefusedError, OSError, socket.timeout):
            time.sleep(1)
    raise TimeoutError(f"Port {host}:{port} not available after {timeout}s")


def _run_cmd(cmd):
    print(f"  $ {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        print(f"    {result.stdout.strip()}")
    if result.returncode != 0 and result.stderr.strip():
        print(f"    stderr: {result.stderr.strip()}")
    return result


def _wait_for_workers(expected, timeout=600):
    """Wait until Ray cluster has expected number of alive nodes."""
    ray.init(address="auto")
    start = time.time()
    while time.time() - start < timeout:
        alive = [n for n in ray.nodes() if n["Alive"]]
        if len(alive) >= expected:
            ray.shutdown()
            return
        print(f"  ... {len(alive)}/{expected} nodes ({int(time.time() - start)}s)")
        time.sleep(5)
    ray.shutdown()
    raise TimeoutError(f"Only got {len(alive)}/{expected} nodes after {timeout}s")


def _wait_for_head_done():
    """Worker blocks here until head shuts down Ray."""
    while True:
        result = subprocess.run("ray status", shell=True,
                                capture_output=True, text=True)
        if result.returncode != 0:
            break
        time.sleep(10)


def bootstrap_ray_cluster(hosts, current_host, num_gpus=0, object_store_bytes=None):
    """Start multi-node Ray cluster. Returns True if this node is the head."""
    head_host = hosts[0]
    is_head = current_host == head_host
    head_ip = _resolve_ip(head_host)
    num_cpus = os.cpu_count()

    print(f"\n{'='*60}")
    print(f"RAY CLUSTER BOOTSTRAP")
    print(f"{'='*60}")
    print(f"  Nodes:         {len(hosts)}")
    print(f"  CPUs/node:     {num_cpus}")
    print(f"  GPUs/node:     {num_gpus}")
    if object_store_bytes:
        print(f"  Object store:  {object_store_bytes / 1e9:.1f} GB")
    print(f"  Head:          {head_host} ({head_ip})")
    print(f"  Current:       {current_host} ({'HEAD' if is_head else 'WORKER'})")
    print(f"{'='*60}\n")

    # Pass --num-gpus so Ray registers GPU resources even if auto-detect fails
    gpu_flag = f" --num-gpus={num_gpus}" if num_gpus > 0 else ""
    obj_store_flag = f" --object-store-memory={object_store_bytes}" if object_store_bytes else ""

    if is_head:
        _run_cmd(
            f"ray start --head --port={RAY_HEAD_PORT} "
            f"--num-cpus={num_cpus}{gpu_flag}{obj_store_flag} --disable-usage-stats"
        )
        if len(hosts) > 1:
            print(f"[Head] Waiting for {len(hosts) - 1} worker(s)...")
            _wait_for_workers(len(hosts))
            print(f"[Head] All {len(hosts)} nodes joined.")
    else:
        head_addr = f"{head_ip}:{RAY_HEAD_PORT}"
        print(f"[Worker] Waiting for head at {head_addr}...")
        _wait_for_port(head_ip, RAY_HEAD_PORT)
        _run_cmd(
            f"ray start --address={head_addr} "
            f"--num-cpus={num_cpus}{gpu_flag}{obj_store_flag} --disable-usage-stats"
        )
        print("[Worker] Joined cluster. Idling until head completes.")

    return is_head


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # SageMaker channels
    input_dir = os.environ.get(
        "SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"
    )
    val_dir = os.environ.get("SM_CHANNEL_VALIDATION")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    num_gpus = int(os.environ.get("SM_NUM_GPUS", 0))
    num_cpus = int(os.environ.get("SM_NUM_CPUS", os.cpu_count()))

    # Ensure NCCL P2P and SHM transport are disabled for GPU instances that
    # may lack NVLink (e.g., g6.12xlarge with L4 GPUs). Forces socket-only
    # transport. Must be set before Ray or XGBoost initialize NCCL.
    if num_gpus > 0:
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_SHM_DISABLE", "1")

    # Hyperparameters
    p = argparse.ArgumentParser()
    p.add_argument("--objective", type=str, default="binary:logistic")
    p.add_argument("--eval-metric", nargs="+", default=None)
    p.add_argument("--tree-method", type=str, default="hist")
    p.add_argument("--num-class", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0,
                   help="Number of training workers. 0=auto-detect from cluster topology")
    p.add_argument("--num-rounds", type=int, default=500)
    p.add_argument("--early-stopping", type=int, default=20)
    p.add_argument("--max-depth", type=int, default=7)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--min-child-weight", type=int, default=5)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-header", nargs="?", const="true", default="false",
                   help="Input CSV has no header (target is first column)")
    p.add_argument("--checkpoint-s3-uri", type=str, default=None,
                   help="S3 URI for Ray Train checkpoint storage (required for multi-node)")
    p.add_argument("--object-store-memory", type=float, default=-1.0,
                   help="Ray object store size in bytes. -1.0=auto (40%% of system RAM)")
    p.add_argument("--object-store-fraction", type=float, default=-1.0,
                   help="Ray object store as fraction of system RAM. -1.0=auto")
    p.add_argument("--class-weight-sample-rows", type=int, default=-1,
                   help="Rows to sample for class weight estimation. -1=auto (10M)")
    args, _ = p.parse_known_args()

    # Detect cluster topology
    hosts, current_host = _get_sagemaker_hosts()
    num_hosts = len(hosts) if hosts else 1
    use_gpu = num_gpus > 0

    # Auto-detect num_workers from cluster topology:
    #   GPU: 1 worker per GPU across all nodes (handles single-node multi-GPU)
    #   CPU: 1 worker per host (each worker uses all node CPUs via OpenMP nthread)
    if args.num_workers > 0:
        num_workers = args.num_workers
        worker_source = "user-specified"
    elif use_gpu:
        num_workers = num_gpus * num_hosts
        worker_source = f"auto: {num_gpus} GPUs/node x {num_hosts} node(s)"
    else:
        num_workers = num_hosts
        worker_source = f"auto: {num_hosts} CPU node(s)"

    print(f"Train:      {input_dir}")
    print(f"Val:        {val_dir or '(split from train)'}")
    print(f"Model dir:  {model_dir}")
    print(f"GPUs/node:  {num_gpus}")
    print(f"CPUs/node:  {num_cpus}")
    print(f"Nodes:      {num_hosts}")
    print(f"Workers:    {num_workers} ({worker_source})")
    print(f"Mode:       {'GPU' if use_gpu else 'CPU'}")
    print()

    # Bootstrap Ray cluster for multi-node, or start local
    multi_node = hosts is not None and num_hosts > 1

    # Multi-node Ray Train needs shared storage for checkpoints.
    # Use the S3 URI passed from the launch script, or None for single-node.
    storage_path = args.checkpoint_s3_uri if multi_node else None
    if multi_node and not storage_path:
        raise ValueError(
            "Multi-node training requires --checkpoint-s3-uri for shared "
            "checkpoint storage. Pass an S3 URI like s3://bucket/ray-checkpoints"
        )

    # Resolve object store size
    total_ram, available_ram, shm_size = _get_memory_info()
    object_store_bytes = _resolve_object_store_bytes(
        args.object_store_memory, args.object_store_fraction,
        total_ram, available_ram, shm_size,
    )
    if total_ram:
        print(f"System RAM:     {total_ram / 1e9:.1f} GB total, "
              f"{(available_ram or 0) / 1e9:.1f} GB available")
    if shm_size:
        print(f"/dev/shm:       {shm_size / 1e9:.1f} GB")
    if object_store_bytes:
        print(f"Object store:   {object_store_bytes / 1e9:.1f} GB")

    if multi_node:
        is_head = bootstrap_ray_cluster(
            hosts, current_host, num_gpus=num_gpus,
            object_store_bytes=object_store_bytes,
        )
        if not is_head:
            _wait_for_head_done()
            print("[Worker] Head completed. Exiting.")
            return
        # Multi-node: object store already configured at ray start time
        ray.init(address="auto")
    else:
        ray_kwargs = {}
        if object_store_bytes:
            ray_kwargs["object_store_memory"] = object_store_bytes
        ray.init(**ray_kwargs)

    print(f"Ray resources: {ray.cluster_resources()}\n")

    run_training(
        input_path=input_dir,
        val_path=val_dir,
        model_dir=model_dir,
        no_header=str(args.no_header).lower() in ("true", "1", "yes", ""),
        val_split=args.val_split,
        use_gpu=use_gpu,
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
        seed=args.seed,
        num_workers=num_workers,
        storage_path=storage_path,
        class_weight_sample_rows=args.class_weight_sample_rows,
    )

    ray.shutdown()


if __name__ == "__main__":
    main()
