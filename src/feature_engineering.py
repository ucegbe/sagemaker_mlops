"""
Ray-based feature engineering pipeline for loan default prediction.

Transforms raw synthetic loan data into ML-ready features for XGBoost.
Uses Ray Data for automatic parallelism across single-node and multi-node clusters.

Pipeline stages:
  1. Ingest & deduplicate
  2. String normalization & typo correction
  3. Numeric cleaning & outlier clamping
  4. Date parsing & temporal features
  5. Loan term parsing
  6. Imputation (median/mode — requires global aggregates)
  7. Derived features (ratios, bins, interactions)
  8. Encoding (target binarization, ordinal, label)
  9. Drop non-predictive columns & write output

Usage:
    # Single node (auto-detects local resources)
    python feature_engineering.py --input raw_loan_data.csv --output features.parquet

    # Multi-node (connect to existing Ray cluster)
    # NOTE: input/output paths must be accessible from ALL nodes (S3, HDFS, NFS — not local disk)
    python feature_engineering.py \\
        --input s3://my-bucket/raw_loan_data.csv \\
        --output s3://my-bucket/features.parquet \\
        --ray-address auto

    # Multi-node with explicit head node address
    python feature_engineering.py \\
        --input s3://my-bucket/raw_loan_data.csv \\
        --output s3://my-bucket/features.parquet \\
        --ray-address ray://head-node:10001

    # Tune parallelism
    python feature_engineering.py --input raw_loan_data.csv --output features.parquet \\
        --batch-size 65536 --num-cpus 16
"""

import argparse
import time

import numpy as np
import pandas as pd
import ray
import ray.data


# ---------------------------------------------------------------------------
# String normalization mappings
# ---------------------------------------------------------------------------

GENDER_MAP = {
    "male": "male", "m": "male", "female": "female", "f": "female",
}

MARITAL_MAP = {
    "single": "single", "married": "married", "maried": "married",
    "divorced": "divorced", "widowed": "widowed",
    "sep": "separated", "separated": "separated",
}

EDUCATION_MAP = {
    "high school": "high_school",
    "bachelor's": "bachelors", "bachelors": "bachelors",
    "master's": "masters", "masters": "masters",
    "phd": "doctorate", "doctorate": "doctorate",
    "associate's": "associates", "associates": "associates",
}

EMPLOYMENT_MAP = {
    "salaried": "salaried",
    "self-employed": "self_employed", "self_employed": "self_employed",
    "business": "business", "freelancer": "freelancer", "freelance": "freelancer",
    "govt": "government", "government": "government",
    "retired": "retired", "unemployed": "unemployed",
}

PURPOSE_MAP = {
    "home": "home",
    "car": "car", "car_loan": "car", "car loan": "car",
    "education": "education",
    "personal": "personal",
    "debt consolidation": "debt_consolidation",
    "medical": "medical", "wedding": "wedding",
    "business": "business", "other": "other",
}

PROPERTY_MAP = {
    "apartment": "apartment", "house": "house",
    "condo": "condo", "townhouse": "townhouse",
}

CHANNEL_MAP = {
    "online": "online", "branch": "branch",
    "mobile app": "mobile", "mobile": "mobile",
}

CO_APPLICANT_MAP = {
    "yes": 1, "y": 1, "1": 1,
    "no": 0, "n": 0, "0": 0,
}

DEPENDENTS_MAP = {
    "0": 0, "1": 1, "2": 2, "two": 2, "3": 3, "3+": 3,
    "4": 4, "5+": 5, "-1": np.nan,
}

STATE_TO_CODE = {
    "new york": "NY", "california": "CA", "ca": "CA", "texas": "TX",
    "ny": "NY", "il": "IL", "tx": "TX", "az": "AZ", "fl": "FL",
    "oh": "OH", "nc": "NC", "in": "IN", "wa": "WA", "co": "CO",
    "ma": "MA", "or": "OR", "tn": "TN", "ga": "GA", "ky": "KY",
    "md": "MD", "wi": "WI", "pa": "PA", "mn": "MN",
}

CITY_TYPOS = {
    "san francsico": "san francisco",
    "nashvile": "nashville",
}

EDUCATION_ORDINAL = {
    "high_school": 0, "associates": 1, "bachelors": 2,
    "masters": 3, "doctorate": 4,
}

DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%d %b %Y"]


# ---------------------------------------------------------------------------
# Combined transform: all row-parallel stages in one pass
#
# Combining stages into fewer map_batches calls reduces Ray scheduling overhead
# and avoids repeated serialization/deserialization between stages.
# ---------------------------------------------------------------------------

def transform_batch(batch: pd.DataFrame) -> pd.DataFrame:
    """Stages 1-5: clean, normalize, parse — single pass over each batch."""

    # ---- Stage 1: Deduplicate & filter ----
    batch = batch.drop_duplicates(subset=["loan_id"], keep="first")
    batch = batch.dropna(subset=["loan_id"])
    batch = batch[batch["loan_status"].isin(["Default", "Paid"])]

    if len(batch) == 0:
        return batch

    # ---- Stage 2: String normalization ----
    batch["gender"] = _map_col(batch["gender"], GENDER_MAP)
    batch["marital_status"] = _map_col(batch["marital_status"], MARITAL_MAP)
    batch["education_level"] = _map_col(batch["education_level"], EDUCATION_MAP)
    batch["employment_type"] = _map_col(batch["employment_type"], EMPLOYMENT_MAP)
    batch["loan_purpose"] = _map_col(batch["loan_purpose"], PURPOSE_MAP)
    batch["property_type"] = _map_col(batch["property_type"], PROPERTY_MAP)
    batch["application_channel"] = _map_col(batch["application_channel"], CHANNEL_MAP)

    batch["has_co_applicant"] = _map_col(
        batch["has_co_applicant"], CO_APPLICANT_MAP
    ).astype("Float32")

    batch["dependents"] = _map_col(
        batch["dependents"], DEPENDENTS_MAP
    ).astype("Float32")

    # city: lowercase + typo fix
    city_lower = batch["city"].astype(str).str.strip().str.lower()
    city_lower = city_lower.replace(["", "nan", "none"], np.nan)
    batch["city"] = city_lower.replace(CITY_TYPOS)

    # state: normalize to 2-letter codes
    state_lower = batch["state"].astype(str).str.strip().str.lower()
    state_lower = state_lower.replace(["", "nan", "none"], np.nan)
    batch["state"] = state_lower.map(STATE_TO_CODE)

    # zip_code: must be 5-digit numeric; invalid -> NaN
    zip_str = batch["zip_code"].astype(str).str.strip()
    valid_zip = zip_str.str.match(r"^\d{5}(\.0)?$", na=False)
    batch["zip_code"] = zip_str.where(valid_zip, np.nan)
    batch.loc[valid_zip, "zip_code"] = (
        batch.loc[valid_zip, "zip_code"]
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )

    # ---- Stage 3: Numeric cleaning ----
    age = batch["applicant_age"]
    batch["applicant_age"] = age.where(age.between(18, 100), np.nan)

    yrs = batch["years_at_current_job"]
    batch["years_at_current_job"] = yrs.where(yrs.between(0, 50), np.nan)

    inc = batch["annual_income"]
    batch["annual_income"] = inc.where(inc > 0, np.nan)

    cs = batch["credit_score"]
    batch["credit_score"] = cs.where(cs.between(300, 850), np.nan)

    ir = batch["interest_rate"]
    batch["interest_rate"] = ir.where(ir.between(0.5, 50), np.nan)

    batch["credit_utilization_pct"] = batch["credit_utilization_pct"].clip(upper=100)

    oi = batch["other_income"]
    batch["other_income"] = oi.where(oi >= 0, np.nan)

    bb = batch["bank_balance"]
    batch["bank_balance"] = bb.where(bb > -10000, np.nan)

    # ---- Stage 4: Date parsing ----
    raw_dates = batch["application_date"]
    parsed = pd.Series(pd.NaT, index=batch.index, dtype="datetime64[ns]")

    remaining_mask = parsed.isna()
    for fmt in DATE_FORMATS:
        if not remaining_mask.any():
            break
        attempt = pd.to_datetime(raw_dates[remaining_mask], format=fmt, errors="coerce")
        parsed[remaining_mask] = attempt
        remaining_mask = parsed.isna()

    batch["app_year"] = parsed.dt.year.astype("Float32")
    batch["app_month"] = parsed.dt.month.astype("Float32")
    batch["app_day_of_week"] = parsed.dt.dayofweek.astype("Float32")
    batch["app_quarter"] = parsed.dt.quarter.astype("Float32")

    ref_date = pd.Timestamp("2020-01-01")
    batch["app_days_since_ref"] = (parsed - ref_date).dt.days.astype("Float32")

    batch = batch.drop(columns=["application_date"])

    # ---- Stage 5: Loan term parsing ----
    term = batch["loan_term"].astype(str).str.strip().str.lower()
    term = term.replace(["", "nan", "none"], np.nan)

    year_mask = term.str.contains("year", na=False)
    num_mask = (~year_mask) & term.notna()

    batch["loan_term_months"] = np.nan
    if year_mask.any():
        batch.loc[year_mask, "loan_term_months"] = (
            term[year_mask].str.extract(r"(\d+)", expand=False).astype(float) * 12
        )
    if num_mask.any():
        batch.loc[num_mask, "loan_term_months"] = pd.to_numeric(term[num_mask], errors="coerce")

    batch["loan_term_months"] = batch["loan_term_months"].astype("Float32")
    batch = batch.drop(columns=["loan_term"])

    return batch


def _map_col(series: pd.Series, mapping: dict) -> pd.Series:
    """Lowercase then map through a dict. Unmapped values become NaN."""
    lower = series.astype(str).str.strip().str.lower()
    lower = lower.replace(["", "nan", "none", "n/a", "na", "-"], np.nan)
    return lower.map(mapping)


# ---------------------------------------------------------------------------
# Imputation: compute global aggregates, then apply
# ---------------------------------------------------------------------------

NUMERIC_IMPUTE_COLS = [
    "applicant_age", "years_at_current_job", "annual_income", "other_income",
    "credit_score", "num_existing_loans", "credit_utilization_pct",
    "loan_amount", "interest_rate", "bank_balance", "monthly_expenses",
    "existing_emi", "savings_balance", "property_value",
    "has_co_applicant", "dependents", "loan_term_months",
    "app_year", "app_month", "app_day_of_week", "app_quarter",
    "app_days_since_ref",
]

CATEGORICAL_IMPUTE_COLS = [
    "gender", "marital_status", "education_level", "employment_type",
    "loan_purpose", "property_type", "application_channel",
    "city", "state", "zip_code",
]


def compute_imputation_values(ds: ray.data.Dataset) -> dict:
    """Compute median for numerics and mode for categoricals from a sample."""
    count = ds.count()
    sample_frac = min(1.0, 2_000_000 / max(count, 1))
    sample_ds = ds.random_sample(sample_frac, seed=42) if sample_frac < 1.0 else ds
    sample_df = sample_ds.to_pandas()

    impute_vals = {}
    for col in NUMERIC_IMPUTE_COLS:
        if col in sample_df.columns:
            med = sample_df[col].median()
            impute_vals[col] = float(med) if pd.notna(med) else 0.0

    for col in CATEGORICAL_IMPUTE_COLS:
        if col in sample_df.columns:
            mode_vals = sample_df[col].mode()
            impute_vals[col] = mode_vals.iloc[0] if len(mode_vals) > 0 else "unknown"

    return impute_vals


def make_impute_fn(impute_vals: dict):
    """Return a batch function closed over precomputed imputation values."""
    def impute_batch(batch: pd.DataFrame) -> pd.DataFrame:
        for col, fill_val in impute_vals.items():
            if col in batch.columns:
                batch[col] = batch[col].fillna(fill_val)
        return batch
    return impute_batch


# ---------------------------------------------------------------------------
# Derived features + encoding + column drop — single pass
# ---------------------------------------------------------------------------

def make_final_transform_fn(label_maps: dict):
    """Return a batch function that derives features, encodes, and drops columns."""
    def final_transform_batch(batch: pd.DataFrame) -> pd.DataFrame:
        # ---- Derived features ----
        inc = batch["annual_income"]
        monthly_inc = inc / 12
        emi = batch["existing_emi"]
        loan_amt = batch["loan_amount"]
        expenses = batch["monthly_expenses"]
        prop_val = batch["property_value"]
        deps = batch["dependents"]
        savings = batch["savings_balance"]
        term = batch["loan_term_months"]

        batch["debt_to_income"] = np.where(inc > 0, (emi * 12) / inc, np.nan)
        batch["loan_to_income"] = np.where(inc > 0, loan_amt / inc, np.nan)
        batch["total_income"] = inc + batch["other_income"].fillna(0)
        batch["monthly_surplus"] = monthly_inc - expenses - emi
        batch["loan_to_value"] = np.where(prop_val > 0, loan_amt / prop_val, np.nan)
        batch["income_per_dependent"] = np.where(deps >= 0, inc / (deps + 1), inc)
        batch["has_other_income"] = (batch["other_income"] > 0).astype(np.int8)
        batch["savings_to_loan"] = np.where(loan_amt > 0, savings / loan_amt, np.nan)
        batch["balance_to_expenses"] = np.where(
            expenses > 0, batch["bank_balance"] / expenses, np.nan
        )
        batch["total_debt_exposure"] = emi * term + loan_amt

        batch["credit_score_bin"] = pd.cut(
            batch["credit_score"],
            bins=[0, 580, 670, 740, 800, 850],
            labels=[0, 1, 2, 3, 4],
            include_lowest=True,
        ).astype("Float32")

        batch["age_bin"] = pd.cut(
            batch["applicant_age"],
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=[0, 1, 2, 3, 4, 5],
            include_lowest=True,
        ).astype("Float32")

        batch["payment_per_month"] = np.where(term > 0, loan_amt / term, np.nan)
        batch["payment_burden"] = np.where(
            monthly_inc > 0, batch["payment_per_month"] / monthly_inc, np.nan
        )

        # ---- Encoding ----
        # Target: Default=1, Paid=0
        batch["target"] = (batch["loan_status"] == "Default").astype(np.int8)
        batch = batch.drop(columns=["loan_status"])

        # Education ordinal
        batch["education_ordinal"] = (
            batch["education_level"].map(EDUCATION_ORDINAL).astype("Float32")
        )

        # Label-encode categoricals
        for col, mapping in label_maps.items():
            if col in batch.columns:
                batch[f"{col}_encoded"] = batch[col].map(mapping).astype("Float32")

        # ---- Drop non-predictive columns ----
        drop_cols = [
            "loan_id", "phone_number", "employer_name",
            "gender", "marital_status", "education_level", "employment_type",
            "loan_purpose", "property_type", "application_channel",
            "city", "state", "zip_code",
        ]
        to_drop = [c for c in drop_cols if c in batch.columns]
        batch = batch.drop(columns=to_drop)

        # Ensure all columns are numeric
        for col in batch.columns:
            if batch[col].dtype == object:
                batch[col] = pd.to_numeric(batch[col], errors="coerce")

        # Downcast float64 -> float32
        float64_cols = batch.select_dtypes(include=["float64"]).columns
        if len(float64_cols) > 0:
            batch[float64_cols] = batch[float64_cols].astype(np.float32)

        # Reorder so target is the first column
        cols = batch.columns.tolist()
        if "target" in cols:
            cols.remove("target")
            cols = ["target"] + cols
            batch = batch[cols]

        return batch

    return final_transform_batch


def compute_label_maps(ds: ray.data.Dataset) -> dict:
    """Build label encoding maps for low-cardinality categoricals."""
    cols = [
        "gender", "marital_status", "employment_type", "loan_purpose",
        "property_type", "application_channel", "city", "state",
    ]
    count = ds.count()
    sample_frac = min(1.0, 2_000_000 / max(count, 1))
    sample_ds = ds.random_sample(sample_frac, seed=42) if sample_frac < 1.0 else ds
    sample_df = sample_ds.to_pandas()

    label_maps = {}
    for col in cols:
        if col in sample_df.columns:
            uniques = sorted(sample_df[col].dropna().unique().tolist())
            label_maps[col] = {v: i for i, v in enumerate(uniques)}
    return label_maps


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def _ensure_dir_all_nodes(path):
    """Create a directory on every node in the Ray cluster."""
    @ray.remote
    def _mkdir(p):
        import os
        os.makedirs(p, exist_ok=True)

    nodes = ray.nodes()
    alive_ips = set(n["NodeManagerAddress"] for n in nodes if n["Alive"])
    ray.get([
        _mkdir.options(resources={f"node:{ip}": 0.001}).remote(path)
        for ip in alive_ips
    ])


def _write_dataset(ds, path, out_format, no_header):
    """Write a Ray Dataset to CSV or Parquet."""
    _ensure_dir_all_nodes(path)

    ds = ds.repartition(100)

    if out_format == "csv":
        if no_header:
            import pyarrow.csv as pa_csv
            ds.write_csv(
                path,
                arrow_csv_args_fn=lambda: {
                    "write_options": pa_csv.WriteOptions(include_header=False)
                },
            )
        else:
            ds.write_csv(path)
    else:
        ds.write_parquet(path)


def run_pipeline(input_path: str, output_path: str, out_format: str,
                 batch_size: int, no_header: bool = False,
                 val_split: float = 0.0) -> None:
    total_start = time.time()

    # ---- Step 1: Read raw data ----
    step_start = time.time()
    print("[1/5] Reading raw data...")
    if input_path.endswith(".parquet") or (
        not input_path.endswith(".csv")
        and "." not in input_path.rsplit("/", 1)[-1]
    ):
        ds = ray.data.read_parquet(input_path)
    else:
        ds = ray.data.read_csv(input_path)
    row_count = ds.count()
    print(f"       {row_count:,} rows loaded in {time.time() - step_start:.1f}s")

    # ---- Step 2: Transform (clean + normalize + parse) ----
    step_start = time.time()
    print("[2/5] Cleaning, normalizing, parsing (combined pass)...")
    ds = ds.map_batches(transform_batch, batch_format="pandas", batch_size=batch_size)
    ds = ds.materialize()
    clean_count = ds.count()
    print(f"       {row_count:,} -> {clean_count:,} rows"
          f" ({row_count - clean_count:,} removed)"
          f" in {time.time() - step_start:.1f}s")

    # ---- Step 3: Imputation (requires global aggregates) ----
    step_start = time.time()
    print("[3/5] Computing imputation values & filling missing...")
    impute_vals = compute_imputation_values(ds)
    print(f"       Computed medians/modes for {len(impute_vals)} columns")
    ds = ds.map_batches(
        make_impute_fn(impute_vals), batch_format="pandas", batch_size=batch_size
    )
    ds = ds.materialize()
    print(f"       Done in {time.time() - step_start:.1f}s")

    # ---- Step 4: Derive features + encode + drop (combined) ----
    step_start = time.time()
    print("[4/5] Computing label maps...")
    label_maps = compute_label_maps(ds)
    total_categories = sum(len(v) for v in label_maps.values())
    print(f"       {len(label_maps)} columns, {total_categories} categories")

    print("       Deriving features, encoding, dropping columns...")
    ds = ds.map_batches(
        make_final_transform_fn(label_maps),
        batch_format="pandas",
        batch_size=batch_size,
    )
    ds = ds.materialize()
    print(f"       Done in {time.time() - step_start:.1f}s")

    # ---- Step 5: Split (optional) + Write output ----
    step_start = time.time()
    header_note = ", no header" if (out_format == "csv" and no_header) else ""

    if val_split > 0:
        print(f"[5/6] Splitting train/val ({1 - val_split:.0%} / {val_split:.0%})...")
        train_ds, val_ds = ds.train_test_split(test_size=val_split, seed=42)
        train_count = train_ds.count()
        val_count = val_ds.count()
        print(f"       Train: {train_count:,}  Val: {val_count:,}")
        print(f"       Done in {time.time() - step_start:.1f}s")

        step_start = time.time()
        train_path = output_path.rstrip("/") + "/train"
        val_path = output_path.rstrip("/") + "/val"
        print(f"[6/6] Writing output ({out_format.upper()}{header_note})...")
        _write_dataset(train_ds, train_path, out_format, no_header)
        _write_dataset(val_ds, val_path, out_format, no_header)
        print(f"       Done in {time.time() - step_start:.1f}s")
    else:
        print(f"[5/5] Writing output ({out_format.upper()}{header_note})...")
        _write_dataset(ds, output_path, out_format, no_header)
        print(f"       Done in {time.time() - step_start:.1f}s")

    # ---- Summary ----
    total_time = time.time() - total_start
    print()
    print("=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)
    print(f"  Input rows:      {row_count:>14,}")
    print(f"  Output rows:     {clean_count:>14,}")
    if val_split > 0:
        print(f"  Train rows:      {train_count:>14,}")
        print(f"  Val rows:        {val_count:>14,}")
    print(f"  Total time:      {total_time:>14.1f}s")
    print(f"  Throughput:      {clean_count / total_time:>14,.0f} rows/s")
    print(f"  Format:          {out_format.upper():>14s}")
    if val_split > 0:
        print(f"  Train output:    {train_path}")
        print(f"  Val output:      {val_path}")
    else:
        print(f"  Output:          {output_path}")

    # Schema report — get column names from the materialized dataset
    schema = ds.schema()
    col_names = schema.names
    n_cols = len(col_names)
    print(f"  Output columns:  {n_cols:>14}")
    print(f"\n  Feature list ({n_cols} columns, target first):")
    for name in col_names:
        print(f"    {name}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ray feature engineering for loan default prediction"
    )
    parser.add_argument("--input", type=str, default="raw_loan_data.csv",
                        help="Path to raw data (CSV or Parquet)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: features)")
    parser.add_argument("--format", choices=["csv", "parquet"], default="parquet",
                        help="Output format (default: parquet)")
    parser.add_argument("--batch-size", type=int, default=32768,
                        help="Rows per batch for map_batches (default: 32768)")
    parser.add_argument("--num-cpus", type=int, default=None,
                        help="Number of CPUs for local Ray (default: all)")
    parser.add_argument("--no-header", action="store_true",
                        help="Omit column header row in CSV output (ignored for Parquet)")
    parser.add_argument("--val-split", type=float, default=0.0,
                        help="Validation split fraction, e.g. 0.2 for 80/20 "
                             "(default: 0.0 = no split)")
    parser.add_argument("--ray-address", type=str, default=None,
                        help="Ray cluster address (e.g., 'auto' or 'ray://host:10001'). "
                             "Omit for local single-node mode.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    ray_init_kwargs = {}
    if args.ray_address:
        ray_init_kwargs["address"] = args.ray_address
        print(f"Connecting to Ray cluster at {args.ray_address}...")
    else:
        if args.num_cpus:
            ray_init_kwargs["num_cpus"] = args.num_cpus
        print("Starting local Ray instance...")

    ray.init(**ray_init_kwargs)
    print(f"Ray cluster resources: {ray.cluster_resources()}")
    print()

    output = args.output or f"features"
    print(f"outputPath: {output}")
    run_pipeline(
        input_path=args.input,
        output_path=output,
        out_format=args.format,
        batch_size=args.batch_size,
        no_header=args.no_header,
        val_split=args.val_split,
    )

    ray.shutdown()
