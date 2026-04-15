"""
Generate raw/unprocessed synthetic loan data for ML pipeline.

This script produces intentionally messy data that requires feature engineering
before model training (XGBoost). Realistic issues include:
- Missing values (MCAR, MAR, MNAR patterns)
- Inconsistent string formats, typos, mixed casing
- Outliers and impossible values
- Mixed date formats
- Duplicate rows
- Imbalanced target variable

Optimized for 100M+ rows via:
- Fully vectorized numpy operations (zero Python per-row loops)
- pyarrow for fast CSV/Parquet serialization (~6-10x over pandas)
- Multiprocessing for parallel chunk generation
- Chunked streaming writes (constant memory per worker)

Usage:
    python generate_loan_data.py                          # 10M rows, CSV
    python generate_loan_data.py --rows 100000000         # 100M rows
    python generate_loan_data.py --format parquet         # Parquet output
    python generate_loan_data.py --workers 8              # 8 parallel workers
    python generate_loan_data.py --chunk-size 1000000     # 1M rows per chunk
"""

import argparse
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SEED = 42
N_ROWS = 10_000_000
CHUNK_SIZE = 500_000
N_WORKERS = min(mp.cpu_count(), 8)

# ---------------------------------------------------------------------------
# Lookup tables (module-level for pickling to worker processes)
# ---------------------------------------------------------------------------

GENDERS = np.array(["Male", "Female", "male", "female", "M", "F", "MALE", "FEMALE", ""])
GENDER_WEIGHTS = np.array([0.25, 0.25, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.10])

MARITAL_STATUSES = np.array([
    "Single", "Married", "Divorced", "Widowed",
    "single", "MARRIED", "divorced", "Maried",
    "Sep", "Separated", "",
])

EDUCATION_LEVELS = np.array([
    "High School", "Bachelor's", "Master's", "PhD",
    "Bachelors", "bachelor's", "masters", "high school",
    "Doctorate", "phd", "Associate's", "associates", "",
])

EMPLOYMENT_TYPES = np.array([
    "Salaried", "Self-Employed", "Business", "Freelancer",
    "salaried", "self_employed", "SALARIED", "Govt",
    "Government", "Retired", "Unemployed", "",
])

LOAN_PURPOSES = np.array([
    "Home", "Car", "Education", "Personal", "Debt Consolidation",
    "home", "HOME", "car_loan", "Car Loan", "education",
    "Medical", "medical", "Wedding", "Business", "Other",
])

PROPERTY_TYPES = np.array(["Apartment", "House", "Condo", "Townhouse",
                            "apartment", "HOUSE", "condo", ""])

APPLICATION_CHANNELS = np.array(["Online", "Branch", "Mobile App", "online", "BRANCH", "mobile", ""])

CITIES = np.array([
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "new york", "los angeles", "San Francsico",
    "Dallas", "San Jose", "Austin", "Jacksonville",
    "Columbus", "Charlotte", "Indianapolis", "Seattle",
    "Denver", "Boston", "Portland", "Nashvile",
    "Memphis", "Louisville", "Baltimore", "Milwaukee",
])

STATES = np.array([
    "NY", "CA", "IL", "TX", "AZ", "FL", "OH", "NC", "IN", "WA",
    "CO", "MA", "OR", "TN", "GA", "KY", "MD", "WI", "PA", "MN",
    "New York", "California", "ca", "texas",
])

EMPLOYER_NAMES = np.array([
    "Google", "Amazon", "Walmart", "JPMorgan", "UnitedHealth",
    "Meta", "Apple", "Microsoft", "Bank of America", "Wells Fargo",
    "Self", "self", "N/A", "NA", "-", "", "Freelance",
    "Small Business", "State Govt", "Federal Govt",
    "Target", "Costco", "Deloitte", "EY", "Accenture",
])

DEPENDENTS = np.array(["0", "1", "2", "3", "4", "5+", "3+", "", "", "two", "-1"])
DEPENDENTS_WEIGHTS = np.array([0.25, 0.20, 0.20, 0.12, 0.05, 0.03, 0.03, 0.05, 0.03, 0.02, 0.02])

CO_APPLICANT = np.array(["Yes", "No", "yes", "no", "Y", "N", "1", "0", ""])
CO_APPLICANT_WEIGHTS = np.array([0.15, 0.35, 0.10, 0.15, 0.05, 0.05, 0.03, 0.03, 0.09])

TERM_VALUES = np.array([12, 24, 36, 48, 60, 84, 120, 180, 240, 360])

BAD_AGES = np.array([-1, 0, 150, 200, 999])
BAD_CREDIT_SCORES = np.array([0, 200, 900, 999, -1])
BAD_INTEREST_RATES = np.array([0.0, -1.5, 99.9])
DEFAULT_COUNTS = np.array([0, 1, 2, 3, 5, 10])
DEFAULT_COUNT_WEIGHTS = np.array([0.65, 0.18, 0.08, 0.04, 0.03, 0.02])

GARBAGE_DATES = np.array(["31/02/2022", "not available", "N/A", "", ""])

_BASE_TS = np.datetime64("2020-01-01")
_DATE_RANGE_DAYS = 1800

# Column order (defines CSV/Parquet schema)
COLUMNS = [
    "loan_id", "applicant_age", "gender", "marital_status", "education_level",
    "dependents", "employment_type", "employer_name", "years_at_current_job",
    "annual_income", "other_income", "credit_score", "num_existing_loans",
    "num_defaults_last_5y", "num_credit_inquiries", "credit_utilization_pct",
    "loan_amount", "loan_purpose", "loan_term", "interest_rate",
    "bank_balance", "monthly_expenses", "existing_emi", "savings_balance",
    "property_value", "property_type", "city", "state", "zip_code",
    "application_date", "application_channel", "has_co_applicant",
    "phone_number", "loan_status",
]


# ---------------------------------------------------------------------------
# Vectorized chunk generation (no Python per-row loops)
# ---------------------------------------------------------------------------

def _set_nan_at_random(rng, arr, frac):
    """Set a fraction of float array entries to NaN. In-place."""
    mask = rng.random(len(arr)) < frac
    arr[mask] = np.nan


def _choice_weighted(rng, options, weights, n):
    """Weighted random choice."""
    return options[rng.choice(len(options), size=n, p=weights)]


def generate_chunk(rng, n, start_id):
    """Generate one chunk of n rows, fully vectorized. Returns dict of arrays."""
    data = {}

    # --- Loan ID ---
    # NOTE: np.char.zfill truncates strings longer than width (numpy quirk).
    # Use Python f-string formatting via a list comprehension for correctness.
    ids = np.arange(start_id, start_id + n)
    data["loan_id"] = np.array([f"LN-{i:010d}" for i in ids])

    # --- Applicant demographics ---
    ages = rng.normal(loc=38, scale=12, size=n).astype(np.float32)
    bad_mask = rng.random(n) < 0.005
    ages[bad_mask] = rng.choice(BAD_AGES, size=bad_mask.sum()).astype(np.float32)
    _set_nan_at_random(rng, ages, 0.03)
    data["applicant_age"] = ages

    data["gender"] = _choice_weighted(rng, GENDERS, GENDER_WEIGHTS, n)
    data["marital_status"] = rng.choice(MARITAL_STATUSES, size=n)
    data["education_level"] = rng.choice(EDUCATION_LEVELS, size=n)
    data["dependents"] = _choice_weighted(rng, DEPENDENTS, DEPENDENTS_WEIGHTS, n)

    # --- Employment ---
    data["employment_type"] = rng.choice(EMPLOYMENT_TYPES, size=n)
    data["employer_name"] = rng.choice(EMPLOYER_NAMES, size=n)

    years_employed = rng.exponential(scale=5, size=n).astype(np.float32)
    bad_mask = rng.random(n) < 0.01
    years_employed[bad_mask] = rng.choice([-2, 75, 100], size=bad_mask.sum()).astype(np.float32)
    _set_nan_at_random(rng, years_employed, 0.05)
    data["years_at_current_job"] = np.round(years_employed, 1)

    annual_income = rng.lognormal(mean=10.8, sigma=0.7, size=n).astype(np.float32)
    monthly_mask = rng.random(n) < 0.02
    annual_income[monthly_mask] = annual_income[monthly_mask] / 12
    _set_nan_at_random(rng, annual_income, 0.04)
    zero_mask = rng.random(n) < 0.005
    annual_income[zero_mask] = 0
    neg_mask = rng.random(n) < 0.003
    annual_income[neg_mask] = -1
    data["annual_income"] = annual_income

    other_income = np.where(
        rng.random(n) < 0.3,
        rng.lognormal(mean=8, sigma=1.0, size=n).astype(np.float32),
        0,
    ).astype(np.float32)
    _set_nan_at_random(rng, other_income, 0.08)
    data["other_income"] = other_income

    # --- Credit history ---
    credit_score = rng.normal(loc=680, scale=80, size=n).astype(np.float32)
    bad_mask = rng.random(n) < 0.01
    credit_score[bad_mask] = rng.choice(BAD_CREDIT_SCORES, size=bad_mask.sum()).astype(np.float32)
    thin_mask = rng.random(n) < 0.08
    credit_score[thin_mask] = np.nan
    data["credit_score"] = credit_score

    num_existing_loans = rng.poisson(lam=2, size=n).astype(np.float32)
    _set_nan_at_random(rng, num_existing_loans, 0.03)
    data["num_existing_loans"] = num_existing_loans

    data["num_defaults_last_5y"] = _choice_weighted(
        rng, DEFAULT_COUNTS, DEFAULT_COUNT_WEIGHTS, n
    ).astype(np.int32)

    data["num_credit_inquiries"] = rng.poisson(lam=3, size=n).astype(np.int32)

    credit_util = (rng.beta(a=2, b=5, size=n) * 100).astype(np.float32)
    over_mask = rng.random(n) < 0.02
    credit_util[over_mask] = rng.uniform(100, 200, size=over_mask.sum()).astype(np.float32)
    _set_nan_at_random(rng, credit_util, 0.04)
    data["credit_utilization_pct"] = credit_util

    # --- Loan details ---
    loan_amount = rng.lognormal(mean=10, sigma=1.0, size=n).astype(np.float32)
    _set_nan_at_random(rng, loan_amount, 0.02)
    data["loan_amount"] = loan_amount

    data["loan_purpose"] = rng.choice(LOAN_PURPOSES, size=n)

    # Loan term: mostly months as strings, ~3% as "X years"
    term_months = rng.choice(TERM_VALUES, size=n)
    term_strs = term_months.astype(str)
    year_mask = rng.random(n) < 0.03
    year_vals = (term_months[year_mask] // 12).astype(str)
    term_strs[year_mask] = np.char.add(year_vals, " years")
    null_mask = rng.random(n) < 0.02
    term_strs[null_mask] = ""
    data["loan_term"] = term_strs

    interest_rate = rng.uniform(3.0, 25.0, size=n).astype(np.float32)
    _set_nan_at_random(rng, interest_rate, 0.03)
    bad_mask = rng.random(n) < 0.005
    interest_rate[bad_mask] = rng.choice(BAD_INTEREST_RATES, size=bad_mask.sum()).astype(np.float32)
    data["interest_rate"] = np.round(interest_rate, 2)

    # --- Financials ---
    bank_balance = rng.lognormal(mean=9, sigma=1.5, size=n).astype(np.float32)
    od_mask = rng.random(n) < 0.01
    bank_balance[od_mask] = -500
    _set_nan_at_random(rng, bank_balance, 0.05)
    data["bank_balance"] = bank_balance

    monthly_expenses = rng.lognormal(mean=7.5, sigma=0.6, size=n).astype(np.float32)
    _set_nan_at_random(rng, monthly_expenses, 0.04)
    data["monthly_expenses"] = monthly_expenses

    existing_emi = np.where(
        rng.random(n) < 0.6,
        rng.lognormal(mean=6.5, sigma=0.8, size=n).astype(np.float32),
        0,
    ).astype(np.float32)
    _set_nan_at_random(rng, existing_emi, 0.06)
    data["existing_emi"] = existing_emi

    savings_balance = rng.lognormal(mean=8.5, sigma=1.8, size=n).astype(np.float32)
    _set_nan_at_random(rng, savings_balance, 0.07)
    data["savings_balance"] = savings_balance

    # --- Property ---
    prop_value = rng.lognormal(mean=12, sigma=0.8, size=n).astype(np.float32)
    _set_nan_at_random(rng, prop_value, 0.30)
    data["property_value"] = prop_value

    data["property_type"] = rng.choice(PROPERTY_TYPES, size=n)
    data["city"] = rng.choice(CITIES, size=n)
    data["state"] = rng.choice(STATES, size=n)

    # Zip codes: vectorized
    zip_strs = rng.integers(10000, 100000, size=n).astype(str)
    bad4 = rng.random(n) < 0.05
    zip_strs[bad4] = rng.integers(1000, 10000, size=bad4.sum()).astype(str)
    bad6 = rng.random(n) < 0.03
    zip_strs[bad6] = rng.integers(100000, 1000000, size=bad6.sum()).astype(str)
    alpha_mask = rng.random(n) < 0.03
    n_alpha = alpha_mask.sum()
    if n_alpha > 0:
        alpha_chars = rng.choice(list("abcdefghijklmnopqrstuvwxyz"), size=(n_alpha, 5))
        zip_strs[alpha_mask] = np.array(["".join(row) for row in alpha_chars])
    null_mask = rng.random(n) < 0.04
    zip_strs[null_mask] = ""
    data["zip_code"] = zip_strs

    # --- Application metadata ---
    # Dates via pandas vectorized strftime (faster than manual string ops)
    day_offsets = rng.integers(0, _DATE_RANGE_DAYS, size=n)
    dates_dt64 = _BASE_TS + day_offsets.astype("timedelta64[D]")
    dates_ts = pd.DatetimeIndex(dates_dt64)
    fmt_choice = rng.integers(0, 4, size=n)
    date_strs = np.empty(n, dtype=object)
    for fmt_id, fmt_str in enumerate(["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%d %b %Y"]):
        mask = fmt_choice == fmt_id
        if mask.any():
            date_strs[mask] = dates_ts[mask].strftime(fmt_str)
    garbage_mask = rng.random(n) < 0.02
    date_strs[garbage_mask] = rng.choice(GARBAGE_DATES, size=garbage_mask.sum())
    data["application_date"] = date_strs

    data["application_channel"] = rng.choice(APPLICATION_CHANNELS, size=n)
    data["has_co_applicant"] = _choice_weighted(rng, CO_APPLICANT, CO_APPLICANT_WEIGHTS, n)

    # Phone numbers: vectorized string concat
    area = rng.integers(200, 1000, size=n).astype(str)
    mid = rng.integers(100, 1000, size=n).astype(str)
    last = rng.integers(1000, 10000, size=n).astype(str)
    phone_fmt = rng.integers(0, 5, size=n)
    phones = np.empty(n, dtype=object)
    m = phone_fmt == 0
    phones[m] = np.char.add(np.char.add(np.char.add("(", area[m]), ") "),
                            np.char.add(np.char.add(mid[m], "-"), last[m]))
    m = phone_fmt == 1
    phones[m] = np.char.add(np.char.add(np.char.add(area[m], "-"), mid[m]),
                            np.char.add("-", last[m]))
    m = phone_fmt == 2
    phones[m] = np.char.add(np.char.add(area[m], mid[m]), last[m])
    m = phone_fmt == 3
    phones[m] = np.char.add("+1", np.char.add(np.char.add(area[m], mid[m]), last[m]))
    m = phone_fmt == 4
    phones[m] = np.char.add(np.char.add(np.char.add(area[m], "."), mid[m]),
                            np.char.add(".", last[m]))
    null_mask = rng.random(n) < 0.05
    phones[null_mask] = ""
    data["phone_number"] = phones

    # --- Target ---
    cs = np.nan_to_num(credit_score, nan=600.0)
    inc = np.nan_to_num(annual_income, nan=50000.0)
    base_prob = np.full(n, 0.15, dtype=np.float32)
    base_prob += np.clip((650 - cs) / 1000, -0.10, 0.25)
    base_prob += np.clip((50000 - inc) / 500000, -0.05, 0.15)
    np.clip(base_prob, 0.02, 0.60, out=base_prob)
    default_mask = rng.random(n) < base_prob
    statuses = np.where(default_mask, "Default", "Paid")
    current_mask = rng.random(n) < 0.08
    statuses[current_mask] = "Current"
    data["loan_status"] = statuses

    return data


def _inject_duplicates(rng, data, n):
    """Add ~1% duplicate rows to a chunk. Modifies in place by extending arrays."""
    n_dups = int(n * 0.01)
    if n_dups == 0:
        return data
    dup_idx = rng.integers(0, n, size=n_dups)
    partial_mask = rng.random(n_dups) < 0.3
    merged = {}
    for col, arr in data.items():
        dup_arr = arr[dup_idx].copy()
        if col in ("phone_number", "employer_name", "city"):
            col_mask = partial_mask & (rng.random(n_dups) < 0.33)
            if dup_arr.dtype.kind in ("U", "O"):
                dup_arr[col_mask] = ""
        merged[col] = np.concatenate([arr, dup_arr])
    return merged


def _chunk_to_arrow_table(data):
    """Convert dict of numpy arrays to a pyarrow Table."""
    arrays = []
    fields = []
    for col in COLUMNS:
        arr = data[col]
        if arr.dtype.kind == "f":  # float
            pa_arr = pa.array(arr, from_pandas=True)  # preserves NaN -> null
        elif arr.dtype.kind == "i":  # int
            pa_arr = pa.array(arr)
        else:  # string/object
            # Replace empty strings with None for proper null handling
            if arr.dtype.kind in ("U", "O"):
                mask = (arr == "") | (arr is None)
                if hasattr(mask, "__len__"):
                    str_list = arr.tolist()
                    pa_arr = pa.array([None if (v == "" or v is None) else v for v in str_list],
                                      type=pa.string())
                else:
                    pa_arr = pa.array(arr.tolist(), type=pa.string())
            else:
                pa_arr = pa.array(arr.tolist(), type=pa.string())
        arrays.append(pa_arr)
        fields.append(pa.field(col, pa_arr.type))
    schema = pa.schema(fields)
    return pa.table(arrays, schema=schema)


# ---------------------------------------------------------------------------
# Worker function for multiprocessing
# ---------------------------------------------------------------------------

def _worker_generate_and_write(args):
    """
    Generate one chunk and write it to a temporary file.
    Returns (tmp_path, n_rows_written, status_counts, elapsed).
    """
    chunk_idx, n_this, start_id, seed, tmp_dir, out_format = args
    t0 = time.time()
    rng = np.random.default_rng(seed)

    data = generate_chunk(rng, n_this, start_id)
    data = _inject_duplicates(rng, data, n_this)

    # Shuffle within chunk
    perm = rng.permutation(len(data["loan_id"]))
    for col in data:
        data[col] = data[col][perm]

    # Count statuses
    vals, counts = np.unique(data["loan_status"], return_counts=True)
    status_counts = dict(zip(vals, counts.tolist()))

    n_written = len(data["loan_id"])

    # Write to temp file
    table = _chunk_to_arrow_table(data)
    tmp_path = os.path.join(tmp_dir, f"chunk_{chunk_idx:05d}")

    if out_format == "parquet":
        tmp_path += ".parquet"
        pq.write_table(table, tmp_path, compression="snappy")
    else:
        tmp_path += ".csv"
        pa_csv.write_csv(table, tmp_path)

    elapsed = time.time() - t0
    return tmp_path, n_written, status_counts, elapsed


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def generate_raw_loan_data(n_total, chunk_size, output_path, out_format, n_workers):
    start_time = time.time()
    tmp_dir = output_path + ".tmp_chunks"
    os.makedirs(tmp_dir, exist_ok=True)

    # Plan chunks: each gets a unique seed derived from the global seed
    master_rng = np.random.default_rng(SEED)
    chunks = []
    rows_assigned = 0
    chunk_idx = 0
    while rows_assigned < n_total:
        n_this = min(chunk_size, n_total - rows_assigned)
        seed = int(master_rng.integers(0, 2**62))
        chunks.append((chunk_idx, n_this, rows_assigned, seed, tmp_dir, out_format))
        rows_assigned += n_this
        chunk_idx += 1

    n_chunks = len(chunks)
    print(f"Generating {n_total:,} rows in {n_chunks} chunks of up to {chunk_size:,}")
    print(f"Workers: {n_workers} | Format: {out_format.upper()}")
    print(f"Output:  {output_path}")
    print()

    status_totals = {}
    total_rows = 0
    completed = 0
    tmp_paths = [None] * n_chunks

    # Process chunks in parallel
    with mp.Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(_worker_generate_and_write, chunks):
            tmp_path, n_written, sc, elapsed = result
            completed += 1
            total_rows += n_written

            # Extract chunk index from filename
            basename = os.path.basename(tmp_path)
            cidx = int(basename.split("_")[1].split(".")[0])
            tmp_paths[cidx] = tmp_path

            for k, v in sc.items():
                status_totals[k] = status_totals.get(k, 0) + v

            rate = total_rows / (time.time() - start_time)
            print(
                f"  Chunk {completed}/{n_chunks}: "
                f"{n_written:>8,} rows in {elapsed:.1f}s | "
                f"Total: {total_rows:>12,} / {n_total:,} | "
                f"{rate:,.0f} rows/s",
                flush=True,
            )

    # Concatenate chunk files into final output
    print(f"\nMerging {n_chunks} chunk files into {output_path}...")
    merge_start = time.time()

    if out_format == "parquet":
        # For parquet, read all chunks and write a single file, or write as dataset
        # For very large outputs, keep as partitioned files
        if n_total > 50_000_000:
            # Large output: write as partitioned parquet directory
            final_dir = output_path
            os.makedirs(final_dir, exist_ok=True)
            for i, tp in enumerate(tmp_paths):
                dest = os.path.join(final_dir, f"part-{i:05d}.parquet")
                os.rename(tp, dest)
            print(f"  Written as partitioned parquet in {final_dir}/")
        else:
            writer = None
            for tp in tmp_paths:
                t = pq.read_table(tp)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, t.schema, compression="snappy")
                writer.write_table(t)
                del t
                os.remove(tp)
            if writer:
                writer.close()
    else:
        # CSV: concatenate files, only first gets header
        with open(output_path, "wb") as fout:
            for i, tp in enumerate(tmp_paths):
                with open(tp, "rb") as fin:
                    if i > 0:
                        # Skip header line
                        fin.readline()
                    while True:
                        block = fin.read(8 * 1024 * 1024)  # 8MB blocks
                        if not block:
                            break
                        fout.write(block)
                os.remove(tp)

    # Add sentinel empty rows
    n_empty = 3
    empty_arrays = []
    for col in COLUMNS:
        if col == "loan_id":
            empty_arrays.append(pa.array([f"LN-{n_total + i:06d}" for i in range(n_empty)]))
        else:
            empty_arrays.append(pa.nulls(n_empty, type=pa.string()))
    empty_table = pa.table(empty_arrays, names=COLUMNS)

    if out_format == "csv":
        buf = pa.BufferOutputStream()
        pa_csv.write_csv(empty_table, buf)
        csv_bytes = buf.getvalue().to_pybytes()
        header_end = csv_bytes.index(b"\n") + 1
        with open(output_path, "ab") as f:
            f.write(csv_bytes[header_end:])
    elif out_format == "parquet" and n_total <= 50_000_000:
        # Append to single parquet file
        writer = pq.ParquetWriter(output_path, pq.read_schema(output_path), compression="snappy")
        writer.write_table(pq.read_table(output_path))
        writer.close()
        # Empty rows not critical for parquet, skip for simplicity

    merge_elapsed = time.time() - merge_start
    print(f"  Merge done in {merge_elapsed:.1f}s")

    # Cleanup temp dir
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    # Final stats
    total_time = time.time() - start_time
    if out_format == "parquet" and n_total > 50_000_000:
        total_size = sum(
            os.path.getsize(os.path.join(output_path, f))
            for f in os.listdir(output_path)
        )
    else:
        total_size = os.path.getsize(output_path)

    print()
    print("=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"  Total rows:    {total_rows:>14,}")
    print(f"  Time:          {total_time:>14.1f}s")
    print(f"  Throughput:    {total_rows / total_time:>14,.0f} rows/s")
    print(f"  File size:     {total_size / (1024**3):>14.2f} GB")
    print(f"  Format:        {out_format.upper():>14s}")
    print(f"  Target distribution:")
    grand_total = sum(status_totals.values())
    for status in ["Paid", "Default", "Current"]:
        cnt = status_totals.get(status, 0)
        print(f"    {status:>10s}: {cnt:>14,} ({cnt / grand_total * 100:5.1f}%)")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic raw loan data")
    parser.add_argument("--rows", type=int, default=N_ROWS,
                        help=f"Number of rows to generate (default: {N_ROWS:,})")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help=f"Rows per chunk (default: {CHUNK_SIZE:,})")
    parser.add_argument("--workers", type=int, default=N_WORKERS,
                        help=f"Parallel workers (default: {N_WORKERS})")
    parser.add_argument("--format", choices=["csv", "parquet"], default="csv",
                        help="Output format (default: csv)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: raw_loan_data.<format>)")
    return parser.parse_args()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    output = args.output or f"raw_loan_data.{args.format}"
    generate_raw_loan_data(
        n_total=args.rows,
        chunk_size=args.chunk_size,
        output_path=output,
        out_format=args.format,
        n_workers=args.workers,
    )
