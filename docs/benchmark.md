# Benchmark Guide

This document explains how to run the HNSW benchmark, where the data comes from,
and how to interpret every number in the output.

---

## Dataset

The benchmark uses the **SIFT-1M** dataset distributed by
[ann-benchmarks](https://github.com/erikbern/ann-benchmarks).

| Property | Value |
|----------|-------|
| Full dataset size | 1,000,000 vectors |
| Vectors used here | first 50,000 (controlled by `N` in `src/main.cpp`) |
| Dimensionality | 128 floats per vector |
| Vector type | SIFT image descriptors (Scale-Invariant Feature Transform) |
| Query set | first 100 vectors from the 10,000-query file |
| Distance metric | Euclidean (L2) |

SIFT descriptors are extracted from patches of natural images.
They have real cluster structure, which makes them a representative benchmark
for approximate nearest-neighbour (ANN) algorithms.

### Obtaining the data

Run the ann-benchmarks download script once:

```bash
git clone https://github.com/erikbern/ann-benchmarks.git
cd ann-benchmarks
pip install -r requirements.txt
python create_dataset.py --dataset sift-128-euclidean
```

This produces `data/sift-128-euclidean.hdf5` (≈ 1 GB).
Convert it to the fvecs binary format expected by this project:

```python
import h5py, numpy as np, struct, pathlib

with h5py.File("data/sift-128-euclidean.hdf5", "r") as f:
    train = np.array(f["train"], dtype=np.float32)   # (1_000_000, 128)
    test  = np.array(f["test"],  dtype=np.float32)   # (10_000, 128)

def write_fvecs(path, matrix):
    with open(path, "wb") as out:
        for row in matrix:
            out.write(struct.pack("i", len(row)))
            out.write(row.tobytes())

pathlib.Path("data").mkdir(exist_ok=True)
write_fvecs("data/sift_base.fvecs",  train)
write_fvecs("data/sift_query.fvecs", test)
```

Place both files in the `data/` directory at the project root.
If the files are absent the benchmark falls back to synthetic uniform random
vectors automatically.

### fvecs binary format

Each record in a `.fvecs` file is laid out as:

```
┌─────────────┬──────────────────────────────────────┐
│  int32 dim  │  float32 × dim                       │
│   4 bytes   │  dim × 4 bytes                       │
└─────────────┴──────────────────────────────────────┘
```

For SIFT-128 every record is 4 + 128 × 4 = **516 bytes**.

---

## Building and running

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/VectorDB-DS
```

The binary runs the unit-test suite first, then the benchmark.

---

## HNSW parameters

These constants are defined at the top of `run_benchmark()` in `src/main.cpp`.

| Parameter | Symbol | Default | Effect |
|-----------|--------|---------|--------|
| Dataset size | `N` | 50 000 | Number of vectors indexed |
| Dimensionality | `D` | 128 | Used only for the synthetic fallback |
| Neighbours returned | `K` | 10 | Size of the result set per query |
| Queries timed | `Q` | 100 | How many queries are measured |
| Max neighbours / layer | `M` | 16 | Graph connectivity; higher → better recall, more memory, slower build |
| Build beam width | `EF_CON` | 200 | Candidate pool during index construction; higher → better graph quality, slower build |
| Search beam width | `EF_SRCH` | 50 | Candidate pool during query; higher → better recall, higher latency |

`EF_SRCH` is the primary runtime trade-off knob.
Increasing it improves recall at the cost of latency; decreasing it does the opposite.

---

## Benchmark output walkthrough

```
  [1/4] loading base vectors... loaded 50000 SIFT vectors  (128-D)

  === Benchmark: Brute-Force vs HNSW ===
    dataset  : SIFT-1M (first 50000)
    k        : 10
    queries  : 100
    M        : 16  |  ef_construction: 200  |  ef_search: 50

  [2/4] building HNSW index... done  (19.27 s,  max_layer=4)
  [3/4] loading query vectors... loaded 100 SIFT queries

  [4/4] running timed queries...

  +-----------------+-----------+-----------+-----------+
  |                 |  avg (ms) |  min (ms) |  max (ms) |
  +-----------------+-----------+-----------+-----------+
  | brute-force     |     3.679 |     3.201 |    17.950 |
  | HNSW            |     0.155 |     0.073 |     0.321 |
  +-----------------+-----------+-----------+-----------+

  speedup  (BF avg / HNSW avg) : 23.7x
  recall@10                   : 0.973
  HNSW build time              : 19.27 s
```

### Steps

| Step | Description |
|------|-------------|
| `[1/4]` | Load up to `N` vectors from `data/sift_base.fvecs` into the `VectorStore`. Falls back to synthetic data if the file is missing. |
| `[2/4]` | Build the HNSW graph. **Timed.** `max_layer` is the number of hierarchy levels created (probabilistic; typical range 3–5 for N=50 000). |
| `[3/4]` | Load up to `Q` vectors from `data/sift_query.fvecs`. Done outside the timed section so I/O does not affect latency numbers. |
| `[4/4]` | Run both brute-force and HNSW search for every query vector. Each method is timed independently per query. |

### Latency table

Each cell is measured across `Q = 100` queries.

| Column | Meaning |
|--------|---------|
| `avg (ms)` | Mean query latency in milliseconds — the primary performance number. |
| `min (ms)` | Fastest single query — represents best-case cache/branch behaviour. |
| `max (ms)` | Slowest single query — can be an outlier due to OS scheduling or cache cold starts. |

The brute-force `max` of **17.950 ms** vs its average of **3.679 ms** is a common OS-scheduling
spike on the first query; the HNSW numbers are tighter because graph traversal touches
far fewer cache lines.

### Speedup

```
speedup = brute_force_avg / hnsw_avg
        = 3.679 ms / 0.155 ms
        = 23.7×
```

This is a **latency speedup**, not a throughput number.
It answers: "for a single query, how many times faster is HNSW than scanning every vector?"

At N = 50 000 vectors and 128 dimensions the brute-force scan performs
50 000 × 128 = **6.4 million** floating-point operations per query.
HNSW explores roughly `ef_search × log(N)` ≈ 50 × 17 ≈ **850 nodes**,
evaluating far fewer distances.

### Recall\@10

```
recall@10 = |HNSW top-10 ∩ BF top-10| / 10
```

For each query, the brute-force result is treated as the exact ground truth.
The formula counts how many of HNSW's 10 returned neighbours are also in the
10 true nearest neighbours, then divides by 10.
The value is averaged across all 100 queries.

| Value | Interpretation |
|-------|---------------|
| 1.000 | Perfect — HNSW returns the exact same neighbours as brute force every time |
| 0.973 | 9.73 out of 10 correct on average — 1 neighbour missed per ~37 queries |
| 0.462 | (synthetic data result) — random 128-D vectors have no cluster structure, so the locality assumption breaks down |
| < 0.5 | Effectively unusable for most applications |

A recall of **0.973** with a **23.7× speedup** is a good operating point for
SIFT-128 at these parameters.
Most production ANN systems target recall ≥ 0.95.

### Build time

The **19.27 s** build time is paid once when the index is constructed from scratch.
In a real deployment the index is serialised to disk after building and loaded
on subsequent starts (see Chapter 3 of this project).

---

## Tuning reference

Adjust `EF_SRCH` (search beam width) to trade recall for latency without rebuilding.

| `ef_search` | Expected recall (SIFT-50K) | Relative latency |
|-------------|---------------------------|-----------------|
| 10 | ~0.85 | ~0.4× baseline |
| 50 | ~0.97 | 1× (default) |
| 100 | ~0.99 | ~2× |
| 200 | ~1.00 | ~4× |

To change it, modify `EF_SRCH` in `src/main.cpp` and rebuild.

Increase `M` or `EF_CON` to improve graph quality (raises recall ceiling and
reduces the recall drop at low `ef_search`), at the cost of a longer build time
and higher memory usage.
