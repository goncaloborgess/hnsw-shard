# VectorDB-DS

A distributed vector database built from scratch in modern C++. Demonstrates high-performance systems programming, approximate nearest-neighbour search, gRPC networking, and distributed scatter-gather query execution.

## Architecture

```
                          ┌──gRPC──► Shard 0 (vectordb_server :50051)
                          │            ├── VectorStore
vectordb_client ──gRPC──► Coordinator  │   └── HnswIndex
                          │            │
                          └──gRPC──► Shard 1 (vectordb_server :50052)
                                       ├── VectorStore
                                       └── HnswIndex
```

**Shards** are stateful gRPC servers, each holding a partitioned subset of vectors in memory with a local HNSW index. **The Coordinator** is a stateless proxy that routes upserts to the correct shard (FNV-1a hash partitioning) and fans out searches to all shards concurrently (`std::async`), then merges the partial top-k results into a global top-k.

The client talks to the coordinator using the same `VectorDBService` proto — it is unaware of the shards behind it.

## Project Structure

```
VectorDB-DS/
├── proto/
│   └── vectordb.proto          # gRPC service definition (Upsert + Search)
├── include/
│   ├── math_vector.h           # RAII float vector (Rule of Five)
│   ├── vector_store.h          # ID-addressed vector storage
│   ├── brute_force_index.h     # Exact k-NN (O(N·D) per query)
│   ├── hnsw_index.h            # HNSW ANN index (O(log N) per query)
│   └── partitioner.h           # FNV-1a hash-based shard assignment
├── src/
│   ├── main.cpp                # Unit tests + benchmark driver
│   ├── server.cpp              # Shard: gRPC server wrapping the engine
│   ├── coordinator.cpp         # Coordinator: scatter-gather proxy
│   ├── client.cpp              # gRPC client demo
│   ├── mathVector.cpp
│   ├── vector_store.cpp
│   ├── brute_force_index.cpp
│   └── hnsw_index.cpp
├── docs/
│   └── benchmark.md            # HNSW benchmark guide (SIFT-1M dataset)
└── CMakeLists.txt
```

## Prerequisites

- CMake ≥ 3.15
- C++17-capable compiler (clang++ or g++)
- gRPC and Protobuf (installed via Anaconda or Homebrew)

**Anaconda** (what this project uses):
```bash
conda install -c conda-forge grpc-cpp protobuf
```

**Homebrew** alternative:
```bash
brew install grpc protobuf
```

## Build

```bash
cmake -S . -B build
cmake --build build --parallel
```

This produces four binaries in `build/`:

| Binary | Description |
|--------|-------------|
| `vectordb_tests` | Unit test suite + HNSW/brute-force benchmark |
| `vectordb_server` | Shard server — gRPC server with HNSW engine (default port 50051) |
| `vectordb_coordinator` | Coordinator — scatter-gather proxy over multiple shards |
| `vectordb_client` | Demo client that upserts vectors and runs searches |

## Run

### Tests and benchmark

```bash
./build/vectordb_tests
```

Runs all unit tests (constructors, Rule of Five, VectorStore, BruteForce, HNSW hierarchy and recall, partitioner), then a latency benchmark comparing brute-force vs HNSW search.

See [docs/benchmark.md](docs/benchmark.md) for how to run with the SIFT-1M dataset.

### Single-node mode

In one terminal:

```bash
./build/vectordb_server            # listens on 0.0.0.0:50051
./build/vectordb_server 50052      # custom port
```

In a second terminal:

```bash
./build/vectordb_client            # connects to localhost:50051
./build/vectordb_client host:port  # custom address
```

### Distributed mode (scatter-gather)

Start two shards and the coordinator:

```bash
# Terminal 1 — Shard 0
./build/vectordb_server 50051

# Terminal 2 — Shard 1
./build/vectordb_server 50052

# Terminal 3 — Coordinator
./build/vectordb_coordinator 50050 localhost:50051 localhost:50052

# Terminal 4 — Client (talks to coordinator)
./build/vectordb_client localhost:50050
```

The coordinator hashes each vector ID to determine which shard receives it. On search, it queries all shards in parallel and merges the results:

```
Connecting to localhost:50050...

[1] Inserting vectors
  Upserted 'a' — created
  Upserted 'b' — created
  Upserted 'c' — created
  Upserted 'd' — created
  Upserted 'e' — created

[2] Searching for nearest to (2.1, 2.1), k=3
  Top-3 results:
    id="b"  distance=0.141421
    id="c"  distance=1.27279
    id="a"  distance=1.55563

[3] Overwriting 'b' to (4.9, 4.9)
  Upserted 'b' — updated

[4] Searching again for nearest to (2.1, 2.1), k=3
  Top-3 results:
    id="c"  distance=1.27279
    id="a"  distance=1.55563
    id="d"  distance=2.68701
```

The results are identical whether the client talks to a single shard or the distributed coordinator — the scatter-gather merge produces the same global top-k.

## Development

1. Add headers to `include/`, source files to `src/`
2. Register new `.cpp` files in `CMakeLists.txt` under the appropriate target
3. The proto file is at `proto/vectordb.proto` — edit it and CMake will regenerate the C++ stubs automatically on the next build
4. Rebuild from the project root: `cmake --build build`
