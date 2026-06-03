# VectorDB-DS

A distributed vector database built from scratch in modern C++. Demonstrates high-performance systems programming, approximate nearest-neighbour search, and gRPC-based networking.

## Architecture

```
vectordb_client  ──gRPC──►  vectordb_server
                              ├── VectorStore     (string ID → float vector)
                              ├── HnswIndex       (O(log N) ANN search)
                              └── BruteForceIndex (exact k-NN, used for benchmarks)
```

The server owns a `VectorStore` and an `HnswIndex`. Clients insert vectors by ID (`Upsert`) and query nearest neighbours (`Search`). The index is rebuilt automatically on the first search after any upsert.

## Project Structure

```
VectorDB-DS/
├── proto/
│   └── vectordb.proto          # gRPC service definition
├── include/
│   ├── math_vector.h           # RAII float vector (Rule of Five)
│   ├── vector_store.h          # ID-addressed vector storage
│   ├── brute_force_index.h     # Exact k-NN (O(N·D) per query)
│   └── hnsw_index.h            # HNSW ANN index (O(log N) per query)
├── src/
│   ├── main.cpp                # Unit tests + benchmark driver
│   ├── server.cpp              # gRPC server wrapping the engine
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

This produces three binaries in `build/`:

| Binary | Description |
|--------|-------------|
| `vectordb_tests` | Unit test suite + HNSW/brute-force benchmark |
| `vectordb_server` | gRPC server (default port 50051) |
| `vectordb_client` | Demo client that upserts vectors and runs searches |

## Run

### Tests and benchmark

```bash
./build/vectordb_tests
```

Runs all unit tests (constructors, Rule of Five, VectorStore, BruteForce, HNSW hierarchy and recall), then a latency benchmark comparing brute-force vs HNSW search.

See [docs/benchmark.md](docs/benchmark.md) for how to run with the SIFT-1M dataset.

### gRPC server + client

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

Expected client output:

```
Connecting to localhost:50051...

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

The HNSW index is rebuilt automatically the first time `Search` is called after an `Upsert`. The rebuild log line appears in the server terminal:

```
Rebuilding HNSW index (5 vectors)... done (max_layer=0)
```

## Development

1. Add headers to `include/`, source files to `src/`
2. Register new `.cpp` files in `CMakeLists.txt` under the appropriate target
3. The proto file is at `proto/vectordb.proto` — edit it and CMake will regenerate the C++ stubs automatically on the next build
4. Rebuild from the project root: `cmake --build build`
