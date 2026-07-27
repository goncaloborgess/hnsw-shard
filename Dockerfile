# ---- Stage 1: Builder ----
FROM ubuntu:24.04 AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake make g++ \
        libgrpc++-dev libprotobuf-dev \
        protobuf-compiler protobuf-compiler-grpc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN cmake -S . -B build && \
    cmake --build build --target vectordb_server      --parallel && \
    cmake --build build --target vectordb_coordinator  --parallel && \
    cmake --build build --target vectordb_client       --parallel

# ---- Stage 2: Shard runtime ----
FROM ubuntu:24.04 AS shard

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgrpc++1.51t64 libprotobuf32t64 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/build/vectordb_server /usr/local/bin/vectordb_server

EXPOSE 50051
ENTRYPOINT ["vectordb_server"]

# ---- Stage 3: Coordinator runtime ----
FROM ubuntu:24.04 AS coordinator

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgrpc++1.51t64 libprotobuf32t64 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/build/vectordb_coordinator /usr/local/bin/vectordb_coordinator

EXPOSE 50050
ENTRYPOINT ["vectordb_coordinator"]
