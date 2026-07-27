# Deployment Guide

How to run the distributed system in containers, first with Docker Compose and
then on a local Kubernetes cluster.

---

## Docker

The [Dockerfile](../Dockerfile) is multi-stage: one `builder` stage compiles all
binaries against Ubuntu's gRPC packages, and two thin runtime stages copy out a
single binary each — `shard` and `coordinator`. Neither runtime image contains a
compiler or the source tree.

### Docker Compose

Run the full distributed system with a single command:

```bash
docker compose up --build -d
```

This starts 2 shard containers and 1 coordinator container on a Docker network.
The coordinator is exposed on port 50050:

```bash
./build/vectordb_client localhost:50050
```

Tear down:

```bash
docker compose down
```

### Building the images individually

```bash
docker build --target shard       -t vectordb-shard .
docker build --target coordinator -t vectordb-coordinator .
```

---

## Kubernetes

The `k8s/` directory contains manifests for deploying to a Kubernetes cluster:

- **StatefulSet** (2 replicas) for shards — each pod gets a stable DNS name via a headless Service
- **Deployment** (1 replica) for the coordinator — stateless, references shards by DNS
- **NodePort Service** on port 30050 for external access to the coordinator

The split mirrors the architecture: shards own data and therefore need stable
network identity, while the coordinator holds nothing but shard addresses and can
be replaced freely.

### Prerequisites

You need a local Kubernetes cluster. The easiest option is
[kind](https://kind.sigs.k8s.io/) (Kubernetes IN Docker):

```bash
brew install kind
kind create cluster --name vectordb
```

### Deploy

The images are built locally and must be loaded into the cluster before applying
the manifests:

```bash
# Load local images into the kind cluster
kind load docker-image vectordb-shard --name vectordb
kind load docker-image vectordb-coordinator --name vectordb

# Apply all manifests
kubectl apply -f k8s/

# Wait for all pods to be ready
kubectl wait --for=condition=ready pod -l app=vectordb-shard -n vectordb --timeout=60s
kubectl wait --for=condition=ready pod -l app=vectordb-coordinator -n vectordb --timeout=60s

# Check everything is running
kubectl get pods -n vectordb
kubectl get services -n vectordb
```

### Test

NodePort requires the node IP, which is inconvenient locally. Use `port-forward`
instead:

```bash
kubectl port-forward svc/vectordb-coordinator 50050:50050 -n vectordb &
./build/vectordb_client localhost:50050
```

Expected output is identical to the Docker Compose run above.

### Inspect logs

```bash
kubectl logs -l app=vectordb-shard -n vectordb
kubectl logs -l app=vectordb-coordinator -n vectordb
```

### Tear down

```bash
kubectl delete -f k8s/
kind delete cluster --name vectordb
```
