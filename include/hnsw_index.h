#pragma once

#include "math_vector.h"
#include "vector_store.h"

#include <cstddef>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Task 2.1a — HnswNode
//
// A node in the HNSW graph. It does NOT own vector data — that lives in
// VectorStore and is indexed by 'index'. The node's only job is to record
// its adjacency lists at each layer it participates in.
//
// neighbors[layer] holds up to M indices of neighbor nodes at that layer.
// The outer vector is sparse by design: a node assigned to layer L has
// neighbors[0..L] populated and nothing above — the jagged shape is
// intentional and reflects the hierarchy.
// ---------------------------------------------------------------------------
struct HnswNode {
    std::size_t                           index;     // position in VectorStore
    std::vector<std::vector<std::size_t>> neighbors; // neighbors[layer] = neighbor indices
};

// ---------------------------------------------------------------------------
// Task 2.1b — HnswIndex
//
// Builds and queries an HNSW graph over a VectorStore.
// Like BruteForceIndex, the index holds a non-owning reference to the store.
// The store must outlive the index.
//
// Hyperparameters:
//   M               — max neighbors per node per layer (default: 16)
//                     Controls the degree of each node. Higher M = better
//                     recall, more memory, slower inserts.
//   ef_construction — candidate pool size during graph build (default: 200)
//                     Larger pool = higher quality graph, slower build.
//                     This cost is paid once at index construction time.
//
// mL is derived from M as 1/ln(M) and is not exposed as a parameter because
// it has a mathematically optimal value for a given M.
// ---------------------------------------------------------------------------
class HnswIndex {
public:
    explicit HnswIndex(const VectorStore& store,
                       std::size_t        M                = 16,
                       std::size_t        ef_construction  = 200);

    // Inserts all vectors currently in the store into the graph.
    // Must be called once after the store is fully populated.
    void build();

    // Search for the k nearest neighbours of query.
    // Returns results sorted ascending by distance.
    // Precondition: query.size() == store_.dimensionality()
    // NOTE: In Step 2.1 this is a brute-force scan used to verify graph
    //       construction correctness. Step 2.3 replaces it with the real
    //       greedy traversal.
    std::vector<SearchResult> search(const MathVector& query, std::size_t k) const;

    // Number of nodes in the graph.
    std::size_t size() const noexcept;

private:
    // Task 2.1b — Neighbor selection (simple heuristic).
    //
    // Given a pool of (index, distance) candidates, returns the indices of
    // the M closest ones. Isolated as a function so the selection strategy
    // can be upgraded to the full HNSW diversity heuristic without touching
    // insert or search logic.
    std::vector<std::size_t> select_neighbors(
        const std::vector<SearchResult>& candidates,
        std::size_t                      M) const;

    // Task 2.1c — Insert a single VectorStore index into the graph.
    // In Step 2.1 all nodes are assigned to layer 0 only.
    // Step 2.2 introduces the layer draw and multi-layer wiring.
    void insert_node(std::size_t node_idx);

    const VectorStore& store_;
    std::size_t        M_;
    std::size_t        ef_construction_;
    double             mL_;             // 1 / ln(M), used in Step 2.2

    // One HnswNode per vector in the store, in insertion order.
    // Invariant: nodes_[i].index == i (build() inserts sequentially).
    // This lets us use a VectorStore index as a direct subscript into nodes_.
    std::vector<HnswNode> nodes_;

    std::size_t entry_point_; // VectorStore index of the graph entry point
    int         max_layer_;   // highest layer currently in the graph

    std::mt19937 rng_;        // seeded RNG for layer assignment (Step 2.2)
};
