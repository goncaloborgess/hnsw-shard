#pragma once

#include "math_vector.h"
#include "vector_store.h"

#include <cstddef>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// HnswNode
//
// A node in the HNSW graph. Does NOT own vector data — that lives in
// VectorStore and is addressed by 'index'.
//
// neighbors[layer] holds up to M_max indices at that layer.
// The outer vector is jagged: a node assigned to layer L has
// neighbors[0..L] populated. A node at layer 0 only has neighbors[0].
// Allocating only the layers a node actually inhabits keeps memory tight
// and makes the max-layer of any node O(1) to query: neighbors.size() - 1.
// ---------------------------------------------------------------------------
struct HnswNode {
    std::size_t                           index;     // position in VectorStore
    std::vector<std::vector<std::size_t>> neighbors; // neighbors[layer] = neighbor indices
};

// ---------------------------------------------------------------------------
// HnswIndex
//
// Builds and queries a Hierarchical Navigable Small World graph.
// Holds a non-owning reference to VectorStore — the store must outlive
// the index, exactly as with BruteForceIndex.
//
// Hyperparameters:
//   M               — max neighbors per node per layer (default: 16).
//                     Layer 0 uses 2*M (M_max0) for a denser base layer.
//   ef_construction — candidate pool size during graph build (default: 200).
//                     Larger = higher graph quality, slower build.
//   mL              — layer scale factor, derived as 1/ln(M).
//                     Not a free parameter: its optimal value is a function
//                     of M and the Skip List probability derivation.
// ---------------------------------------------------------------------------
class HnswIndex {
public:
    explicit HnswIndex(const VectorStore& store,
                       std::size_t        M               = 16,
                       std::size_t        ef_construction = 200);

    // Build the graph over all vectors currently in the store.
    // Must be called once after the store is fully populated.
    void build();

    // k nearest neighbours of query, sorted ascending by distance.
    // Precondition: query.size() == store_.dimensionality()
    // NOTE: Step 2.1/2.2 placeholder — brute-force scan.
    //       Replaced by the real greedy traversal in Step 2.3.
    std::vector<SearchResult> search(const MathVector& query, std::size_t k) const;

    // --- Inspection accessors (used by tests and Step 2.4 benchmarking) ---

    // Total number of nodes in the graph.
    std::size_t size() const noexcept;

    // Highest layer index currently in the graph (0-based).
    int max_layer() const noexcept;

    // VectorStore index of the current graph entry point.
    // The entry point always lives at max_layer().
    std::size_t entry_point() const noexcept;

    // Number of nodes that exist at the given layer.
    // A node exists at layer L if its neighbors vector has size > L.
    std::size_t layer_population(int layer) const noexcept;

private:
    // Neighbor selection (simple distance heuristic).
    // Returns up to M indices from the candidate pool, closest first.
    // Isolated so the selection strategy can be upgraded independently.
    std::vector<std::size_t> select_neighbors(
        const std::vector<SearchResult>& candidates,
        std::size_t                      M) const;

    // Step 2.2 — Draw the maximum layer for a new node.
    // Formula: floor(-ln(uniform(0,1)) * mL)
    // Produces the geometric distribution that gives HNSW its
    // Skip List-equivalent sparsity and O(log N) complexity.
    int draw_layer();

    // Insert one vector (by VectorStore index) into all its layers.
    void insert_node(std::size_t node_idx);

    const VectorStore& store_;
    std::size_t        M_;
    std::size_t        ef_construction_;
    double             mL_;            // 1 / ln(M)

    // One HnswNode per vector, in insertion order.
    // Invariant: nodes_[i].index == i (build() inserts sequentially).
    std::vector<HnswNode> nodes_;

    std::size_t entry_point_; // VectorStore index of the entry point node
    int         max_layer_;   // highest layer in the graph

    std::mt19937 rng_;        // RNG for draw_layer() — seeded for reproducibility
};
