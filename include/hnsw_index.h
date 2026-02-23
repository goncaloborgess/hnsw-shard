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
//                     This cost is paid once at index build time.
//   ef_search       — beam width during query (default: 50).
//                     Larger = higher recall, slower queries.
//                     This is the primary knob for the accuracy/speed tradeoff
//                     at runtime — it can be tuned without rebuilding the index.
//   mL              — layer scale factor, derived as 1/ln(M).
//                     Not a free parameter; computed from M.
// ---------------------------------------------------------------------------
class HnswIndex {
public:
    explicit HnswIndex(const VectorStore& store,
                       std::size_t        M               = 16,
                       std::size_t        ef_construction = 200,
                       std::size_t        ef_search       = 50);

    // Build the graph over all vectors currently in the store.
    // Must be called once after the store is fully populated.
    void build();

    // k nearest neighbours of query, sorted ascending by distance.
    // Uses multi-layer greedy traversal (Step 2.3).
    // Precondition: query.size() == store_.dimensionality()
    std::vector<SearchResult> search(const MathVector& query, std::size_t k) const;

    // --- Inspection accessors (used by tests and Step 2.4 benchmarking) ---

    std::size_t size()       const noexcept;
    int         max_layer()  const noexcept;
    std::size_t entry_point() const noexcept;
    std::size_t layer_population(int layer) const noexcept;

private:
    // Step 2.3a — Single-layer greedy beam search.
    //
    // Starting from ep_idx, explores the graph at 'layer' maintaining:
    //   - a min-heap of candidates (closest unexplored node at top)
    //   - a max-heap of results   (furthest accepted result at top)
    //
    // Terminates when every remaining candidate is farther than the
    // worst accepted result — no further improvement is possible.
    // Returns up to ef results in unspecified order; caller sorts.
    std::vector<SearchResult> search_layer(const MathVector& query,
                                           std::size_t       ep_idx,
                                           std::size_t       ef,
                                           int               layer) const;

    // Neighbor selection (simple distance heuristic).
    std::vector<std::size_t> select_neighbors(
        const std::vector<SearchResult>& candidates,
        std::size_t                      M) const;

    // Draw the maximum layer for a new node: floor(-ln(U) * mL)
    int draw_layer();

    // Insert one vector (by VectorStore index) into all its layers.
    void insert_node(std::size_t node_idx);

    const VectorStore& store_;
    std::size_t        M_;
    std::size_t        ef_construction_;
    std::size_t        ef_search_;      // beam width at query time
    double             mL_;            // 1 / ln(M)

    // One HnswNode per vector, in insertion order.
    // Invariant: nodes_[i].index == i (build() inserts sequentially).
    std::vector<HnswNode> nodes_;

    std::size_t entry_point_;
    int         max_layer_;

    std::mt19937 rng_;
};
