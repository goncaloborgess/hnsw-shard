#include "hnsw_index.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Constructor
//
// mL is derived as 1/ln(M). This is not arbitrary: it calibrates the
// geometric distribution used for layer assignment so that the expected
// number of nodes at layer L decreases by a factor of M for each step up.
// The result is the same sparsity ratio as a classical Skip List, which is
// the theoretical basis for HNSW's O(log N) complexity guarantee.
// ---------------------------------------------------------------------------
HnswIndex::HnswIndex(const VectorStore& store,
                     std::size_t        M,
                     std::size_t        ef_construction)
    : store_(store)
    , M_(M)
    , ef_construction_(ef_construction)
    , mL_(1.0 / std::log(static_cast<double>(M)))
    , entry_point_(0)
    , max_layer_(0)
    , rng_(42) // fixed seed: reproducible builds and benchmarks
{}

// ---------------------------------------------------------------------------
// build()
//
// Iterates the store in insertion order and calls insert_node() for each
// vector. The sequential order is deliberate: the first node bootstraps the
// entry point, and every subsequent insert can already use the graph to
// locate its neighbors.
// ---------------------------------------------------------------------------
void HnswIndex::build() {
    if (store_.empty()) return;

    nodes_.reserve(store_.size());

    for (std::size_t i = 0; i < store_.size(); ++i) {
        insert_node(i);
    }
}

std::size_t HnswIndex::size() const noexcept {
    return nodes_.size();
}

// ---------------------------------------------------------------------------
// Task 2.1b — select_neighbors (simple heuristic)
//
// Selects the best M candidates from the pool purely by distance.
// The candidates vector is not assumed to be sorted on entry.
//
// Why isolate this?
// The HNSW paper defines a second heuristic ("select neighbors heuristic")
// that prefers candidates which extend reach in new directions, avoiding
// the tight clustering that the simple version can produce. By keeping this
// function separate we can swap the strategy in one place, and both insert()
// and potential future re-wiring code pick up the change automatically.
// ---------------------------------------------------------------------------
std::vector<std::size_t> HnswIndex::select_neighbors(
    const std::vector<SearchResult>& candidates,
    std::size_t                      M) const
{
    // Work on a local sorted copy — do not mutate the caller's pool.
    std::vector<SearchResult> sorted = candidates;
    std::sort(sorted.begin(), sorted.end(),
              [](const SearchResult& a, const SearchResult& b) {
                  return a.distance < b.distance;
              });

    const std::size_t count = std::min(M, sorted.size());
    std::vector<std::size_t> result;
    result.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        result.push_back(sorted[i].index);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Task 2.1c — insert_node()
//
// Inserts a single vector (identified by its VectorStore index) into the
// graph at layer 0.
//
// Step 2.1 keeps every node at layer 0 only. Step 2.2 adds the layer draw
// and multi-layer wiring. The structure here is intentionally written to
// make that extension obvious: the layer loop (currently trivial) is where
// Step 2.2's logic slots in.
//
// Key invariant maintained throughout: BIDIRECTIONAL EDGES.
// If node A lists node B as a neighbor, B must also list A.
// Violation creates directed dead-zones where search can enter a region
// but cannot navigate within it, causing silent recall collapse.
// ---------------------------------------------------------------------------
void HnswIndex::insert_node(std::size_t node_idx) {
    // --- Allocate the node ---
    // Layer 0 only for now. neighbors is resized to 1 layer.
    HnswNode node;
    node.index = node_idx;
    node.neighbors.resize(1); // layer 0
    nodes_.push_back(std::move(node));

    // --- Bootstrap: first node becomes the entry point ---
    // There is nothing to connect to, so we return immediately.
    // The entry point is updated in Step 2.2 when nodes can reach higher layers.
    if (nodes_.size() == 1) {
        entry_point_ = node_idx;
        max_layer_   = 0;
        return;
    }

    // --- Find candidate neighbors via brute-force scan over existing nodes ---
    //
    // We scan every node already in nodes_ (all except the one just pushed)
    // and compute distances to the new vector.
    //
    // This O(N) scan is the temporary stand-in for search_layer(), which will
    // replace it in Step 2.3. At that point, insert_node() calls search_layer()
    // with ef = ef_construction_ to obtain the candidate pool in O(log N) time.
    //
    // Using ef_construction_ candidates (not just M) before pruning is
    // intentional: a larger initial pool gives select_neighbors() more material
    // to choose from, which produces higher-quality connections.
    const MathVector& query = store_.get(node_idx);

    std::vector<SearchResult> candidates;
    candidates.reserve(nodes_.size() - 1);

    for (std::size_t i = 0; i < nodes_.size() - 1; ++i) {
        const std::size_t idx  = nodes_[i].index;
        const float       dist = euclidean_distance(query, store_.get(idx));
        candidates.push_back({idx, dist});
    }

    // Cap the pool at ef_construction_ to bound the work done by select_neighbors.
    std::sort(candidates.begin(), candidates.end(),
              [](const SearchResult& a, const SearchResult& b) {
                  return a.distance < b.distance;
              });
    if (candidates.size() > ef_construction_) {
        candidates.resize(ef_construction_);
    }

    // --- Prune candidates to M neighbors ---
    const std::vector<std::size_t> chosen = select_neighbors(candidates, M_);

    // --- Write FORWARD edges: new node → chosen neighbors ---
    nodes_.back().neighbors[0] = chosen;

    // --- Write REVERSE edges: each chosen neighbor → new node ---
    //
    // Invariant: nodes_[i].index == i because build() inserts sequentially.
    // This lets us subscript nodes_ directly with a VectorStore index.
    //
    // Adding the reverse edge may push a neighbor's degree over M_.
    // When that happens, re-run select_neighbors over the neighbor's full
    // (now enlarged) neighbor list to prune it back to M_.
    // The evicted edge is simply dropped — the graph remains valid because
    // the remaining edges were a better set of M connections.
    for (const std::size_t neighbor_idx : chosen) {
        std::vector<std::size_t>& rev = nodes_[neighbor_idx].neighbors[0];
        rev.push_back(node_idx);

        if (rev.size() > M_) {
            // Recompute distances from the neighbor to all its current links.
            const MathVector& neighbor_vec = store_.get(neighbor_idx);
            std::vector<SearchResult> re_candidates;
            re_candidates.reserve(rev.size());
            for (const std::size_t nb : rev) {
                const float d = euclidean_distance(neighbor_vec, store_.get(nb));
                re_candidates.push_back({nb, d});
            }
            rev = select_neighbors(re_candidates, M_);
        }
    }
}

// ---------------------------------------------------------------------------
// search() — Step 2.1 placeholder
//
// Brute-force scan over all nodes. Functionally correct; used exclusively
// during Step 2.1 to verify that the graph was built without assertion
// failures and that SearchResult output matches BruteForceIndex.
// Replaced entirely by knn_search() in Step 2.3.
// ---------------------------------------------------------------------------
std::vector<SearchResult> HnswIndex::search(const MathVector& query,
                                             std::size_t        k) const
{
    if (query.size() != store_.dimensionality()) {
        throw std::invalid_argument("query dimensionality does not match store");
    }
    if (nodes_.empty()) return {};

    std::vector<SearchResult> results;
    results.reserve(nodes_.size());
    for (const HnswNode& node : nodes_) {
        const float dist = euclidean_distance(query, store_.get(node.index));
        results.push_back({node.index, dist});
    }

    const std::size_t actual_k = std::min(k, results.size());
    std::partial_sort(results.begin(), results.begin() + actual_k, results.end(),
                      [](const SearchResult& a, const SearchResult& b) {
                          return a.distance < b.distance;
                      });
    results.resize(actual_k);
    return results;
}
