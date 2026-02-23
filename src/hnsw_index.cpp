#include "hnsw_index.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Constructor
//
// mL = 1/ln(M) is the layer assignment scale factor. It is not arbitrary.
//
// The HNSW paper proves that setting mL = 1/ln(M) produces a geometric
// distribution where the probability that a node reaches layer L is (1/M)^L.
// This makes the expected number of nodes at each layer decrease by a factor
// of M per step upward — identical to the sparsity of a Skip List with
// promotion probability 1/M. That sparsity is what gives both structures
// their O(log N) search complexity.
//
// M_max0 = 2*M is the neighbor cap for layer 0. Layer 0 carries the full
// dataset and is the densest layer. Allowing 2*M connections there
// (vs M everywhere else) improves recall without a meaningful memory cost
// because layer 0 is the only layer all N nodes inhabit.
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
    , rng_(42)
{}

void HnswIndex::build() {
    if (store_.empty()) return;
    nodes_.reserve(store_.size());
    for (std::size_t i = 0; i < store_.size(); ++i) {
        insert_node(i);
    }
}

// ---------------------------------------------------------------------------
// Inspection accessors
// ---------------------------------------------------------------------------

std::size_t HnswIndex::size() const noexcept {
    return nodes_.size();
}

int HnswIndex::max_layer() const noexcept {
    return max_layer_;
}

std::size_t HnswIndex::entry_point() const noexcept {
    return entry_point_;
}

// A node "exists at layer L" if its neighbors jagged vector has at least
// L+1 inner vectors, i.e., neighbors.size() > static_cast<size_t>(layer).
std::size_t HnswIndex::layer_population(int layer) const noexcept {
    if (layer < 0) return 0;
    std::size_t count = 0;
    for (const HnswNode& node : nodes_) {
        if (static_cast<int>(node.neighbors.size()) > layer) {
            ++count;
        }
    }
    return count;
}

// ---------------------------------------------------------------------------
// select_neighbors
//
// Simple heuristic: return the M closest candidates by distance.
// Sorting a local copy preserves the caller's candidate pool ordering.
// ---------------------------------------------------------------------------
std::vector<std::size_t> HnswIndex::select_neighbors(
    const std::vector<SearchResult>& candidates,
    std::size_t                      M) const
{
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
// Step 2.2 — draw_layer()
//
// Samples the maximum layer for a new node using the formula:
//
//     l = floor( -ln(uniform(0,1)) * mL )
//
// This is the inverse-CDF transform of the geometric distribution.
// -ln(U) where U ~ Uniform(0,1) gives an Exponential(1) random variable.
// Multiplying by mL scales it so that the floor hits 0 with probability
// 1 - 1/M, hits 1 with probability 1/M - 1/M^2, and so on.
//
// The clamp to std::numeric_limits<double>::min() prevents ln(0) = -inf
// in the astronomically unlikely event that the RNG returns exactly 0.
// ---------------------------------------------------------------------------
int HnswIndex::draw_layer() {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    double r = uniform(rng_);
    if (r < std::numeric_limits<double>::min()) {
        r = std::numeric_limits<double>::min();
    }
    return static_cast<int>(-std::log(r) * mL_);
}

// ---------------------------------------------------------------------------
// Step 2.2 — insert_node()
//
// Inserts one vector into the graph at every layer from 0 to node_max_layer.
//
// Construction algorithm (per the HNSW paper, Algorithm 1):
//
//   1. Draw node_max_layer from the geometric distribution.
//   2. Allocate the node's jagged neighbor lists (one per layer up to its max).
//   3. For each layer from node_max_layer down to 0:
//        a. Collect candidates: existing nodes that inhabit this layer.
//        b. Cap the pool at ef_construction_, sort by distance.
//        c. Prune to M (or M_max0=2M for layer 0) via select_neighbors.
//        d. Write forward edges (new node → chosen).
//        e. Write reverse edges (chosen → new node), re-pruning any
//           neighbor whose degree now exceeds its layer cap.
//   4. If node_max_layer > max_layer_, promote it to the entry point.
//
// The descent from node_max_layer down to 0 (not 0 up to node_max_layer)
// is deliberate: upper layers have fewer nodes, so their candidate scans
// are cheap. By the time we reach layer 0 we have already established
// coarse connections in the sparse upper layers.
//
// Step 2.3 will replace the O(N) candidate scan with search_layer(),
// giving O(log N) candidate discovery per layer.
// ---------------------------------------------------------------------------
void HnswIndex::insert_node(std::size_t node_idx) {
    const int node_max_layer = draw_layer();

    // Allocate the node with one neighbor list per layer it inhabits.
    // The jagged shape encodes the node's reach: neighbors.size()-1 == its max layer.
    HnswNode node;
    node.index = node_idx;
    node.neighbors.resize(static_cast<std::size_t>(node_max_layer) + 1);
    nodes_.push_back(std::move(node));

    // Bootstrap: first node — nothing to connect, just set the entry point.
    if (nodes_.size() == 1) {
        entry_point_ = node_idx;
        max_layer_   = node_max_layer;
        return;
    }

    const MathVector& query = store_.get(node_idx);

    // Process every layer this node inhabits, top-down.
    for (int layer = node_max_layer; layer >= 0; --layer) {

        // M_max for this layer.
        // Layer 0 gets 2*M connections — a denser base improves recall
        // without affecting upper-layer traversal speed.
        const std::size_t M_max = (layer == 0) ? 2 * M_ : M_;

        // --- Collect candidates: all existing nodes at this layer ---
        // A node exists at layer L iff neighbors.size() > L.
        // We exclude the node just pushed (nodes_.back()) by iterating
        // only nodes_.size() - 1 entries.
        std::vector<SearchResult> candidates;
        for (std::size_t i = 0; i < nodes_.size() - 1; ++i) {
            if (static_cast<int>(nodes_[i].neighbors.size()) > layer) {
                const float dist = euclidean_distance(query, store_.get(nodes_[i].index));
                candidates.push_back({nodes_[i].index, dist});
            }
        }

        // If no nodes exist at this layer yet, skip — happens for the first
        // node that reaches a new upper layer. The entry-point update below
        // will anchor this node as the layer's seed.
        if (candidates.empty()) continue;

        // Cap the pool at ef_construction_ and sort ascending by distance.
        std::sort(candidates.begin(), candidates.end(),
                  [](const SearchResult& a, const SearchResult& b) {
                      return a.distance < b.distance;
                  });
        if (candidates.size() > ef_construction_) {
            candidates.resize(ef_construction_);
        }

        // --- Prune to M_max neighbors ---
        const std::vector<std::size_t> chosen = select_neighbors(candidates, M_max);

        // --- Forward edges: new node → chosen ---
        nodes_.back().neighbors[static_cast<std::size_t>(layer)] = chosen;

        // --- Reverse edges: chosen → new node (bidirectional invariant) ---
        for (const std::size_t neighbor_idx : chosen) {
            std::vector<std::size_t>& rev =
                nodes_[neighbor_idx].neighbors[static_cast<std::size_t>(layer)];
            rev.push_back(node_idx);

            // Re-prune if the reverse edge pushes the neighbor over its cap.
            if (rev.size() > M_max) {
                const MathVector& neighbor_vec = store_.get(neighbor_idx);
                std::vector<SearchResult> re_candidates;
                re_candidates.reserve(rev.size());
                for (const std::size_t nb : rev) {
                    const float d = euclidean_distance(neighbor_vec, store_.get(nb));
                    re_candidates.push_back({nb, d});
                }
                rev = select_neighbors(re_candidates, M_max);
            }
        }
    }

    // --- Promote entry point if this node reaches a higher layer ---
    //
    // The entry point must always be the node with the highest max-layer.
    // Searches start there and descend; starting below the true maximum
    // would skip part of the hierarchy and degrade recall.
    if (node_max_layer > max_layer_) {
        entry_point_ = node_idx;
        max_layer_   = node_max_layer;
    }
}

// ---------------------------------------------------------------------------
// search() — Step 2.1/2.2 placeholder (brute-force)
// Replaced by knn_search() in Step 2.3.
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
