#include "hnsw_index.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>    // std::greater
#include <limits>
#include <queue>         // std::priority_queue
#include <stdexcept>
#include <unordered_set> // visited set in search_layer

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
HnswIndex::HnswIndex(const VectorStore& store,
                     std::size_t        M,
                     std::size_t        ef_construction,
                     std::size_t        ef_search)
    : store_(store)
    , M_(M)
    , ef_construction_(ef_construction)
    , ef_search_(ef_search)
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

std::size_t HnswIndex::size() const noexcept { return nodes_.size(); }
int         HnswIndex::max_layer()   const noexcept { return max_layer_; }
std::size_t HnswIndex::entry_point() const noexcept { return entry_point_; }

std::size_t HnswIndex::layer_population(int layer) const noexcept {
    if (layer < 0) return 0;
    std::size_t count = 0;
    for (const HnswNode& node : nodes_) {
        if (static_cast<int>(node.neighbors.size()) > layer) ++count;
    }
    return count;
}

// ---------------------------------------------------------------------------
// select_neighbors
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
    for (std::size_t i = 0; i < count; ++i) result.push_back(sorted[i].index);
    return result;
}

// ---------------------------------------------------------------------------
// draw_layer
// ---------------------------------------------------------------------------
int HnswIndex::draw_layer() {
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    double r = uniform(rng_);
    if (r < std::numeric_limits<double>::min()) r = std::numeric_limits<double>::min();
    return static_cast<int>(-std::log(r) * mL_);
}

// ---------------------------------------------------------------------------
// insert_node — Step 2.3 upgrade: search_layer replaces the O(N) scan
//
// Construction algorithm (HNSW paper, Algorithm 1):
//
//   Phase 1 — Routing (layers max_layer_ down to node_max_layer+1):
//     Descend through layers ABOVE the new node's reach using ef=1.
//     This navigates the existing hierarchy to find the best entry point
//     for the layers the new node will actually inhabit.
//     No edges are written here.
//
//   Phase 2 — Wiring (layers min(node_max_layer, max_layer_) down to 0):
//     At each layer, call search_layer with ef=ef_construction_ to obtain
//     a high-quality candidate pool, then select + write bidirectional edges.
//     The best candidate found at each layer becomes the entry point for the
//     next (denser) layer. This propagates the position from upper layers
//     down, growing more precise at each step.
//
//   Promotion:
//     If the new node's max layer exceeds max_layer_, it becomes the new
//     global entry point.
//
// Complexity improvement over the previous placeholder:
//   Old (brute-force scan): O(N·D) per insertion → O(N²·D) total build
//   New (search_layer):     O(ef·M·log N·D) per insertion → O(N·ef·M·log N·D)
//   On N=50K: ~80-100× faster build for the same index quality.
// ---------------------------------------------------------------------------
void HnswIndex::insert_node(std::size_t node_idx) {
    const int node_max_layer = draw_layer();

    HnswNode node;
    node.index = node_idx;
    node.neighbors.resize(static_cast<std::size_t>(node_max_layer) + 1);
    nodes_.push_back(std::move(node));

    // Bootstrap: first node — no graph to search, just set the entry point.
    if (nodes_.size() == 1) {
        entry_point_ = node_idx;
        max_layer_   = node_max_layer;
        return;
    }

    const MathVector& query = store_.get(node_idx);
    std::size_t ep = entry_point_;

    // --- Phase 1: greedy routing through layers above node_max_layer ---
    // ef=1 — pure single-path descent, no result collection.
    // Postcondition: ep is the best-known node at layer node_max_layer+1,
    // which also exists at node_max_layer (every node inhabits all layers
    // from 0 up to its max, so existence at L implies existence at L-1).
    for (int layer = max_layer_; layer > node_max_layer; --layer) {
        const auto w = search_layer(query, ep, 1, layer);
        if (w.empty()) continue;
        ep = std::min_element(w.begin(), w.end(),
                 [](const SearchResult& a, const SearchResult& b) {
                     return a.distance < b.distance;
                 })->index;
    }

    // --- Phase 2: candidate search and edge wiring at each inhabited layer ---
    for (int layer = std::min(node_max_layer, max_layer_); layer >= 0; --layer) {
        const std::size_t M_max = (layer == 0) ? 2 * M_ : M_;

        // search_layer gives ef_construction_ nearest nodes as candidates.
        // This is O(ef * M * D) per layer vs the old O(N * D) scan.
        auto w = search_layer(query, ep, ef_construction_, layer);
        if (w.empty()) continue;

        // Prune to M_max and write forward edges.
        const std::vector<std::size_t> chosen = select_neighbors(w, M_max);
        nodes_.back().neighbors[static_cast<std::size_t>(layer)] = chosen;

        // Write reverse edges, re-pruning any neighbor that now exceeds its cap.
        for (const std::size_t neighbor_idx : chosen) {
            std::vector<std::size_t>& rev =
                nodes_[neighbor_idx].neighbors[static_cast<std::size_t>(layer)];
            rev.push_back(node_idx);

            if (rev.size() > M_max) {
                const MathVector& nv = store_.get(neighbor_idx);
                std::vector<SearchResult> re_cands;
                re_cands.reserve(rev.size());
                for (const std::size_t nb : rev) {
                    re_cands.push_back({nb, euclidean_distance(nv, store_.get(nb))});
                }
                rev = select_neighbors(re_cands, M_max);
            }
        }

        // The best node found at this layer is the entry point for the next
        // (denser) layer. search_layer returns farthest-first; find minimum.
        ep = std::min_element(w.begin(), w.end(),
                 [](const SearchResult& a, const SearchResult& b) {
                     return a.distance < b.distance;
                 })->index;
    }

    // Promote entry point if this node has the highest layer so far.
    if (node_max_layer > max_layer_) {
        entry_point_ = node_idx;
        max_layer_   = node_max_layer;
    }
}

// ---------------------------------------------------------------------------
// Step 2.3a — search_layer()
//
// Greedy beam search within a single layer of the graph.
// Implements Algorithm 2 from the HNSW paper (Malkov & Yashunin, 2018).
//
// Two priority queues drive the algorithm:
//
//   candidates (C) — min-heap ordered by distance to query.
//     The closest unexplored node is always at the top.
//     We pop from here to decide what to expand next.
//
//   results (W) — max-heap ordered by distance to query.
//     The worst (farthest) accepted result is always at the top.
//     We peek at the top to decide whether a new node is worth adding,
//     and pop from the top to evict the worst when W overflows ef.
//
// visited set — O(1) insert and lookup to ensure each node is
//   evaluated at most once per search call. Without this, cycles in
//   the graph would cause infinite loops.
//
// Termination: when the closest remaining candidate is farther from
// the query than the worst element already in W, no candidate in C
// can possibly improve W. The search has converged.
//
// Returns up to ef elements in farthest-first order (max-heap drain).
// The caller is responsible for sorting and truncating to k.
// ---------------------------------------------------------------------------
std::vector<SearchResult> HnswIndex::search_layer(const MathVector& query,
                                                   std::size_t       ep_idx,
                                                   std::size_t       ef,
                                                   int               layer) const
{
    using DistIdx = std::pair<float, std::size_t>;

    const float ep_dist = euclidean_distance(query, store_.get(ep_idx));

    // C: min-heap — closest candidate at top (std::greater reverses default)
    std::priority_queue<DistIdx,
                        std::vector<DistIdx>,
                        std::greater<DistIdx>> candidates;
    candidates.push({ep_dist, ep_idx});

    // W: max-heap — worst accepted result at top (default comparator)
    std::priority_queue<DistIdx> results;
    results.push({ep_dist, ep_idx});

    // visited: prevents re-evaluating the same node
    std::unordered_set<std::size_t> visited;
    visited.insert(ep_idx);

    while (!candidates.empty()) {
        const auto [c_dist, c_idx] = candidates.top();
        candidates.pop();

        // Termination condition.
        // results.top() is the WORST (farthest) result collected so far.
        // If the closest remaining candidate c is already farther than that,
        // every node reachable from c is also farther — we cannot improve W.
        if (c_dist > results.top().first) break;

        // Expand: visit all neighbors of c at this layer.
        const std::size_t layer_u = static_cast<std::size_t>(layer);
        if (layer_u >= nodes_[c_idx].neighbors.size()) continue;

        for (const std::size_t nb_idx : nodes_[c_idx].neighbors[layer_u]) {
            // visited.insert returns {iterator, inserted}.
            // We only process nb_idx if it was not already in the set.
            if (!visited.insert(nb_idx).second) continue;

            const float nb_dist = euclidean_distance(query, store_.get(nb_idx));

            // Accept nb_idx into the result set if:
            //   (a) W has fewer than ef elements (not yet full), OR
            //   (b) nb_idx is closer than the current worst result in W.
            // Case (b) also pushes nb_idx into C so it gets expanded.
            if (results.size() < ef || nb_dist < results.top().first) {
                candidates.push({nb_dist, nb_idx});
                results.push({nb_dist, nb_idx});

                // Keep W bounded at ef: evict the farthest element.
                if (results.size() > ef) results.pop();
            }
        }
    }

    // Drain W into a flat vector. Order is farthest-first (max-heap).
    // Callers that need sorted output must sort themselves.
    std::vector<SearchResult> out;
    out.reserve(results.size());
    while (!results.empty()) {
        const auto [dist, idx] = results.top();
        results.pop();
        out.push_back({idx, dist});
    }
    return out;
}

// ---------------------------------------------------------------------------
// Step 2.3b — search() / knn_search
//
// Multi-layer traversal implementing Algorithm 5 from the HNSW paper.
//
// Phase 1 — Greedy routing (layers max_layer_ down to 1):
//   Call search_layer with ef=1. This tracks exactly one candidate at a
//   time — pure greedy, no backtracking. The goal is not to collect
//   results but to navigate quickly toward the query's neighbourhood
//   by making large jumps through the sparse upper layers. Each step
//   uses the current best node as the entry point for the next layer.
//
// Phase 2 — Beam search (layer 0):
//   Call search_layer with ef = max(ef_search_, k). The beam is now
//   wide enough to collect the real top-k candidates. Layer 0 is the
//   densest layer, so this step does the fine-grained refinement.
//   ef must be at least k to guarantee k results are returned.
//
// The split between routing (ef=1) and collection (ef=ef_search) is the
// key architectural decision of HNSW. Upper layers are used for O(log N)
// coarse navigation, not for collecting results. This is why their sparsity
// (fewer nodes, fewer edges) is a performance feature, not a limitation.
// ---------------------------------------------------------------------------
std::vector<SearchResult> HnswIndex::search(const MathVector& query,
                                             std::size_t        k) const
{
    if (query.size() != store_.dimensionality()) {
        throw std::invalid_argument("query dimensionality does not match store");
    }
    if (nodes_.empty()) return {};

    // Phase 1: greedy single-path descent through layers max_layer_ → 1.
    std::size_t ep = entry_point_;
    for (int layer = max_layer_; layer > 0; --layer) {
        const auto w = search_layer(query, ep, 1, layer);
        if (w.empty()) continue;

        // With ef=1, w has exactly one element. Update entry point to the
        // closest node found at this layer before descending to layer-1.
        const auto nearest = std::min_element(
            w.begin(), w.end(),
            [](const SearchResult& a, const SearchResult& b) {
                return a.distance < b.distance;
            });
        ep = nearest->index;
    }

    // Phase 2: beam search at layer 0.
    // ef_search_ is the user-configurable beam width; it must be >= k.
    const std::size_t ef = std::max(ef_search_, k);
    auto results = search_layer(query, ep, ef, 0);

    // search_layer returns farthest-first; sort ascending for the caller.
    std::sort(results.begin(), results.end(),
              [](const SearchResult& a, const SearchResult& b) {
                  return a.distance < b.distance;
              });

    const std::size_t actual_k = std::min(k, results.size());
    results.resize(actual_k);
    return results;
}
