#include "brute_force_index.h"

#include <algorithm>  // std::partial_sort
#include <stdexcept>  // std::invalid_argument

BruteForceIndex::BruteForceIndex(const VectorStore& store)
    : store_(store)
{}

std::vector<SearchResult> BruteForceIndex::search(const MathVector& query,
                                                   std::size_t k) const {
    if (store_.empty()) return {};

    if (query.size() != store_.dimensionality())
        throw std::invalid_argument(
            "BruteForceIndex::search: query dimension mismatch");

    const std::size_t n       = store_.size();
    const std::size_t actual_k = (k < n) ? k : n;  // clamp k to n

    // --- Step 1: compute every distance ---
    // Access store_.vectors() once and hold the reference for the whole loop.
    // This avoids repeated virtual dispatch or indirection on every iteration.
    const auto& vecs = store_.vectors();

    std::vector<SearchResult> candidates;
    candidates.reserve(n);  // pre-allocate to avoid reallocation mid-loop

    for (std::size_t i = 0; i < n; ++i)
        candidates.push_back({i, euclidean_distance(query, vecs[i])});

    // --- Step 2: partially sort to bring the k smallest to the front ---
    // std::partial_sort guarantees [begin, begin+k) are the k smallest,
    // sorted ascending. Elements after [begin+k) are in unspecified order.
    // Complexity: O(N log k) — significantly better than O(N log N) full sort
    // when k << N, which is always the case in practice (k=10 vs N=50000).
    std::partial_sort(
        candidates.begin(),
        candidates.begin() + static_cast<std::ptrdiff_t>(actual_k),
        candidates.end(),
        [](const SearchResult& a, const SearchResult& b) {
            return a.distance < b.distance;
        });

    candidates.resize(actual_k);
    return candidates;
}
