#pragma once

#include "math_vector.h"
#include "vector_store.h"

#include <vector>

class BruteForceIndex {
public:
    // Takes a reference to an existing VectorStore.
    // The index does not own the store — it is a read-only view over it.
    // The store must outlive the index.
    explicit BruteForceIndex(const VectorStore& store);

    // Linear scan over every vector in the store.
    // Returns the k nearest neighbours to 'query', sorted ascending by distance.
    // If k > store.size(), returns all vectors (clamped).
    // Precondition: query.size() == store.dimensionality()
    std::vector<SearchResult> search(const MathVector& query, std::size_t k) const;

private:
    const VectorStore& store_;
};
