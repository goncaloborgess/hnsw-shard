#pragma once

#include "math_vector.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

class VectorStore {
public:
    VectorStore() = default;

    // Inserts a vector with a user-facing string ID.
    // Returns the internal index assigned to this vector.
    // Throws std::invalid_argument if:
    //   - 'id' already exists in the store
    //   - 'vec' dimensionality does not match previously inserted vectors
    std::size_t insert(const std::string& id, MathVector vec);

    // Retrieve a vector by string ID.
    // Throws std::out_of_range if id is not found.
    const MathVector& get(const std::string& id) const;

    // Retrieve a vector by internal index.
    // Throws std::out_of_range if index is out of bounds.
    const MathVector& get(std::size_t index) const;

    // Translate an internal index back to its string ID.
    // Throws std::out_of_range if index is out of bounds.
    const std::string& get_id(std::size_t index) const;

    // Returns true if the string ID exists in the store.
    bool contains(const std::string& id) const;

    // Number of stored vectors.
    std::size_t size() const noexcept;

    // True if no vectors have been inserted.
    bool empty() const noexcept;

    // Dimensionality of all stored vectors (0 if empty).
    std::size_t dimensionality() const noexcept;

    // Direct read-only access to the contiguous vector storage.
    // Used by scanners (brute-force k-NN, HNSW) for cache-friendly iteration.
    const std::vector<MathVector>& vectors() const noexcept;

private:
    std::vector<MathVector>                      vectors_;       // contiguous storage
    std::unordered_map<std::string, std::size_t> id_to_index_;   // string → internal index
    std::vector<std::string>                     index_to_id_;   // internal index → string
    std::size_t                                  dimensionality_ = 0;
};
