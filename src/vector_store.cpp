#include "vector_store.h"

#include <stdexcept>  // std::invalid_argument, std::out_of_range

std::size_t VectorStore::insert(const std::string& id, MathVector vec) {
    // Reject duplicate string IDs — the store is append-only.
    if (id_to_index_.count(id) > 0)
        throw std::invalid_argument("VectorStore::insert: duplicate id '" + id + "'");

    // On the first insert, record the dimensionality for all future inserts.
    // Every subsequent vector must match — mixed dimensions would make distance
    // calculations meaningless and are always a caller error.
    if (vectors_.empty()) {
        dimensionality_ = vec.size();
    } else if (vec.size() != dimensionality_) {
        throw std::invalid_argument(
            "VectorStore::insert: dimension mismatch — expected " +
            std::to_string(dimensionality_) + ", got " +
            std::to_string(vec.size()));
    }

    const std::size_t index = vectors_.size();

    // All three containers are kept in sync: same index, same insertion order.
    // std::move avoids a deep copy of the MathVector's heap allocation.
    id_to_index_[id] = index;
    index_to_id_.push_back(id);
    vectors_.push_back(std::move(vec));

    return index;
}

bool VectorStore::upsert(const std::string& id, MathVector vec) {
    auto it = id_to_index_.find(id);
    if (it != id_to_index_.end()) {
        if (vec.size() != dimensionality_)
            throw std::invalid_argument(
                "VectorStore::upsert: dimension mismatch — expected " +
                std::to_string(dimensionality_) + ", got " +
                std::to_string(vec.size()));
        vectors_[it->second] = std::move(vec);
        return false;
    }
    insert(id, std::move(vec));
    return true;
}

const MathVector& VectorStore::get(const std::string& id) const {
    // unordered_map::at throws std::out_of_range on missing key — correct behaviour.
    return vectors_[id_to_index_.at(id)];
}

const MathVector& VectorStore::get(std::size_t index) const {
    // std::vector::at throws std::out_of_range on out-of-bounds access.
    return vectors_.at(index);
}

const std::string& VectorStore::get_id(std::size_t index) const {
    return index_to_id_.at(index);
}

bool VectorStore::contains(const std::string& id) const {
    return id_to_index_.count(id) > 0;
}

std::size_t VectorStore::size() const noexcept {
    return vectors_.size();
}

bool VectorStore::empty() const noexcept {
    return vectors_.empty();
}

std::size_t VectorStore::dimensionality() const noexcept {
    return dimensionality_;
}

const std::vector<MathVector>& VectorStore::vectors() const noexcept {
    return vectors_;
}
