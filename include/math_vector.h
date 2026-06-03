#pragma once

#include <cstddef>
#include <initializer_list>

class MathVector {
public:
    // --- Constructors & Destructor ---
    MathVector();
    explicit MathVector(std::size_t size);
    MathVector(std::initializer_list<float> values);

    // Rule of Five
    MathVector(const MathVector& other);
    MathVector(MathVector&& other) noexcept;
    MathVector& operator=(const MathVector& other);
    MathVector& operator=(MathVector&& other) noexcept;
    ~MathVector();

    // --- Element Access ---
    float& operator[](std::size_t index);
    const float& operator[](std::size_t index) const;
    float& at(std::size_t index);
    const float& at(std::size_t index) const;
    float* data() noexcept;
    const float* data() const noexcept;

    // --- Capacity ---
    std::size_t size() const noexcept;
    bool empty() const noexcept;

    // --- Iterators ---
    float* begin() noexcept;
    float* end() noexcept;
    const float* begin() const noexcept;
    const float* end() const noexcept;

private:
    float*      data_;
    std::size_t size_;
};

// --- Search Result ---
// Returned by any index search. index is the internal position in VectorStore;
// the caller translates it to a string ID via VectorStore::get_id().
struct SearchResult {
    std::size_t index;
    float       distance;
};

// --- Free Functions ---
float euclidean_distance(const MathVector& a, const MathVector& b);
