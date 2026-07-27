#include "math_vector.h"

#include <algorithm>   // std::copy
#include <cmath>        // std::sqrt
#include <stdexcept>    // std::out_of_range

// =============================================================================
// Constructors
// =============================================================================

// Default constructor: empty, valid object. No heap allocation.
MathVector::MathVector()
    : data_(nullptr)
    , size_(0)
{}

// Size constructor: allocates and zero-initialises 'size' floats.
// new float[size]() — the () triggers value-initialisation (sets all to 0.0f).
MathVector::MathVector(std::size_t size)
    : data_(size > 0 ? new float[size]() : nullptr)
    , size_(size)
{}

// Initializer-list constructor: MathVector v{1.0f, 2.0f, 3.0f}
// std::copy writes from the compiler-generated temporary list into our allocation.
MathVector::MathVector(std::initializer_list<float> values)
    : data_(values.size() > 0 ? new float[values.size()] : nullptr)
    , size_(values.size())
{
    std::copy(values.begin(), values.end(), data_);
}

// =============================================================================
// Rule of Five
// =============================================================================

// Destructor: releases the heap allocation.
// delete[] on nullptr is a no-op — safe to call on default-constructed objects.
MathVector::~MathVector() {
    delete[] data_;
}

// Copy constructor: deep copy — allocates new memory and duplicates all elements.
// The source object is unchanged. Both objects own independent arrays.
MathVector::MathVector(const MathVector& other)
    : data_(other.size_ > 0 ? new float[other.size_] : nullptr)
    , size_(other.size_)
{
    std::copy(other.data_, other.data_ + other.size_, data_);
}

// Move constructor: steals the resource from a temporary (rvalue).
// No allocation, no copy — just pointer transfer.
// The source is left in a valid empty state so its destructor is safe to call.
MathVector::MathVector(MathVector&& other) noexcept
    : data_(other.data_)
    , size_(other.size_)
{
    other.data_ = nullptr;
    other.size_ = 0;
}

// Copy assignment: replaces this object's data with a deep copy of 'other'.
// Allocates new memory first, then releases the old — exception-safe order.
// Self-assignment guard prevents deleting our own data before copying it.
MathVector& MathVector::operator=(const MathVector& other) {
    if (this == &other) return *this;

    float* new_data = other.size_ > 0 ? new float[other.size_] : nullptr;
    std::copy(other.data_, other.data_ + other.size_, new_data);

    delete[] data_;
    data_ = new_data;
    size_ = other.size_;
    return *this;
}

// Move assignment: releases our current resource, then steals from 'other'.
// Self-assignment guard prevents a use-after-free on the same object.
MathVector& MathVector::operator=(MathVector&& other) noexcept {
    if (this == &other) return *this;

    delete[] data_;
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
    return *this;
}

// =============================================================================
// Element Access
// =============================================================================

// operator[]: unchecked access, zero overhead. Returns a reference so the
// caller can both read and write: v[0] = 9.0f; float x = v[0];
float& MathVector::operator[](std::size_t index) {
    return data_[index];
}

const float& MathVector::operator[](std::size_t index) const {
    return data_[index];
}

// at(): bounds-checked access. Throws std::out_of_range on bad index.
// Use this at system boundaries (user input, network data), not in hot loops.
float& MathVector::at(std::size_t index) {
    if (index >= size_)
        throw std::out_of_range("MathVector::at: index out of range");
    return data_[index];
}

const float& MathVector::at(std::size_t index) const {
    if (index >= size_)
        throw std::out_of_range("MathVector::at: index out of range");
    return data_[index];
}

// data(): raw pointer access for SIMD intrinsics, BLAS, and low-level operations.
float* MathVector::data() noexcept {
    return data_;
}

const float* MathVector::data() const noexcept {
    return data_;
}

// =============================================================================
// Capacity
// =============================================================================

std::size_t MathVector::size() const noexcept {
    return size_;
}

bool MathVector::empty() const noexcept {
    return size_ == 0;
}

// =============================================================================
// Iterators
// =============================================================================

// begin()/end() return raw pointers — the simplest valid iterator for a
// contiguous array. This makes MathVector compatible with range-for loops
// and every algorithm in <algorithm>.
float* MathVector::begin() noexcept { return data_; }
float* MathVector::end()   noexcept { return data_ + size_; }

const float* MathVector::begin() const noexcept { return data_; }
const float* MathVector::end()   const noexcept { return data_ + size_; }

// =============================================================================
// Free Functions
// =============================================================================

// Euclidean distance: sqrt( sum( (a_i - b_i)^2 ) )
// Uses operator[] (unchecked) — this is a hot path, no bounds overhead.
// Precondition: a.size() == b.size(). Enforced by the caller / upper layers.
// diff * diff avoids std::pow, which is significantly slower for squares.
float euclidean_distance(const MathVector& a, const MathVector& b) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}
