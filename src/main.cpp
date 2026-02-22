#include "math_vector.h"
#include "vector_store.h"

#include <cassert>    // assert()
#include <cmath>      // std::abs
#include <iostream>
#include <stdexcept>  // std::invalid_argument

// Float comparison: exact equality is unreliable for floating-point results.
// We test that the result is within a small epsilon of the expected value.
static constexpr float EPSILON = 1e-5f;

static bool near(float a, float b) {
    return std::abs(a - b) < EPSILON;
}

// =============================================================================
// Test sections
// =============================================================================

static void test_constructors() {
    std::cout << "[constructors]\n";

    // Default: empty, valid object
    MathVector v_default;
    assert(v_default.size() == 0);
    assert(v_default.empty());
    assert(v_default.data() == nullptr);
    std::cout << "  default constructor      OK\n";

    // Size: allocates N zero-initialised elements
    MathVector v_size(4);
    assert(v_size.size() == 4);
    assert(!v_size.empty());
    for (std::size_t i = 0; i < v_size.size(); ++i)
        assert(v_size[i] == 0.0f);
    std::cout << "  size constructor         OK\n";

    // Initializer-list: elements match the literal
    MathVector v_init{1.0f, 2.0f, 3.0f};
    assert(v_init.size() == 3);
    assert(near(v_init[0], 1.0f));
    assert(near(v_init[1], 2.0f));
    assert(near(v_init[2], 3.0f));
    std::cout << "  initializer_list ctor    OK\n";
}

static void test_rule_of_five() {
    std::cout << "[rule of five]\n";

    MathVector original{1.0f, 2.0f, 3.0f};

    // Copy constructor: produces an independent deep copy
    MathVector copy(original);
    assert(copy.size() == original.size());
    assert(near(copy[0], 1.0f));
    original[0] = 99.0f;          // mutate the original
    assert(near(copy[0], 1.0f));  // copy must be unaffected
    original[0] = 1.0f;           // restore
    std::cout << "  copy constructor         OK\n";

    // Move constructor: transfers ownership, source becomes empty
    MathVector donor{4.0f, 5.0f, 6.0f};
    MathVector moved(std::move(donor));
    assert(moved.size() == 3);
    assert(near(moved[0], 4.0f));
    assert(donor.size() == 0);     // source must be emptied
    assert(donor.data() == nullptr);
    std::cout << "  move constructor         OK\n";

    // Copy assignment: replaces target, source unchanged, objects are independent
    MathVector target{9.0f, 9.0f};
    target = original;
    assert(target.size() == original.size());
    assert(near(target[1], 2.0f));
    original[1] = 77.0f;
    assert(near(target[1], 2.0f));  // target must be independent
    original[1] = 2.0f;
    std::cout << "  copy assignment          OK\n";

    // Move assignment: target receives donor's data, donor becomes empty
    MathVector move_target{0.0f};
    MathVector move_donor{7.0f, 8.0f, 9.0f};
    move_target = std::move(move_donor);
    assert(move_target.size() == 3);
    assert(near(move_target[2], 9.0f));
    assert(move_donor.size() == 0);
    std::cout << "  move assignment          OK\n";

    // Self-assignment: must not corrupt the object
    MathVector self{1.0f, 2.0f};
    self = self;
    assert(near(self[0], 1.0f));
    assert(near(self[1], 2.0f));
    std::cout << "  self-assignment guard    OK\n";
}

static void test_element_access() {
    std::cout << "[element access]\n";

    MathVector v{10.0f, 20.0f, 30.0f};

    // operator[]: read
    assert(near(v[2], 30.0f));
    std::cout << "  operator[] read          OK\n";

    // operator[]: write through reference
    v[1] = 99.0f;
    assert(near(v[1], 99.0f));
    std::cout << "  operator[] write         OK\n";

    // at(): valid index
    assert(near(v.at(0), 10.0f));
    std::cout << "  at() valid index         OK\n";

    // at(): out-of-range throws std::out_of_range
    bool threw = false;
    try { v.at(100); }
    catch (const std::out_of_range&) { threw = true; }
    assert(threw);
    std::cout << "  at() out-of-range throw  OK\n";

    // const access
    const MathVector cv{5.0f, 6.0f};
    assert(near(cv[0], 5.0f));
    assert(near(cv.at(1), 6.0f));
    std::cout << "  const element access     OK\n";

    // data(): pointer points to first element
    assert(v.data() == &v[0]);
    std::cout << "  data() pointer           OK\n";
}

static void test_iterators() {
    std::cout << "[iterators]\n";

    MathVector v{1.0f, 2.0f, 3.0f, 4.0f};

    // Range-for loop
    float expected = 1.0f;
    for (float val : v) {
        assert(near(val, expected));
        expected += 1.0f;
    }
    std::cout << "  range-for loop           OK\n";

    // begin()/end() pointer arithmetic
    assert(v.end() - v.begin() == static_cast<std::ptrdiff_t>(v.size()));
    std::cout << "  end() - begin() == size  OK\n";

    // Const iterators
    const MathVector cv{5.0f, 6.0f};
    float sum = 0.0f;
    for (const float val : cv) sum += val;
    assert(near(sum, 11.0f));
    std::cout << "  const iterators          OK\n";
}

static void test_euclidean_distance() {
    std::cout << "[euclidean_distance]\n";

    // Pythagorean triple 3-4-5:
    // a = (0, 0), b = (3, 4) → distance = sqrt(9 + 16) = 5.0
    MathVector a{0.0f, 0.0f};
    MathVector b{3.0f, 4.0f};
    assert(near(euclidean_distance(a, b), 5.0f));
    std::cout << "  3-4-5 right triangle     OK\n";

    // Distance from a vector to itself must be zero
    MathVector v{1.0f, 2.0f, 3.0f};
    assert(near(euclidean_distance(v, v), 0.0f));
    std::cout << "  self-distance == 0       OK\n";

    // Distance is symmetric: d(a,b) == d(b,a)
    MathVector x{1.0f, 0.0f, 0.0f};
    MathVector y{0.0f, 1.0f, 0.0f};
    assert(near(euclidean_distance(x, y), euclidean_distance(y, x)));
    std::cout << "  symmetry d(a,b)==d(b,a)  OK\n";
}

static void test_vector_store() {
    std::cout << "[vector_store]\n";

    VectorStore store;
    assert(store.empty());
    assert(store.size() == 0);
    assert(store.dimensionality() == 0);
    std::cout << "  empty store              OK\n";

    // Insert three vectors with string IDs
    store.insert("doc-0", MathVector{1.0f, 2.0f, 3.0f});
    store.insert("doc-1", MathVector{4.0f, 5.0f, 6.0f});
    store.insert("doc-2", MathVector{7.0f, 8.0f, 9.0f});

    assert(store.size() == 3);
    assert(!store.empty());
    assert(store.dimensionality() == 3);
    std::cout << "  insert three vectors     OK\n";

    // get(string) returns the correct vector
    assert(near(store.get("doc-1")[0], 4.0f));
    assert(near(store.get("doc-2")[2], 9.0f));
    std::cout << "  get(string)              OK\n";

    // get(index) returns the correct vector
    assert(near(store.get(0)[0], 1.0f));
    assert(near(store.get(2)[1], 8.0f));
    std::cout << "  get(index)               OK\n";

    // get_id(index) translates back to the original string
    assert(store.get_id(0) == "doc-0");
    assert(store.get_id(1) == "doc-1");
    assert(store.get_id(2) == "doc-2");
    std::cout << "  get_id(index)            OK\n";

    // contains() is true for inserted IDs, false for unknowns
    assert(store.contains("doc-0"));
    assert(!store.contains("doc-99"));
    std::cout << "  contains()               OK\n";

    // vectors() exposes the contiguous store — pointer of first element matches
    assert(store.vectors().size() == 3);
    assert(store.vectors().data() == &store.get(0));
    std::cout << "  vectors() contiguous     OK\n";

    // Duplicate ID must throw std::invalid_argument
    bool threw = false;
    try { store.insert("doc-0", MathVector{0.0f, 0.0f, 0.0f}); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
    std::cout << "  duplicate id throws      OK\n";

    // Dimension mismatch must throw std::invalid_argument
    threw = false;
    try { store.insert("doc-bad", MathVector{1.0f, 2.0f}); }  // 2D into 3D store
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
    std::cout << "  dimension mismatch throws OK\n";
}

// =============================================================================
// Entry point
// =============================================================================

int main() {
    std::cout << "=== VectorDB test driver ===\n\n";

    test_constructors();
    std::cout << '\n';
    test_rule_of_five();
    std::cout << '\n';
    test_element_access();
    std::cout << '\n';
    test_iterators();
    std::cout << '\n';
    test_euclidean_distance();
    std::cout << '\n';
    test_vector_store();

    std::cout << "\nAll tests passed.\n";
    return 0;
}
