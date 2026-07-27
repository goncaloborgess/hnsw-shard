#include "math_vector.h"
#include "vector_store.h"
#include "brute_force_index.h"
#include "hnsw_index.h"

#include <algorithm>  // std::min_element, std::max_element
#include <chrono>     // high_resolution_clock
#include <cstdint>    // int32_t
#include <fstream>    // std::ifstream (fvecs loader)
#include <iomanip>    // std::setprecision
#include <iostream>
#include <numeric>    // std::accumulate
#include <random>     // std::mt19937, std::uniform_real_distribution
#include <string>
#include <unordered_set>
#include <vector>

// Generates a single random D-dimensional float vector.
// Values drawn from uniform [-1.0, 1.0] — a common distribution for
// synthetic ANN benchmarks. The rng is passed by reference so the caller's
// state advances and every call produces a different vector.
static MathVector generate_random_vector(std::size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    MathVector v(dim);
    for (std::size_t i = 0; i < dim; ++i)
        v[i] = dist(rng);
    return v;
}

// ---------------------------------------------------------------------------
// fvecs loader helpers
//
// fvecs binary format (one record per vector):
//   [int32_t dim][float32 × dim]
//
// load_fvecs_store  — reads up to max_vectors records into a VectorStore.
// load_fvecs_queries — reads up to max_vectors records into a plain vector.
//
// Both return the number of vectors actually loaded.
// An empty return means the file could not be opened.
// ---------------------------------------------------------------------------
static std::size_t load_fvecs_store(const std::string& path,
                                    VectorStore&       store,
                                    std::size_t        max_vectors)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return 0;

    std::size_t count = 0;
    while (count < max_vectors) {
        int32_t dim = 0;
        if (!f.read(reinterpret_cast<char*>(&dim), sizeof(dim))) break;

        MathVector v(static_cast<std::size_t>(dim));
        f.read(reinterpret_cast<char*>(v.data()),
               static_cast<std::streamsize>(dim) * sizeof(float));
        if (!f) break;

        store.insert("vec-" + std::to_string(count), std::move(v));
        ++count;
    }
    return count;
}

static std::size_t load_fvecs_queries(const std::string&         path,
                                      std::vector<MathVector>&   queries,
                                      std::size_t                max_vectors)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return 0;

    std::size_t count = 0;
    while (count < max_vectors) {
        int32_t dim = 0;
        if (!f.read(reinterpret_cast<char*>(&dim), sizeof(dim))) break;

        MathVector v(static_cast<std::size_t>(dim));
        f.read(reinterpret_cast<char*>(v.data()),
               static_cast<std::streamsize>(dim) * sizeof(float));
        if (!f) break;

        queries.push_back(std::move(v));
        ++count;
    }
    return count;
}

static void run_benchmark() {
    constexpr std::size_t N       = 50'000; // vectors in the dataset
    constexpr std::size_t D       = 128;    // dimensions per vector (synthetic fallback)
    constexpr std::size_t K       = 10;     // neighbours to retrieve
    constexpr std::size_t Q       = 100;    // timed queries
    constexpr std::size_t M       = 16;     // HNSW max neighbours per layer
    constexpr std::size_t EF_CON  = 200;    // HNSW build beam width
    constexpr std::size_t EF_SRCH = 50;     // HNSW query beam width

    using clock = std::chrono::high_resolution_clock;
    using ms    = std::chrono::duration<double, std::milli>;
    using sec   = std::chrono::duration<double>;

    std::mt19937 rng(42);

    // -----------------------------------------------------------------------
    // [1/4] Load (or generate) the dataset
    // -----------------------------------------------------------------------
    VectorStore store;
    std::string dataset_label;

    std::cout << "  [1/4] loading base vectors... " << std::flush;
    const std::size_t loaded_n = load_fvecs_store("data/sift_base.fvecs", store, N);
    if (loaded_n > 0) {
        dataset_label = "SIFT-1M (first " + std::to_string(loaded_n) + ")";
        std::cout << "loaded " << loaded_n << " SIFT vectors  (128-D)\n";
    } else {
        std::cout << "data/sift_base.fvecs not found — generating synthetic data... " << std::flush;
        for (std::size_t i = 0; i < N; ++i)
            store.insert("vec-" + std::to_string(i), generate_random_vector(D, rng));
        dataset_label = "synthetic uniform[-1,1] (" + std::to_string(N) + " x " + std::to_string(D) + ")";
        std::cout << "done\n";
    }

    // -----------------------------------------------------------------------
    // Print benchmark header (dataset source is now known)
    // -----------------------------------------------------------------------
    std::cout << "\n=== Benchmark: Brute-Force vs HNSW ===\n";
    std::cout << "  dataset  : " << dataset_label << "\n";
    std::cout << "  k        : " << K << "\n";
    std::cout << "  queries  : " << Q << "\n";
    std::cout << "  M        : " << M << "  |  ef_construction: " << EF_CON
              << "  |  ef_search: " << EF_SRCH << "\n\n";

    // -----------------------------------------------------------------------
    // Build HNSW index (timed)
    // The graph build uses search_layer for O(ef*M*log N) insertion cost.
    // -----------------------------------------------------------------------
    std::cout << "  [2/4] building HNSW index... " << std::flush;
    HnswIndex hnsw(store, M, EF_CON, EF_SRCH);
    const auto build_t0 = clock::now();
    hnsw.build();
    const auto build_t1 = clock::now();
    const double build_s = sec(build_t1 - build_t0).count();
    std::cout << "done  (" << std::fixed << std::setprecision(2)
              << build_s << " s,  max_layer=" << hnsw.max_layer() << ")\n";

    // -----------------------------------------------------------------------
    // [3/4] Load (or generate) query vectors outside any timed section
    // -----------------------------------------------------------------------
    std::cout << "  [3/4] loading query vectors... " << std::flush;
    std::vector<MathVector> queries;
    queries.reserve(Q);
    const std::size_t loaded_q = load_fvecs_queries("data/sift_query.fvecs", queries, Q);
    if (loaded_q > 0) {
        std::cout << "loaded " << loaded_q << " SIFT queries\n\n";
    } else {
        const std::size_t qdim = store.dimensionality();
        for (std::size_t i = 0; i < Q; ++i)
            queries.push_back(generate_random_vector(qdim, rng));
        std::cout << "done (synthetic)\n\n";
    }

    // -----------------------------------------------------------------------
    // Time brute-force search and store results for recall computation
    // -----------------------------------------------------------------------
    std::cout << "  [4/4] running timed queries...\n\n";

    BruteForceIndex bf(store);
    std::vector<std::vector<SearchResult>> bf_results(Q);
    std::vector<double> bf_lat;
    bf_lat.reserve(Q);

    for (std::size_t i = 0; i < Q; ++i) {
        const auto t0 = clock::now();
        bf_results[i]  = bf.search(queries[i], K);
        const auto t1 = clock::now();
        bf_lat.push_back(ms(t1 - t0).count());
    }

    // -----------------------------------------------------------------------
    // Time HNSW search and compute recall@K against brute-force ground truth
    // -----------------------------------------------------------------------
    std::vector<double> hnsw_lat;
    hnsw_lat.reserve(Q);
    double total_recall = 0.0;

    for (std::size_t i = 0; i < Q; ++i) {
        const auto t0      = clock::now();
        const auto hnsw_r  = hnsw.search(queries[i], K);
        const auto t1      = clock::now();
        hnsw_lat.push_back(ms(t1 - t0).count());

        // recall@K = |HNSW_topK ∩ BF_topK| / K
        std::unordered_set<std::size_t> gt;
        for (const auto& r : bf_results[i]) gt.insert(r.index);
        std::size_t hits = 0;
        for (const auto& r : hnsw_r) { if (gt.count(r.index)) ++hits; }
        total_recall += static_cast<double>(hits) / static_cast<double>(K);
        (void)hnsw_r;
    }

    // -----------------------------------------------------------------------
    // Report
    // -----------------------------------------------------------------------
    const double bf_avg   = std::accumulate(bf_lat.begin(),   bf_lat.end(),   0.0) / Q;
    const double bf_min   = *std::min_element(bf_lat.begin(), bf_lat.end());
    const double bf_max   = *std::max_element(bf_lat.begin(), bf_lat.end());

    const double hnsw_avg = std::accumulate(hnsw_lat.begin(), hnsw_lat.end(), 0.0) / Q;
    const double hnsw_min = *std::min_element(hnsw_lat.begin(), hnsw_lat.end());
    const double hnsw_max = *std::max_element(hnsw_lat.begin(), hnsw_lat.end());

    const double speedup  = bf_avg / hnsw_avg;
    const double recall   = total_recall / static_cast<double>(Q);

    std::cout << std::fixed << std::setprecision(3);

    // Latency table
    std::cout << "  +-----------------+-----------+-----------+-----------+\n";
    std::cout << "  |                 |  avg (ms) |  min (ms) |  max (ms) |\n";
    std::cout << "  +-----------------+-----------+-----------+-----------+\n";
    std::cout << "  | brute-force     | " << std::setw(9) << bf_avg
              << " | " << std::setw(9) << bf_min
              << " | " << std::setw(9) << bf_max << " |\n";
    std::cout << "  | HNSW            | " << std::setw(9) << hnsw_avg
              << " | " << std::setw(9) << hnsw_min
              << " | " << std::setw(9) << hnsw_max << " |\n";
    std::cout << "  +-----------------+-----------+-----------+-----------+\n\n";

    std::cout << "  speedup  (BF avg / HNSW avg) : " << std::setprecision(1)
              << speedup << "x\n";
    std::cout << "  recall@" << K << "                   : " << std::setprecision(3)
              << recall << "\n";
    std::cout << "  HNSW build time              : " << std::setprecision(2)
              << build_s << " s\n";
}

int main() {
    run_benchmark();
    return 0;
}
