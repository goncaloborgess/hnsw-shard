#include "vector_store.h"
#include "hnsw_index.h"

#include "vectordb.grpc.pb.h"
#include <grpcpp/grpcpp.h>

#include <iostream>
#include <memory>
#include <mutex>
#include <string>

class VectorDBServiceImpl final : public vectordb::VectorDBService::Service {
public:
    VectorDBServiceImpl(std::size_t M, std::size_t ef_construction, std::size_t ef_search)
        : M_(M), ef_construction_(ef_construction), ef_search_(ef_search) {}

    grpc::Status Upsert(grpc::ServerContext* /*context*/,
                        const vectordb::UpsertRequest* request,
                        vectordb::UpsertResponse* response) override {
        if (request->vector().empty())
            return {grpc::INVALID_ARGUMENT, "vector must not be empty"};

        MathVector vec(static_cast<std::size_t>(request->vector().size()));
        for (int i = 0; i < request->vector().size(); ++i)
            vec[static_cast<std::size_t>(i)] = request->vector(i);

        std::lock_guard<std::mutex> lock(mu_);
        try {
            bool created = store_.upsert(request->id(), std::move(vec));
            response->set_created(created);
            dirty_ = true;
        } catch (const std::invalid_argument& e) {
            return {grpc::INVALID_ARGUMENT, e.what()};
        }

        return grpc::Status::OK;
    }

    grpc::Status Search(grpc::ServerContext* /*context*/,
                        const vectordb::SearchRequest* request,
                        vectordb::SearchResponse* response) override {
        if (request->query().empty())
            return {grpc::INVALID_ARGUMENT, "query must not be empty"};
        if (request->k() == 0)
            return {grpc::INVALID_ARGUMENT, "k must be > 0"};

        MathVector query(static_cast<std::size_t>(request->query().size()));
        for (int i = 0; i < request->query().size(); ++i)
            query[static_cast<std::size_t>(i)] = request->query(i);

        std::lock_guard<std::mutex> lock(mu_);

        if (store_.empty())
            return grpc::Status::OK;

        if (dirty_) {
            std::cout << "Rebuilding HNSW index (" << store_.size()
                      << " vectors)..." << std::flush;
            index_ = std::make_unique<HnswIndex>(store_, M_, ef_construction_, ef_search_);
            index_->build();
            dirty_ = false;
            std::cout << " done (max_layer=" << index_->max_layer() << ")\n";
        }

        try {
            auto results = index_->search(query, request->k());
            for (const auto& r : results) {
                auto* hit = response->add_results();
                hit->set_id(store_.get_id(r.index));
                hit->set_distance(r.distance);
            }
        } catch (const std::invalid_argument& e) {
            return {grpc::INVALID_ARGUMENT, e.what()};
        }

        return grpc::Status::OK;
    }

private:
    VectorStore                    store_;
    std::unique_ptr<HnswIndex>    index_;
    std::mutex                     mu_;
    bool                           dirty_ = false;

    std::size_t M_;
    std::size_t ef_construction_;
    std::size_t ef_search_;
};

int main(int argc, char* argv[]) {
    const std::string port = (argc > 1) ? argv[1] : "50051";
    const std::string address = "0.0.0.0:" + port;

    VectorDBServiceImpl service(/*M=*/16, /*ef_construction=*/200, /*ef_search=*/50);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    auto server = builder.BuildAndStart();
    std::cout << "VectorDB server listening on " << address << "\n";
    server->Wait();

    return 0;
}
