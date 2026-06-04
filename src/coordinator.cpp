#include "partitioner.h"

#include "vectordb.grpc.pb.h"
#include <grpcpp/grpcpp.h>

#include <algorithm>
#include <future>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class CoordinatorServiceImpl final : public vectordb::VectorDBService::Service {
public:
    explicit CoordinatorServiceImpl(const std::vector<std::string>& shard_addresses) {
        for (const auto& addr : shard_addresses) {
            auto channel = grpc::CreateChannel(addr, grpc::InsecureChannelCredentials());
            stubs_.push_back(vectordb::VectorDBService::NewStub(channel));
        }
        num_shards_ = stubs_.size();
    }

    grpc::Status Upsert(grpc::ServerContext* /*context*/,
                        const vectordb::UpsertRequest* request,
                        vectordb::UpsertResponse* response) override {
        if (request->vector().empty())
            return {grpc::INVALID_ARGUMENT, "vector must not be empty"};

        const std::size_t shard = assign_shard(request->id(), num_shards_);

        grpc::ClientContext ctx;
        grpc::Status status = stubs_[shard]->Upsert(&ctx, *request, response);
        if (!status.ok())
            return {grpc::UNAVAILABLE,
                    "shard " + std::to_string(shard) + ": " + status.error_message()};

        std::cout << "Upsert '" << request->id() << "' -> shard " << shard << "\n";
        return grpc::Status::OK;
    }

    grpc::Status Search(grpc::ServerContext* /*context*/,
                        const vectordb::SearchRequest* request,
                        vectordb::SearchResponse* response) override {
        if (request->query().empty())
            return {grpc::INVALID_ARGUMENT, "query must not be empty"};
        if (request->k() == 0)
            return {grpc::INVALID_ARGUMENT, "k must be > 0"};

        // Fan out to all shards concurrently
        using FanoutResult = std::pair<grpc::Status, vectordb::SearchResponse>;
        std::vector<std::future<FanoutResult>> futures;
        futures.reserve(num_shards_);

        for (std::size_t i = 0; i < num_shards_; ++i) {
            futures.push_back(std::async(std::launch::async,
                [this, i, request]() -> FanoutResult {
                    grpc::ClientContext ctx;
                    vectordb::SearchRequest req;
                    *req.mutable_query() = request->query();
                    req.set_k(request->k());

                    vectordb::SearchResponse resp;
                    auto status = stubs_[i]->Search(&ctx, req, &resp);
                    return {status, std::move(resp)};
                }
            ));
        }

        // Gather partial results
        struct MergeEntry { std::string id; float distance; };
        std::vector<MergeEntry> all_results;

        for (std::size_t i = 0; i < futures.size(); ++i) {
            auto [status, resp] = futures[i].get();
            if (!status.ok())
                return {grpc::UNAVAILABLE,
                        "shard " + std::to_string(i) + ": " + status.error_message()};

            for (const auto& r : resp.results())
                all_results.push_back({r.id(), r.distance()});
        }

        // Merge: sort by distance, truncate to k
        std::sort(all_results.begin(), all_results.end(),
                  [](const MergeEntry& a, const MergeEntry& b) {
                      return a.distance < b.distance;
                  });

        const std::size_t k = std::min(
            static_cast<std::size_t>(request->k()), all_results.size());

        for (std::size_t i = 0; i < k; ++i) {
            auto* hit = response->add_results();
            hit->set_id(all_results[i].id);
            hit->set_distance(all_results[i].distance);
        }

        return grpc::Status::OK;
    }

private:
    std::vector<std::unique_ptr<vectordb::VectorDBService::Stub>> stubs_;
    std::size_t num_shards_;
};

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: vectordb_coordinator <port> <shard1:port> [shard2:port ...]\n";
        return 1;
    }

    const std::string port = argv[1];
    std::vector<std::string> shard_addresses;
    for (int i = 2; i < argc; ++i)
        shard_addresses.emplace_back(argv[i]);

    const std::string address = "0.0.0.0:" + port;
    CoordinatorServiceImpl service(shard_addresses);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    auto server = builder.BuildAndStart();
    std::cout << "Coordinator listening on " << address
              << " with " << shard_addresses.size() << " shard(s):";
    for (const auto& addr : shard_addresses) std::cout << " " << addr;
    std::cout << "\n";

    server->Wait();
    return 0;
}
