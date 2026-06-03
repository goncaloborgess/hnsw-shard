#include "vectordb.grpc.pb.h"
#include <grpcpp/grpcpp.h>

#include <iostream>
#include <string>
#include <vector>

class VectorDBClient {
public:
    explicit VectorDBClient(std::shared_ptr<grpc::Channel> channel)
        : stub_(vectordb::VectorDBService::NewStub(channel)) {}

    bool upsert(const std::string& id, const std::vector<float>& vec) {
        vectordb::UpsertRequest request;
        request.set_id(id);
        for (float v : vec) request.add_vector(v);

        vectordb::UpsertResponse response;
        grpc::ClientContext context;

        grpc::Status status = stub_->Upsert(&context, request, &response);
        if (!status.ok()) {
            std::cerr << "  Upsert failed: " << status.error_message() << "\n";
            return false;
        }

        std::cout << "  Upserted '" << id << "' — "
                  << (response.created() ? "created" : "updated") << "\n";
        return true;
    }

    bool search(const std::vector<float>& query, uint32_t k) {
        vectordb::SearchRequest request;
        for (float v : query) request.add_query(v);
        request.set_k(k);

        vectordb::SearchResponse response;
        grpc::ClientContext context;

        grpc::Status status = stub_->Search(&context, request, &response);
        if (!status.ok()) {
            std::cerr << "  Search failed: " << status.error_message() << "\n";
            return false;
        }

        std::cout << "  Top-" << k << " results:\n";
        for (const auto& result : response.results()) {
            std::cout << "    id=\"" << result.id()
                      << "\"  distance=" << result.distance() << "\n";
        }
        return true;
    }

private:
    std::unique_ptr<vectordb::VectorDBService::Stub> stub_;
};

int main(int argc, char* argv[]) {
    const std::string target = (argc > 1) ? argv[1] : "localhost:50051";

    std::cout << "Connecting to " << target << "...\n\n";
    VectorDBClient client(
        grpc::CreateChannel(target, grpc::InsecureChannelCredentials()));

    // --- Upsert five vectors at known 2D positions ---
    std::cout << "[1] Inserting vectors\n";
    client.upsert("a", {1.0f, 1.0f});
    client.upsert("b", {2.0f, 2.0f});
    client.upsert("c", {3.0f, 3.0f});
    client.upsert("d", {4.0f, 4.0f});
    client.upsert("e", {5.0f, 5.0f});

    // --- Search: (2.1, 2.1) should be nearest to "b" ---
    std::cout << "\n[2] Searching for nearest to (2.1, 2.1), k=3\n";
    client.search({2.1f, 2.1f}, 3);

    // --- Overwrite "b" to a new position ---
    std::cout << "\n[3] Overwriting 'b' to (4.9, 4.9)\n";
    client.upsert("b", {4.9f, 4.9f});

    // --- Search again: "b" should now be near "e", not near (2.1, 2.1) ---
    std::cout << "\n[4] Searching again for nearest to (2.1, 2.1), k=3\n";
    client.search({2.1f, 2.1f}, 3);

    std::cout << "\nDone.\n";
    return 0;
}
