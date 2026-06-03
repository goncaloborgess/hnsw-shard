cmake_minimum_required(VERSION 3.15)
project(VectorDB-DS)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/anaconda3")

find_package(Threads REQUIRED)
find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)

# ---------------------------------------------------------------------------
# Proto code generation
# ---------------------------------------------------------------------------

set(PROTO_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/proto")
set(GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY ${GENERATED_DIR})

set(PROTO_FILE "${PROTO_SRC_DIR}/vectordb.proto")
set(PB_CC  "${GENERATED_DIR}/vectordb.pb.cc")
set(PB_H   "${GENERATED_DIR}/vectordb.pb.h")
set(GRPC_CC "${GENERATED_DIR}/vectordb.grpc.pb.cc")
set(GRPC_H  "${GENERATED_DIR}/vectordb.grpc.pb.h")

add_custom_command(
    OUTPUT ${PB_CC} ${PB_H} ${GRPC_CC} ${GRPC_H}
    COMMAND protobuf::protoc
        --proto_path=${PROTO_SRC_DIR}
        --cpp_out=${GENERATED_DIR}
        --grpc_out=${GENERATED_DIR}
        --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
        ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
    COMMENT "Generating C++ sources from vectordb.proto"
)

add_library(vectordb_proto STATIC ${PB_CC} ${GRPC_CC})
target_include_directories(vectordb_proto PUBLIC ${GENERATED_DIR})
target_link_libraries(vectordb_proto PUBLIC protobuf::libprotobuf gRPC::grpc++)

# ---------------------------------------------------------------------------
# Core engine library
# ---------------------------------------------------------------------------

add_library(vectordb_core STATIC
    src/mathVector.cpp
    src/vector_store.cpp
    src/brute_force_index.cpp
    src/hnsw_index.cpp
)
target_include_directories(vectordb_core PUBLIC include)

# ---------------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------------

# Tests and benchmark (the original binary)
add_executable(vectordb_tests src/main.cpp)
target_link_libraries(vectordb_tests PRIVATE vectordb_core)

# gRPC server
add_executable(vectordb_server src/server.cpp)
target_link_libraries(vectordb_server PRIVATE vectordb_core vectordb_proto)

# gRPC client
add_executable(vectordb_client src/client.cpp)
target_link_libraries(vectordb_client PRIVATE vectordb_proto)
