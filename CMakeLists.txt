cmake_minimum_required(VERSION 3.15)
project(VectorDB-DS)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/mathVector.cpp
    src/vector_store.cpp
    src/brute_force_index.cpp
    src/hnsw_index.cpp
)
