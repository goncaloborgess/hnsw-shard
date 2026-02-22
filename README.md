# VectorDB-DS

## Structure

```
VectorDB-DS/
├── include/        # header files (.h)
├── src/
│   └── main.cpp   # source files
└── CMakeLists.txt
```

## Build & Run

```bash
mkdir build && cd build
cmake ..
make
./VectorDB-DS
```

## Development

1. Add headers to `include/`
2. Add source files to `src/`
3. Register new `.cpp` files in `CMakeLists.txt` under `add_executable`
4. Rebuild from the `build/` directory with `make`
