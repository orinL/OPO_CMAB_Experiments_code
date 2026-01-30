# Vowpal Wabbit

This is the Vowpal Wabbit fast online learning code.

## Description

Vowpal Wabbit is a machine learning system which pushes the frontier of machine learning with techniques such as online, hashing, allreduce, reductions, learning2search, active, and interactive learning.

## Build Requirements

### System Dependencies
- **C++ compiler** with C++11 support (GCC 4.7+ or Clang 3.4+)
- **CMake** (≥3.10) for building
- **curl** for downloading dependencies
- **tar** for extracting archives

### External Dependencies
This build requires specific versions of external libraries that must be downloaded manually:

- **Boost 1.83.0** (program_options, system libraries)
- **fmt 7.1.3** 
- **spdlog 1.8.2**
- **rapidjson 1.1.0**

## Building Instructions

### Step 1: Download External Dependencies

The external libraries need to be populated in the `ext_libs/` directory:

```bash
# Download fmt 7.1.3
cd ext_libs/fmt
curl -L -o fmt.tar.gz https://github.com/fmtlib/fmt/archive/7.1.3.tar.gz
tar -xzf fmt.tar.gz --strip-components=1
rm fmt.tar.gz

# Download spdlog 1.8.2  
cd ../spdlog
curl -L -o spdlog.tar.gz https://github.com/gabime/spdlog/archive/v1.8.2.tar.gz
tar -xzf spdlog.tar.gz --strip-components=1
rm spdlog.tar.gz

# Download rapidjson 1.1.0
cd ../rapidjson
curl -L -o rapidjson.tar.gz https://github.com/Tencent/rapidjson/archive/v1.1.0.tar.gz
tar -xzf rapidjson.tar.gz --strip-components=1
rm rapidjson.tar.gz
cd ../..
```

### Step 2: Build Boost 1.83.0 (if not available system-wide)

```bash
# Create directory for external dependencies (from project root)
mkdir -p ../external_libs
cd ../external_libs

# Download Boost 1.83.0
curl -L -O https://archives.boost.io/release/1.83.0/source/boost_1_83_0.tar.bz2
tar -xjf boost_1_83_0.tar.bz2
cd boost_1_83_0

# Configure Boost for required libraries only
./bootstrap.sh --with-libraries=program_options,system

# Build Boost libraries
./b2 --with-program_options --with-system

cd ../../vowpal_wabbit
```

### Step 3: Build VowpalWabbit

```bash
# Create and enter build directory
mkdir -p build
cd build

# Configure CMake with local Boost installation
cmake -DBOOST_ROOT=$(pwd)/../../external_libs/boost_1_83_0 \
      -DBoost_INCLUDE_DIR=$(pwd)/../../external_libs/boost_1_83_0 \
      -DBoost_LIBRARY_DIR=$(pwd)/../../external_libs/boost_1_83_0/stage/lib \
      -DBUILD_TESTS=OFF ..

# Build VowpalWabbit library and executable
make -j$(sysctl -n hw.ncpu) vw vw-bin

# Test the build
./vowpalwabbit/vw --version
```

### Expected Output

If successful, you should see:
```
8.11.0 (git commit: )
```

### VowpalWabbit Executable Location

The built executable will be located at:
```
build/vowpalwabbit/vw
```

Use the full absolute path when configuring experiments:
```
$(pwd)/build/vowpalwabbit/vw
```

## Troubleshooting

### External Dependencies Issues
- Ensure all external libraries are downloaded with exact versions specified
- Check that CMakeLists.txt files exist in `ext_libs/fmt/`, `ext_libs/spdlog/`, and `ext_libs/rapidjson/`

### Boost Issues  
- Verify Boost libraries were built: `ls ../external_libs/boost_1_83_0/stage/lib/`
- Should see `libboost_program_options.*` and `libboost_system.*` files

### Compiler Warnings
- Function type cast warnings are normal and can be ignored
- Build should complete with `[100%] Built target vw-bin`

## Usage in OPOCMAB Experiments

This code has been built and configured for the OPOCMAB experiments. When setting up experiments, use the full path to the VowpalWabbit executable in your configuration files. 