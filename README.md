# cuda-proj-tmpl

A minimal cmake based project skeleton for developping a CUDA application

## Download this skeleton

```bash
git clone --recursive git@github.com:pkestene/cuda-proj-tmpl.git
```

## Modern CMake and CUDA

See https://github.com/CLIUtils/modern_cmake

## Requirements

- cmake version >= 3.10
- cuda toolkit

## How to build ?

```bash
# set default CUDA flags passed to nvcc (Nvidia compiler wrapper)
# at least one hardware architecture flags
export CUDAFLAGS="-arch=sm_30 --expt-extended-lambda"
mkdir build
cd build
cmake ..
make
# then you can run the application
./src/saxpy_cuda
```
