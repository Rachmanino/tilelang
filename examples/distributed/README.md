# Distributed Examples

This directory contains examples demonstrating distributed computing capabilities using TileLang.

For example, 
```
./tilelang/distributed/launch.sh examples/distributed/example_allgather.py
```

## Prerequisites

Before running the examples, you need to install the `pynvshmem` package, which provides wrapped Python API for NVSHMEM.
```bash 
export NVSHMEM_SRC="your_custom_nvshmem_dir" 
# default to 3rdparty/nvshmem_src
cd tilelang/distributed
source build_nvshmem.sh
cd ./pynvshmem
python setup.py install
export LD_LIBRARY_PATH="$NVSHMEM_SRC/build/src/lib:$LD_LIBRARY_PATH"
```

Then you can test python import:
```bash
python -c "import pynvshmem"
```