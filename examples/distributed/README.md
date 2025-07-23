# Distributed Examples

This directory contains examples demonstrating distributed computing capabilities using TileLang.

For example, 
```bash
./tilelang/distributed/launch.sh examples/distributed/example_allgather.py
```

## Prerequisites

Before running the examples, you need to build NVSHMEM library for device-side code generation.

```bash 
export NVSHMEM_SRC="your_custom_nvshmem_dir" # default to 3rdparty/nvshmem_src
cd tilelang/distributed
bash build_nvshmem.sh
```

You also need to install the `nvshmem4py` package, which provides official Python bindings for host-side NVSHMEM APIs.

```bash
pip install nvidia-nvshmem-cu12==3.3.9
```

> Note: We have recently migrated from `pynvshmem` package to `nvshmem4py` for host-side API. Therefore, the correctness of examples here has not been fully verified yet.