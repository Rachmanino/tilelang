// Manual stages=3 implementation
// 145 TFLOPS?

#include <tl_templates/cuda/instruction/tcgen05mma.h>
#include <tl_templates/cuda/tcgen_05.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void gemm_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, __grid_constant__ const CUtensorMap C_desc);
extern "C" __global__ void __launch_bounds__(128, 1) gemm_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, __grid_constant__ const CUtensorMap C_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ uint C_tmem[1];
  // __shared__ uint64_t mbar_mem[1];
  // auto mbar = reinterpret_cast<Barrier*>(mbar_mem);
  __shared__ uint64_t mbarrier_mem[6];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  tl::Tcgen05SMemDescriptor desc_a;
  tl::Tcgen05SMemDescriptor desc_b;
  float C_local[128];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
    tl::prefetch_tma_descriptor(C_desc);
  }
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_allocate((&(C_tmem[0])), 128);
  }
  __syncthreads();
  if (tl::tl_shuffle_elect<0>()) {
    // mbar[0].init(1);
    mbarrier[0].init(32);  // loaded
    mbarrier[1].init(32);
    mbarrier[2].init(32);
    mbarrier[3].init(1);  // consumed
    mbarrier[4].init(1);
    mbarrier[5].init(1);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if ((32 <= ((int)threadIdx.x)) && (((int)threadIdx.x) < 64)) {
    for (int k = 0; k < 32; ++k) {
      mbarrier[((k % 3) + 3)].wait((((k % 6) / 3) ^ 1));
      if (tl::tl_shuffle_elect<32>()) {
        mbarrier[(k % 3)].expect_transaction(32768);
        tl::fence_proxy_async();
        tl::tma_load(A_desc, mbarrier[(k % 3)], (&(((bfloat16_t*)buf_dyn_shmem)[((k % 3) * 16384)])), (k * 128), (((int)blockIdx.y) * 128));
        tl::tma_load(A_desc, mbarrier[(k % 3)], (&(((bfloat16_t*)buf_dyn_shmem)[(((k % 3) * 16384) + 8192)])), ((k * 128) + 64), (((int)blockIdx.y) * 128));
        mbarrier[(k % 3)].expect_transaction(32768);
        tl::fence_proxy_async();
        tl::tma_load(B_desc, mbarrier[(k % 3)], (&(((bfloat16_t*)buf_dyn_shmem)[(((k % 3) * 16384) + 49152)])), (((int)blockIdx.x) * 128), (k * 128));
        tl::tma_load(B_desc, mbarrier[(k % 3)], (&(((bfloat16_t*)buf_dyn_shmem)[(((k % 3) * 16384) + 57344)])), ((((int)blockIdx.x) * 128) + 64), (k * 128));
      }
      mbarrier[(k % 3)].arrive();
    }
  } else if (((int)threadIdx.x) < 32) {
    for (int k = 0; k < 32; ++k) {
        mbarrier[(k % 3)].wait(((k % 6) / 3));
        tl::initialize_tcgen05_descriptor(desc_a, (&(((bfloat16_t*)buf_dyn_shmem)[(k%3)*16384])), 1, 64, 0, 0, 2);
        tl::initialize_tcgen05_descriptor(desc_b, (&(((bfloat16_t*)buf_dyn_shmem)[(k%3)*16384+49152])), 1024, 64, 0, 0, 2);
        #pragma unroll
        for (int ki = 0; ki < 8; ++ki) {
            tl::tcgen05mma_ss<tl::DataType::kBFloat16>(uint64_t(desc_a + (((ki >> 2) * 16384) + ((ki & 3) * 32))), uint64_t(desc_b + (ki * 2048)), (*reinterpret_cast<uint32_t*>(C_tmem)) + 0, ((0 < ki) ? 1 : ((k == 0) ? 0 : 1)), static_cast<uint32_t>(136381584), 0, 0, 0, 0);
        }
        tl::tcgen05_mma_arrive((&(mbarrier[((k % 3) + 3)])));
        mbarrier[((k % 3) + 3)].wait((k%6)/3);
    }
  }
  __syncthreads();
  tl::tcgen05_ld_32dp32bNx<128, false>(C_tmem[0], 0, (&(C_local[0])));
  __syncthreads();
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    uint2 __1;
    float4 v_ = *(float4*)(C_local + (i * 4));
    (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
    (reinterpret_cast<__nv_bfloat162*>(&__1))[1] = __float22bfloat162_rn(((float2*)(&v_))[1]);
    *(uint2*)(((bfloat16_t*)buf_dyn_shmem) + (((((((i >> 4) * 8192) + (((int)threadIdx.x) * 64)) + (((((i & 15) >> 3) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((i & 7) >> 2) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((i & 3) >> 1) + (((int)threadIdx.x) & 1)) & 1) * 8)) + ((i & 1) * 4))) = __1;
  }
  tl::fence_proxy_async();
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    tl::tma_store(C_desc, (&(((bfloat16_t*)buf_dyn_shmem)[0])), (((int)blockIdx.x) * 128), (((int)blockIdx.y) * 128));
    tl::tma_store_arrive();
    tl::tma_store_wait<0>();
    tl::tma_store(C_desc, (&(((bfloat16_t*)buf_dyn_shmem)[8192])), ((((int)blockIdx.x) * 128) + 64), (((int)blockIdx.y) * 128));
    tl::tma_store_arrive();
    tl::tma_store_wait<0>();
  }
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_deallocate((&(C_tmem[0])), 128);
  }
}