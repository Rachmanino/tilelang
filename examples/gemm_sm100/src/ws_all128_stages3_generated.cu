#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void gemm_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, half_t* __restrict__ C);
extern "C" __global__ void __launch_bounds__(256, 1) gemm_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap B_desc, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[128];
  half_t A_local[32];
  half_t B_local[32];
  __shared__ uint64_t mbarrier_mem[6];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(B_desc);
    mbarrier[0].init(128);
    mbarrier[1].init(128);
    mbarrier[2].init(128);
    mbarrier[3].init(128);
    mbarrier[4].init(128);
    mbarrier[5].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (128 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    for (int k = 0; k < 32; ++k) {
      mbarrier[((k % 3) + 3)].wait((((k % 6) / 3) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(k % 3)].expect_transaction(32768);
        tl::fence_proxy_async();
        tl::tma_load(A_desc, mbarrier[(k % 3)], (&(((half_t*)buf_dyn_shmem)[((k % 3) * 16384)])), (k * 128), (((int)blockIdx.y) * 128));
        tl::tma_load(A_desc, mbarrier[(k % 3)], (&(((half_t*)buf_dyn_shmem)[(((k % 3) * 16384) + 8192)])), ((k * 128) + 64), (((int)blockIdx.y) * 128));
        mbarrier[(k % 3)].expect_transaction(32768);
        tl::fence_proxy_async();
        tl::tma_load(B_desc, mbarrier[(k % 3)], (&(((half_t*)buf_dyn_shmem)[(((k % 3) * 16384) + 49152)])), (((int)blockIdx.x) * 128), (k * 128));
        tl::tma_load(B_desc, mbarrier[(k % 3)], (&(((half_t*)buf_dyn_shmem)[(((k % 3) * 16384) + 57344)])), ((((int)blockIdx.x) * 128) + 64), (k * 128));
      }
      mbarrier[(k % 3)].arrive();
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float2*)(C_local + (i * 2)) = make_float2(broadcast_var, broadcast_var);
    }
    for (int k_1 = 0; k_1 < 32; ++k_1) {
      mbarrier[(k_1 % 3)].wait(((k_1 % 6) / 3));
      for (int ki = 0; ki < 8; ++ki) {
        for (int i_1 = 0; i_1 < 4; ++i_1) {
          tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[(((((((k_1 % 3) * 16384) + ((ki >> 2) * 8192)) + (((((int)threadIdx.x) & 63) >> 5) * 4096)) + (i_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511))])) + 0, A_local + (i_1 * 8));
        }
        for (int i_2 = 0; i_2 < 4; ++i_2) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)buf_dyn_shmem)[(((((((k_1 % 3) * 16384) + ((((int)threadIdx.x) >> 6) * 8192)) + (ki * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (i_2 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 49152)])) + 0, B_local + (i_2 * 8));
        }
        for (int i_3 = 0; i_3 < 4; ++i_3) {
          for (int j = 0; j < 4; ++j) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + ((i_3 * 32) + (j * 8))), reinterpret_cast<const unsigned*>(A_local + (i_3 * 8)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_local + (((i_3 * 32) + (j * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_3 * 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
          }
        }
      }
      mbarrier[((k_1 % 3) + 3)].arrive();
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 64; ++i_4) {
      uint1 __1;
      float2 v_ = *(float2*)(C_local + (i_4 * 2));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      *(uint1*)(C + (((((((((((int)blockIdx.y) * 524288) + (((((int)threadIdx.x) & 63) >> 5) * 262144)) + ((i_4 >> 4) * 65536)) + ((i_4 & 1) * 32768)) + (((((int)threadIdx.x) & 31) >> 2) * 4096)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 6) * 64)) + (((i_4 & 15) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __1;
    }
  }
}