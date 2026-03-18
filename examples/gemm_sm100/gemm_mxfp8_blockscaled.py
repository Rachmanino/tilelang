# MXFP8 Block-Scaled GEMM on SM100 (Blackwell)
# DeepGEMM-style warp-specialized schedule:
#   Warp 0: TMA loads (A, B data + SFA, SFB scale factors)
#   Warp 1: MMA issue (SF transpose + UTCCP SF copy + block-scaled tcgen05.mma)
#
# Scale factor layout: per-block E8M0 scaling with granularity of 128 elements along K.
# Global SF tensors store 4 packed E8M0 values per uint32 (matching DeepGEMM convention).
# Each uint32 covers 4 * sf_granularity_k = 512 K elements.

import torch
import tilelang
import tilelang.language as T
from tilelang.layout import make_full_bank_swizzled_layout
from tilelang.profiler import do_bench

tilelang.disable_cache()


@tilelang.jit
def mxfp8_blockscaled_gemm(
    A, B, SFA, SFB,
    block_M, block_N, block_K,
    in_dtype, out_dtype, accum_dtype,
    num_stages,
    sf_granularity_k=128,
):
    """Block-scaled MXFP8 GEMM.

    A:   [M, K] in FP8 (E4M3 or E5M2)
    B:   [K, N] in FP8 (E4M3 or E5M2)
    SFA: [M, ceil(K / (sf_granularity_k * 4))] in uint32 (4 packed E8M0 scale factors for A)
    SFB: [N, ceil(K / (sf_granularity_k * 4))] in uint32 (4 packed E8M0 scale factors for B)
    """
    M, N, K = T.const("M, N, K")

    k_iters = T.ceildiv(K, block_K)
    # 4 packed E8M0 per uint32 → load every 4 stages
    sf_load_period = sf_granularity_k * 4 // block_K

    A: T.Tensor[[M, K], in_dtype]
    B: T.Tensor[[K, N], in_dtype]
    SFA: T.Tensor[[M, T.ceildiv(K, sf_granularity_k * 4)], "uint32"]
    SFB: T.Tensor[[N, T.ceildiv(K, sf_granularity_k * 4)], "uint32"]
    C = T.empty((M, N), out_dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        # Data shared memory (pipelined)
        A_shared = T.alloc_shared((num_stages, block_M, block_K), in_dtype)
        B_shared = T.alloc_shared((num_stages, block_K, block_N), in_dtype)

        # Scale factor shared memory — uint32 (128 elements = 512 bytes per stage)
        # TMA loads packed uint32 directly from global memory.
        SFA_shared = T.alloc_shared((num_stages, block_M), "uint32")
        SFB_shared = T.alloc_shared((num_stages, block_N), "uint32")

        # Accumulator in tensor memory
        C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)

        # Scale factors in tensor memory
        # UTCCP copies 128 uint32 per call = 4 TMEM columns per call
        # For block_M/block_N > 128, need multiple UTCCP calls
        sfa_num_chunks = block_M // 128  # number of 128-element UTCCP chunks
        sfb_num_chunks = block_N // 128
        SFA_tmem = T.alloc_tmem([32, sfa_num_chunks * 4], "uint32")
        SFB_tmem = T.alloc_tmem([32, sfb_num_chunks * 4], "uint32")

        # Output buffers
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        C_shared = T.alloc_shared((block_M, block_N), out_dtype)

        # Barriers
        loaded = T.alloc_barrier([32] * num_stages)
        consumed = T.alloc_barrier([1] * num_stages)
        tmem_full = T.alloc_barrier([1])

        # Annotate shared memory layouts so TMA writes data in the swizzled format
        # that the MMA descriptor expects (128B swizzle = full-bank swizzle).
        # Without this, TMA defaults to linear layout, causing a swizzle mismatch.
        T.annotate_layout({
            A_shared: make_full_bank_swizzled_layout(A_shared),
            B_shared: make_full_bank_swizzled_layout(B_shared),
            C_tmem: T.make_blockscaled_gemm_layout(C_tmem, A_shared[0, :, :]),
        })

        tx = T.get_thread_binding()
        T.use_swizzle(8)

        if tx < 32:
            # Warp 0: TMA load
            for k in T.serial(k_iters):
                T.mbarrier_wait_parity(consumed[k % num_stages], ((k // num_stages) & 1) ^ 1)
                T.tma_copy(
                    A[by * block_M:(by + 1) * block_M, k * block_K:(k + 1) * block_K],
                    A_shared[k % num_stages, :, :],
                    barrier=loaded[k % num_stages],
                )
                T.tma_copy(
                    B[k * block_K:(k + 1) * block_K, bx * block_N:(bx + 1) * block_N],
                    B_shared[k % num_stages, :, :],
                    barrier=loaded[k % num_stages],
                )
                # Load packed SF every sf_load_period iterations
                if k % sf_load_period == 0:
                    sf_k_idx = k // sf_load_period
                    T.copy(
                        SFA[by * block_M:(by + 1) * block_M, sf_k_idx:sf_k_idx + 1],
                        SFA_shared[k % num_stages, :],
                    )
                    T.copy(
                        SFB[bx * block_N:(bx + 1) * block_N, sf_k_idx:sf_k_idx + 1],
                        SFB_shared[k % num_stages, :],
                    )
                T.mbarrier_arrive(loaded[k % num_stages])

        elif tx < 64:
            # Warp 1: MMA issue + SF transpose/UTCCP
            for k in T.serial(k_iters):
                T.mbarrier_wait_parity(loaded[k % num_stages], (k // num_stages) & 1)
                T.tcgen05_after_thread_sync()

                # SF transpose + UTCCP when new SF was loaded
                # Each call handles 128 uint32 elements = 4 TMEM columns
                if k % sf_load_period == 0:
                    for ci in range(sfa_num_chunks):
                        T.sf_warp_transpose(SFA_shared[k % num_stages, ci * 128])
                    for ci in range(sfb_num_chunks):
                        T.sf_warp_transpose(SFB_shared[k % num_stages, ci * 128])
                    for ci in range(sfa_num_chunks):
                        T.tcgen05_cp(SFA_shared[k % num_stages, ci * 128], SFA_tmem, tmem_col_offset=ci * 4)
                    for ci in range(sfb_num_chunks):
                        T.tcgen05_cp(SFB_shared[k % num_stages, ci * 128], SFB_tmem, tmem_col_offset=ci * 4)

                # sf_id selects which of the 4 packed E8M0 values to use
                T.blockscaled_gemm(
                    A_shared[k % num_stages, :, :],
                    B_shared[k % num_stages, :, :],
                    C_tmem,
                    SFA_tmem,
                    SFB_tmem,
                    mbar=consumed[k % num_stages],
                    clear_accum=k == 0,
                    sf_a_id=k % sf_load_period,
                    sf_b_id=k % sf_load_period,
                )

            T.tcgen05_mma_arrive(tmem_full)

        # Epilogue: all warps
        T.mbarrier_wait_parity(tmem_full, 0)
        T.sync_threads()

        T.copy(C_tmem, C_local)
        T.copy(C_local, C_shared)
        T.copy(C_shared, C[by * block_M, bx * block_N])

    return C


def main():
    M, N, K = 8192, 8192, 8192
    block_M, block_N, block_K = 128, 256, 128
    in_dtype, out_dtype, accum_dtype = T.float8_e4m3fn, T.bfloat16, T.float
    num_stages = 4
    sf_granularity_k = 128

    a = torch.randn(M, K, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    # this is correct 
    # b = torch.full((K, N), 3, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    # not correct 
    b = torch.randn(K, N, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)

    # Pack scale factors: 4 uint8 E8M0 values per uint32
    # Each uint32 covers 4 * sf_granularity_k = 512 K elements
    sf_k_blocks = (K + sf_granularity_k - 1) // sf_granularity_k
    sf_k_packed = (sf_k_blocks + 3) // 4    

    # Create unpacked E8M0 scale factors (exponent 127 = scale 1.0)
    sfa_unpacked = torch.full((M, sf_k_blocks), 127, device="cuda", dtype=torch.uint8)
    sfb_unpacked = torch.full((N, sf_k_blocks), 127, device="cuda", dtype=torch.uint8)

    # Pad to multiple of 4 and pack into uint32
    if sf_k_blocks % 4 != 0:
        pad = 4 - sf_k_blocks % 4
        sfa_unpacked = torch.nn.functional.pad(sfa_unpacked, (0, pad), value=127)
        sfb_unpacked = torch.nn.functional.pad(sfb_unpacked, (0, pad), value=127)

    # Pack: [M, sf_k_packed, 4] uint8 → [M, sf_k_packed] uint32
    sfa = sfa_unpacked.view(M, sf_k_packed, 4).contiguous().view(torch.uint32).squeeze(-1).contiguous()
    sfb = sfb_unpacked.view(N, sf_k_packed, 4).contiguous().view(torch.uint32).squeeze(-1).contiguous()

    c = mxfp8_blockscaled_gemm(
        a, b, sfa, sfb,
        block_M, block_N, block_K,
        in_dtype, out_dtype, accum_dtype,
        num_stages,
        sf_granularity_k,
    )
    print(mxfp8_blockscaled_gemm.get_kernel_source(
        a, b, sfa, sfb,
        block_M, block_N, block_K,
        in_dtype, out_dtype, accum_dtype,
        num_stages,
        sf_granularity_k,
    ))

    ref_c = (a.to(torch.float) @ b.to(torch.float)).to(torch.bfloat16)
    print(f"Output shape: {c.shape}, dtype: {c.dtype}")
    print(f"Max abs error: {(c.float() - ref_c.float()).abs().max().item():.6f}")
    print(f'{c=}, {ref_c=}')

    tl_latency = do_bench(
        lambda: mxfp8_blockscaled_gemm(
            a, b, sfa, sfb,
            block_M, block_N, block_K,
            in_dtype, out_dtype, accum_dtype,
            num_stages,
            sf_granularity_k,
        ),
        backend="cupti",
    )
    print(f"Tilelang MXFP8 latency: {tl_latency} ms")
    print(f"TFLOPS: {2 * M * N * K / (tl_latency / 1e3) / 1e12:.2f}")


if __name__ == "__main__":
    main()
