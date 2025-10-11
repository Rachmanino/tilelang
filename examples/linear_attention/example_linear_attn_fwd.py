import torch
import tilelang as tl
import tilelang.language as T
from tilelang.profiler import do_bench
import argparse
from fla.ops.linear_attn import fused_chunk_linear_attn  # We compare with FLA


@tl.jit(out_idx=[3, 4], pass_configs={
    tl.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
})
def tl_fused_chunk_fwd_kernel(
    B,
    S,
    H,
    DK,
    DV,
    dtype: str = 'float16',
    scale: float = None,
) -> torch.Tensor:

    if scale is None:
        scale = DK**-0.5
    accum_dtype = 'float'

    chunk_size = 64
    BK = BV = 64  # Set to 128 can be faster, but has some numerical differences with FLA
    assert S % chunk_size == 0 and DK % BK == 0 and DV % BV == 0
    NK = tl.cdiv(DK, BK)
    NV = tl.cdiv(DV, BV)
    NT = tl.cdiv(S, chunk_size)

    @T.prim_func
    def main(
            Q: T.Tensor([B, S, H, DK], dtype),  # type: ignore
            K: T.Tensor([B, S, H, DK], dtype),  # type: ignore
            V: T.Tensor([B, S, H, DV], dtype),  # type: ignore
            O: T.Tensor([NK, B, S, H, DV], accum_dtype),  # type: ignore
            final_state: T.Tensor([B, H, DK, DV], accum_dtype)):  # type: ignore
        with T.Kernel(NV, NK, B * H) as (i_v, i_k, i_bh):
            i_b = i_bh // H
            i_h = i_bh % H

            q = T.alloc_shared([chunk_size, BK], dtype)
            k = T.alloc_shared([chunk_size, BK], dtype)
            v = T.alloc_shared([chunk_size, BV], dtype)
            h = T.alloc_fragment([BK, BV], accum_dtype)
            h_shared = T.alloc_shared([BK, BV], dtype)
            s = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)
            s_shared = T.alloc_shared([chunk_size, chunk_size], dtype)
            o = T.alloc_fragment([chunk_size, BV], accum_dtype)
            T.clear(h)

            T.use_swizzle(10)

            for i in T.Pipelined(0, NT, num_stages=0):
                for row, col in T.Parallel(chunk_size, BK):
                    q[row, col] = Q[i_b, i * chunk_size + row, i_h, i_k * BK + col] * scale
                T.copy(K[i_b, i * chunk_size:(i + 1) * chunk_size, i_h, i_k * BK:(i_k + 1) * BK], k)
                T.copy(V[i_b, i * chunk_size:(i + 1) * chunk_size, i_h, i_v * BV:(i_v + 1) * BV], v)
                
                T.gemm(q, k, s, clear_accum=True, transpose_B=True)
                for row, col in T.Parallel(chunk_size, chunk_size):
                    s_shared[row, col] = T.if_then_else(row >= col, s[row, col], 0)

                T.gemm(s_shared, v, o, clear_accum=True)
                T.copy(h, h_shared)
                T.gemm(k, v, h, transpose_A=True)
                T.gemm(q, h_shared, o)
                #TODO: use vectorized atomic add or tma reduce on sm90
                T.copy(o, O[i_k, i_b, i * chunk_size:(i + 1) * chunk_size, i_h,
                         i_v * BV:(i_v + 1) * BV])

            # Output final state
            T.copy(h, final_state[i_b, i_h, i_k * BK:(i_k + 1) * BK, i_v * BV:(i_v + 1) * BV])

    return main


def tl_fused_chunk_fwd(q, k, v):
    B, S, H, D = q.shape
    kernel = tl_fused_chunk_fwd_kernel(B, S, H, D, D)
    o, h = kernel(q, k, v)
    return o.sum(0).to(torch.float16), h


def main(B=1, S=1024, H=16, D=128):
    q = torch.randn((B, S, H, D), device='cuda', dtype=torch.float16)
    k = torch.randn((B, S, H, D), device='cuda', dtype=torch.float16)
    v = torch.randn((B, S, H, D), device='cuda', dtype=torch.float16)

    o, h = tl_fused_chunk_fwd(q, k, v)
    o_ref, h_ref = fused_chunk_linear_attn(q, k, v, output_final_state=True, normalize=False)

    assert torch.allclose(o, o_ref), f'o max err: {(o - o_ref).abs().max()}'
    assert torch.allclose(h, h_ref), f'h max err: {(h - h_ref).abs().max()}'
    print('Passed all tests!✅')

    t1 = do_bench(
        lambda: fused_chunk_linear_attn(q, k, v, output_final_state=True, normalize=False),
        warmup=25,
        rep=100)
    t2 = do_bench(lambda: tl_fused_chunk_fwd(q, k, v), warmup=25, rep=100)
    print(f'Triton latency: {t1:.3f} ms')
    print(f'TileLang latency: {t2:.3f} ms')
    print(f'Speedup: {t1/t2:.3f}x')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--B', type=int, default=8, help='Batch size')
    parser.add_argument('--S', type=int, default=4096, help='Seq len')
    parser.add_argument('--H', type=int, default=32, help='Num heads')
    parser.add_argument('--D', type=int, default=128, help='Head dim')
    args = parser.parse_args()

    main(args.B, args.S, args.H, args.D)
