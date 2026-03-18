"""GEMM (General Matrix Multiplication) operators exposed on the TileLang language surface."""

from __future__ import annotations

from tilelang._typing import BufferLikeType, BarrierType
from tilelang.tileop.base import GemmWarpPolicy
import tilelang.language as T
from tvm import tir
from tilelang.utils.language import (
    to_buffer_region,
    retrieve_shape,
    retrieve_stride,
    retrieve_offset,
    prim_expr_equal,
)
from tilelang.language.utils import (
    buffer_region_to_tile_region,
)
from tilelang.env import env as _env


def _gemm_impl(
    op_key: str,
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
    mbar: BarrierType | None = None,
) -> tir.PrimExpr:
    """Shared GEMM implementation.

    Returns a call_intrin handle for the given op key.
    """

    def legalize_arguments(arg: BufferLikeType | tir.Var) -> BufferLikeType:
        """Convert let-bound variables to their corresponding buffers.

        Args:
            arg (Union[tir.Buffer, tir.Var]): Input argument to legalize

        Returns:
            Union[tir.Buffer, tir.Var]: The legalized argument
        """
        if isinstance(arg, tir.Var) and T.has_let_value(arg):
            return T.get_let_value(arg).buffer
        return arg

    A = legalize_arguments(A)
    B = legalize_arguments(B)
    C = legalize_arguments(C)
    mbar = legalize_arguments(mbar) if mbar is not None else None

    # Normalize A/B/C to BufferRegion for shape/stride/offset analysis
    A_region = to_buffer_region(A)
    B_region = to_buffer_region(B)
    C_region = to_buffer_region(C)

    A_shape = retrieve_shape(A_region)
    B_shape = retrieve_shape(B_region)
    C_shape = retrieve_shape(C_region)

    A_stride = retrieve_stride(A_region)
    B_stride = retrieve_stride(B_region)

    assert len(C_shape) == 2, "current only support C as a 2D tensor"
    assert len(A_shape) >= 2, "current only support A as a 2D or higher-order tensor"
    assert len(B_shape) >= 2, "current only support B as a 2D or higher-order tensor"
    if len(A_shape) > 2:
        for i in range(len(A_shape) - 2):
            assert A_shape[i] == 1, (
                "current only support A as a 2D or higher-order tensor with the last two dimensions being the matrix dimensions"
            )
    if len(B_shape) > 2:
        for i in range(len(B_shape) - 2):
            assert B_shape[i] == 1, (
                "current only support B as a 2D or higher-order tensor with the last two dimensions being the matrix dimensions"
            )

    M, N = C_shape
    K = A_shape[-2] if transpose_A else A_shape[-1]
    K_B = B_shape[-1] if transpose_B else B_shape[-2]
    assert prim_expr_equal(K, K_B), f"T.gemm K shape check failed: K_A = {K}, K_B = {K_B}"

    stride_a = A_stride[-2]
    stride_b = B_stride[-2]

    A_offset = retrieve_offset(A_region)
    B_offset = retrieve_offset(B_region)
    assert A_offset[-2] == 0, "The offset of the first dimension of A must be 0"
    assert B_offset[-2] == 0, "The offset of the first dimension of B must be 0"
    offset_a = A_offset[-1]
    offset_b = B_offset[-1]

    if mbar is not None:
        assert isinstance(mbar, (tir.Buffer, tir.BufferLoad)), (
            f"mbar for tcgen5mma must be a tir.Buffer or tir.BufferLoad, but got {type(mbar)}"
        )
        mbar = to_buffer_region(mbar, access_type="rw")
    C_coords = [r.min for r in C_region.region]
    # Convert BufferRegion to tl.region calls for arguments
    A_arg = buffer_region_to_tile_region(A_region, "r", [r for r in A_shape])
    B_arg = buffer_region_to_tile_region(B_region, "r", [r for r in B_shape])
    C_arg = buffer_region_to_tile_region(C_region, "rw", [r for r in C_shape])
    # When mbar is None, pass a placeholder constant (0).
    # The C++ side checks if arg 16 is a BufferLoadNode before using it,
    # so a non-BufferLoad value will be correctly ignored.
    mbar_arg = mbar if mbar is not None else tir.const(0, dtype="int32")
    return tir.call_intrin(
        "handle",
        tir.op.Op.get(op_key),
        A_arg,
        B_arg,
        C_arg,
        transpose_A,
        transpose_B,
        M,
        N,
        K,
        policy,
        clear_accum,
        stride_a,
        stride_b,
        offset_a,
        offset_b,
        k_pack,
        wg_wait,
        mbar_arg,
        C_coords[0],
        C_coords[1],
    )


# Public wrappers
def gemm_v1(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
    mbar: BarrierType | None = None,
) -> tir.PrimExpr:
    """GEMM v1: use op tl.gemm."""
    return _gemm_impl(
        "tl.tileop.gemm",
        A,
        B,
        C,
        transpose_A,
        transpose_B,
        policy,
        clear_accum,
        k_pack,
        wg_wait,
        mbar,
    )


# experimental currently, for fast compilation
def gemm_v2(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
    mbar: BarrierType | None = None,
) -> tir.PrimExpr:
    """GEMM v2: use op tl.gemm_py."""
    return _gemm_impl(
        "tl.tileop.gemm_py",
        A,
        B,
        C,
        transpose_A,
        transpose_B,
        policy,
        clear_accum,
        k_pack,
        wg_wait,
        mbar,
    )


def gemm(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    wg_wait: int = 0,
    mbar: BarrierType | None = None,
) -> tir.PrimExpr:
    """TileLang GEMM operator.

    Args:
        A (BufferLikeType, i.e. Buffer | BufferLoad | BufferRegion, or Var): Input buffer A.
        B (BufferLikeType): Input buffer B.
        C (BufferLikeType): Output buffer C.
        transpose_A (bool): Whether to transpose A. Defaults to False.
        transpose_B (bool): Whether to transpose B. Defaults to False.
        policy (GemmWarpPolicy): GEMM warp partition policy.
        clear_accum (bool): Whether to clear the accumulator.
        k_pack (int): Numbers of packed matrix cores, for ROCm only. Defaults to 1.
        wg_wait (int): Int identifier of the warpgroup MMA batch to wait on.. Defaults to 0.
        mbar (BarrierType, i.e. Buffer | BufferLoad, or Var, optional): Mbarrier in Blackwell. Defaults to None.

    Returns:
        tir.Call: A handle to the GEMM operation.
    """
    impl = gemm_v1 if _env.use_gemm_v1() else gemm_v2
    return impl(A, B, C, transpose_A, transpose_B, policy, clear_accum, k_pack, wg_wait, mbar)


def blockscaled_gemm(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    SFA_tmem: BufferLikeType,
    SFB_tmem: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    clear_accum=False,
    wg_wait: int = 0,
    mbar: BarrierType | None = None,
    sf_a_id: int = 0,
    sf_b_id: int = 0,
) -> tir.PrimExpr:
    """Block-scaled GEMM for MXFP8 on SM100 (Blackwell).

    Issues ``tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale`` instructions.
    A, B are FP8 (E4M3/E5M2) in shared memory, C is accumulator in tensor memory.
    SFA_tmem, SFB_tmem are E8M0 scale factors already in tensor memory (loaded via UTCCP).

    This function is designed for warp-specialized schedules where one warp issues MMA.
    Scale factor loading (TMA → smem → transpose → UTCCP → tmem) must be managed by
    the user schedule (see examples/gemm_sm100/gemm_mxfp8_blockscaled.py).

    Args:
        A: FP8 input buffer A in shared memory.
        B: FP8 input buffer B in shared memory.
        C: FP32 accumulator in tensor memory.
        SFA_tmem: Scale factors for A in tensor memory.
        SFB_tmem: Scale factors for B in tensor memory.
        transpose_A: Whether A is MN-major. Default: False (K-major).
        transpose_B: Whether B is K-major. Default: False (MN-major).
        clear_accum: Whether to zero the accumulator.
        wg_wait: Warp group wait identifier.
        mbar: Mbarrier for MMA completion signaling.
        sf_a_id: Scale factor ID for A (0-3).
        sf_b_id: Scale factor ID for B (0-3).
    """
    from tilelang.intrinsics.tcgen05_macro_generator import (
        TensorCoreIntrinEmitter,
    )
    from tilelang.layout import make_full_bank_swizzled_layout

    def legalize(arg):
        if isinstance(arg, tir.Var) and T.has_let_value(arg):
            return T.get_let_value(arg).buffer
        return arg

    A = legalize(A)
    B = legalize(B)
    C = legalize(C)
    SFA_tmem = legalize(SFA_tmem)
    SFB_tmem = legalize(SFB_tmem)
    mbar = legalize(mbar) if mbar is not None else None

    A_region = to_buffer_region(A)
    B_region = to_buffer_region(B)
    C_region = to_buffer_region(C)

    A_shape = retrieve_shape(A_region)
    B_shape = retrieve_shape(B_region)
    C_shape = retrieve_shape(C_region)

    M, N = int(C_shape[0]), int(C_shape[1])
    K = int(A_shape[-2] if transpose_A else A_shape[-1])

    a_dtype = str(A_region.buffer.dtype)
    accum_dtype = str(C_region.buffer.dtype)

    # Create intrinsic emitter — for block-scaled, TCGEN5 always uses 1 warp group
    emitter = TensorCoreIntrinEmitter(
        a_dtype=a_dtype,
        b_dtype=a_dtype,
        accum_dtype=accum_dtype,
        a_transposed=transpose_A,
        b_transposed=transpose_B,
        block_row_warps=1,
        block_col_warps=1,
        warp_row_tiles=M,
        warp_col_tiles=N,
        chunk=K,
    )

    # Assign shared layouts (default: full-bank swizzled for FP8)
    a_buf = A_region.buffer if isinstance(A_region, tir.BufferRegion) else A
    b_buf = B_region.buffer if isinstance(B_region, tir.BufferRegion) else B
    emitter._assign_a_shared_layout(make_full_bank_swizzled_layout(a_buf))
    emitter._assign_b_shared_layout(make_full_bank_swizzled_layout(b_buf))

    # Convert mbar to a pointer, same as the regular gemm tile-op does
    from tilelang.utils.language import retrieve_ptr as _retrieve_ptr
    mbarptr = _retrieve_ptr(mbar, "rw") if mbar is not None else None
    return emitter.tcgen05mma_blockscaled(
        A, B, C, SFA_tmem, SFB_tmem,
        mbarptr, clear_accum, sf_a_id, sf_b_id,
    )


def make_blockscaled_gemm_layout(
    C: BufferLikeType,
    A: BufferLikeType,
    transpose_A: bool = False,
) -> "Layout":
    """Build the TMEM store layout for the C accumulator of a block-scaled GEMM.

    Users must call ``T.annotate_layout({C_tmem: layout})`` with the returned layout
    so that subsequent ``T.copy(C_tmem, ...)`` can be lowered correctly.

    Args:
        C: The TMEM accumulator buffer (block_M, block_N).
        A: The FP8 operand A buffer (used to infer K and dtype).
        transpose_A: Whether A is MN-major.

    Returns:
        A Layout object for C's TMEM storage.
    """
    from tilelang.intrinsics.tcgen05_macro_generator import TensorCoreIntrinEmitter

    C_region = to_buffer_region(C)
    A_region = to_buffer_region(A)

    C_shape = retrieve_shape(C_region)
    A_shape = retrieve_shape(A_region)

    M, N = int(C_shape[0]), int(C_shape[1])
    K = int(A_shape[-2] if transpose_A else A_shape[-1])
    a_dtype = str(A_region.buffer.dtype)
    accum_dtype = str(C_region.buffer.dtype)

    emitter = TensorCoreIntrinEmitter(
        a_dtype=a_dtype,
        b_dtype=a_dtype,
        accum_dtype=accum_dtype,
        a_transposed=transpose_A,
        b_transposed=False,
        block_row_warps=1,
        block_col_warps=1,
        warp_row_tiles=M,
        warp_col_tiles=N,
        chunk=K,
    )

    c_buf = C_region.buffer if isinstance(C_region, tir.BufferRegion) else C
    return emitter.make_mma_store_layout(c_buf)
