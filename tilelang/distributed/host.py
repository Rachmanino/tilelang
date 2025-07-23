"""Provide common-used host-side bindings for NVSHMEM operations based on nvshmem4py"""

import nvshmem.core
import torch
from cuda import cuda, cudart
from .utils import CUDA_CHECK, create_nvshmem_stream
from typing import List, Optional, Sequence
    
    
### Query ###

def nvshmem_my_pe() -> int:
    """Get the current Processing Element (PE) ID of this process."""
    return nvshmem.core.my_pe()


def nvshmem_n_pes() -> int:
    """Get the total number of Processing Elements (PEs) in the default team."""
    return nvshmem.core.n_pes()


def nvshmem_team_my_pe(team) -> int:
    """Get the PE ID of this process within a specified team.
    Args:
        team: The team to query.
    Returns:
        int: The PE ID of this process within the specified team.
    """
    return nvshmem.core.team_my_pe(team)


def nvshmem_team_n_pes(team) -> int:
    """Get the number of Processing Elements (PEs) in a specified team.
    Args:
        team: The team to query.
    Returns:
        int: The number of PEs in the specified team.
    """
    return nvshmem.core.team_n_pes(team)


### Memory ### 

def nvshmem_malloc(size: int) -> torch.Tensor:
    raise NotImplementedError()
    #TODO: Do we really need this function?
    
    
def nvshmem_create_tensor(shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
    """Create a tensor that is symmetrically distributed across all PEs.
    Args:
        shape (Sequence[int]): The shape of the tensor to create.
        dtype (torch.dtype): The data type of the tensor to create.
    Returns:
        torch.Tensor: The local tensor, whose counterpart is symmetrically distributed across all PEs.
    """
    torch.cuda.synchronize()
    tensor = nvshmem.core.tensor(shape, dtype=dtype)
    setattr(tensor, "__symm_tensor__", True)  # mark the tensor as symmetrically distributed
    torch.cuda.synchronize()
    return tensor


def nvshmem_peer_tensor(tensor: torch.Tensor, peer: int) -> torch.Tensor:
    """Get the pointer of a tensor on a specific peer.
    Args:
        tensor (torch.Tensor): The local tensor.
        peer (int): The PE ID of the peer to get the tensor from.
    Returns:
        torch.Tensor: The tensor on the specified peer.
    """
    torch.cuda.synchronize()
    peer_tensor = nvshmem.core.get_peer_tensor(tensor, peer)
    torch.cuda.synchronize()
    return peer_tensor


def nvshmem_create_tensor_list_intra_node(shape: Sequence[int], dtype: torch.dtype) -> List[torch.Tensor]:
    """Create a list of tensors that are symmetrically distributed across all PEs in the same node.
    Args:
        shape (Sequence[int]): The shape of the tensor to create.
        dtype (torch.dtype): The data type of the tensor to create.
    Returns:
        List[torch.Tensor]: A list of tensors, one for each PE in the same node
    """
    local_rank = nvshmem_team_my_pe(nvshmem.core.Teams.TEAM_NODE)
    rank = nvshmem_my_pe()
    local_world_size = nvshmem_team_n_pes(nvshmem.core.Teams.TEAM_NODE)
    
    def _get_peer_tensor(t, peer) -> torch.Tensor:
        # avoid create tensor on the same buf again. nvshmem4py can't handle multiple reference with grace. so we handle it here.
        # https://forums.developer.nvidia.com/t/nvshmem4py-nvshmem-core-finalize-does-not-handle-everything/337979
        if peer == rank:
            return t
        return nvshmem.core.get_peer_tensor(t, peer)

    offset = rank - local_rank
    torch.cuda.synchronize()
    tensor = nvshmem_create_tensor(shape, dtype=dtype)
    torch.cuda.synchronize()
    return [_get_peer_tensor(tensor, i + offset) for i in range(local_world_size)]


def nvshmem_free_tensor(tensor):
    """Free a tensor that is symmetrically distributed across all PEs.
    Args:
        tensor (torch.Tensor): The tensor to free.
    Raises:
        ValueError: If the tensor is not a symmetrically distributed tensor.
    """
    torch.cuda.synchronize()
    if not getattr(tensor, "__symm_tensor__", False):
        raise ValueError("tensor is not a symmetrically distributed tensor")
    nvshmem.core.free_tensor(tensor)
    torch.cuda.synchronize()
    
    
### Sync ###

def nvshmem_barrier_all():
    """Barrier all PEs in the default team."""
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD)


def nvshmemx_barrier_all_on_stream(stream: Optional[torch.cuda.Stream] = None):
    """Barrier all PEs in the default team on a specific CUDA stream.
    Args:
        stream (Optional[torch.cuda.Stream]): The CUDA stream to synchronize on. If None, uses the current stream.
    """
    if stream is None:
        stream = torch.cuda.current_stream()
    nvshmem.core.barrier(nvshmem.core.Teams.TEAM_WORLD, stream=create_nvshmem_stream(stream))
    

### Signal ### 

signal_dtype = 'int64'
SIGNAL_DTYPE = torch.int64


def write_i32(
    tensor: torch.Tensor, 
    value: int,
    stream: Optional[torch.cuda.Stream] = None
):
    """Atomic write an int32 value to a tensor.
    Args:
        tensor (torch.Tensor): The tensor to write to, must be of dtype torch.int32.
        value (int): The value to write.
        stream (Optional[torch.cuda.Stream]): The CUDA stream to use for the operation.
            If None, the current stream will be used.
    """
    assert isinstance(tensor, torch.Tensor) and tensor.dtype == torch.int32, \
        f"Input must be a torch.Tensor with dtype torch.int32, but got {tensor.dtype}"
    if stream is None:
        stream = torch.cuda.current_stream()
    (err, ) = cuda.cuStreamWriteValue32(
            stream.cuda_stream,
            tensor.data_ptr(),
            value,
            cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
        )
    CUDA_CHECK(err)


def write_i64(
    tensor: torch.Tensor, 
    value: int,
    stream: Optional[torch.cuda.Stream] = None
):
    """Atomic write an int64 value to a tensor.
    Args:
        tensor (torch.Tensor): The tensor to write to, must be of dtype torch.int64.
        value (int): The value to write.
        stream (Optional[torch.cuda.Stream]): The CUDA stream to use for the operation.
            If None, the current stream will be used.
    """
    assert isinstance(tensor, torch.Tensor) and tensor.dtype == torch.int64, \
        f"Input must be a torch.Tensor with dtype torch.int64, but got {tensor.dtype}"
    if stream is None:
        stream = torch.cuda.current_stream()
    (err, ) = cuda.cuStreamWriteValue64(
            stream.cuda_stream,
            tensor.data_ptr(),
            value,
            cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
        )
    CUDA_CHECK(err)
