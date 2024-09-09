import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from loguru import logger
import time
import os
from typing import Tuple

# Initialize the distributed environment
def setup(rank: int, world_size: int) -> None:
    """
    Sets up the distributed environment if GPUs are available.
    
    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes (GPUs).
    """
    if torch.cuda.is_available():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        logger.info(f"Distributed environment initialized on rank {rank} with {world_size} GPUs.")


# Clean up the distributed environment
def cleanup() -> None:
    """
    Cleans up the distributed environment.
    """
    if torch.cuda.is_available():
        dist.destroy_process_group()
        logger.info("Distributed environment cleaned up.")


# Function to generate distributed or regular data
def make_data(shape: Tuple[int, int, int, int], rank: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates Q, K, and V tensors for attention calculations, with random values.
    
    Args:
        shape (Tuple[int, int, int, int]): The shape of the input tensors (B, nh, T, C).
        rank (int): The rank of the current process (for distributed random seed generation).
        device (torch.device): The device to place the tensors on (CPU or GPU).
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The generated Q, K, and V tensors.
    """
    B, nh, T, C = shape
    torch.manual_seed(0 + rank)  # Ensure different random seeds per rank
    Q = torch.randn(B, 1, nh, C).half().to(device)  # Half precision to match JAX's float16
    K = torch.randn(B, T, nh, C).half().to(device)
    V = torch.randn(B, T, nh, C).half().to(device)
    logger.info(f"Generated data on rank {rank} with shape {shape}.")
    
    return Q, K, V


# Function to compute flash_res_lse equivalent
def flash_res_lse(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, softmax_scale: float = 1.0, is_causal: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulates flash attention by calculating attention scores and performing softmax.

    Args:
        q (torch.Tensor): Query tensor (B, nh, 1, C).
        k (torch.Tensor): Key tensor (B, nh, T, C).
        v (torch.Tensor): Value tensor (B, nh, T, C).
        softmax_scale (float): Scaling factor for attention scores.
        is_causal (bool): Whether to apply causal masking (for autoregressive models).
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The attention output and the logsumexp of attention weights.
    """
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale  # q: (B, nh, 1, C), k: (B, nh, T, C)
    if is_causal:
        attn_weights = torch.tril(attn_weights)  # Apply causal mask
    
    attn_weights = F.softmax(attn_weights, dim=-1)  # Normalize with softmax
    res = torch.matmul(attn_weights, v)  # Multiply by values to get the result
    lse = torch.logsumexp(attn_weights, dim=-1)  # Logsumexp for numerical stability
    
    logger.debug(f"Computed attention with shapes Q: {q.shape}, K: {k.shape}, V: {v.shape}.")
    return res, lse

def tree_decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rank: int, world_size: int, device: torch.device) -> torch.Tensor:
    """
    Decodes using attention with optional distributed logic, handling both CPU and multi-GPU setups.

    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        v (torch.Tensor): Value tensor.
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
        device (torch.device): The device to place the tensors on (CPU or GPU).
    
    Returns:
        torch.Tensor: The attention result after global computation.
    """
    loc_res, loc_lse = flash_res_lse(q, k, v)

    # Expand loc_lse to match the shape of loc_res for element-wise operations
    loc_lse = loc_lse.unsqueeze(-1).expand_as(loc_res)
    
    if torch.cuda.is_available() and world_size > 1:
        # Perform global max and sum across GPUs
        a_max_global = loc_lse.clone()
        dist.all_reduce(a_max_global, op=dist.ReduceOp.MAX)
        
        # Adjust shapes to match during multiplication
        num_global = loc_res * torch.exp(loc_lse - a_max_global.unsqueeze(-1).expand_as(loc_res))
        den_global = torch.exp(loc_lse - a_max_global.unsqueeze(-1).expand_as(loc_res))

        dist.all_reduce(num_global, op=dist.ReduceOp.SUM)
        dist.all_reduce(den_global, op=dist.ReduceOp.SUM)
        logger.info(f"Performed distributed reduction on rank {rank}.")
    else:
        # Perform local computations
        a_max_global = loc_lse.max(dim=-1, keepdim=True)[0]
        num_global = loc_res * torch.exp(loc_lse - a_max_global.expand_as(loc_res))
        den_global = torch.exp(loc_lse - a_max_global.expand_as(loc_res))
        logger.info(f"Performed local computation on rank {rank}.")
    
    return num_global / den_global



# Main function to handle both GPU and CPU environments
def main(rank: int, world_size: int) -> None:
    """
    Main function that initializes the environment, generates data, and runs the tree decode function.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes (GPUs or 1 for CPU).
    """
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    setup(rank, world_size)

    seq_len = 64000
    num_heads = 16
    head_dim = 128

    # Generate the Q, K, V data
    qfl, kfl, vfl = make_data((1, num_heads, seq_len, head_dim), rank, device)
    
    logger.info(f"Rank {rank}: Starting computation with seq_len: {seq_len}, hid_dim: {num_heads * head_dim}")
    
    start_time = time.time()
    output = tree_decode(qfl, kfl, vfl, rank, world_size, device)
    end_time = time.time()

    logger.info(f"Rank {rank}: Computation completed in {end_time - start_time}s")
    
    cleanup()


# Entry point for multiprocessing (supports both GPU and CPU)
if __name__ == '__main__':
    logger.add("tree_attention_log.log", rotation="10 MB")  # Log to file with a 10 MB rotation

    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()  # Number of GPUs available
        logger.info(f"Running on {world_size} GPUs.")
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    else:
        # For CPU, run a single process without multiprocessing
        logger.info("Running on CPU.")
        main(rank=0, world_size=1)
