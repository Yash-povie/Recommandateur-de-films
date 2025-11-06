"""
GPU utility functions for Movie Recommendation System.
Handles GPU detection and GPU-accelerated similarity calculations.
"""

import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def get_device():
    """
    Get the best available device (GPU or CPU).
    
    Returns:
        torch.device: Device object
    """
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            return device
        else:
            print("CUDA not available, using CPU")
            return torch.device('cpu')
    else:
        print("PyTorch not available, using CPU")
        return None


def cosine_similarity_gpu(matrix1, matrix2=None, device=None):
    """
    Calculate cosine similarity on GPU using PyTorch.
    
    Args:
        matrix1: First matrix (numpy array or torch tensor)
        matrix2: Second matrix (optional, if None, computes self-similarity)
        device: torch.device (optional, auto-detected if None)
        
    Returns:
        numpy array: Similarity matrix
    """
    if not TORCH_AVAILABLE:
        # Fallback to CPU
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(matrix1, matrix2)
    
    if device is None:
        device = get_device()
    
    # Convert to torch tensors
    if isinstance(matrix1, np.ndarray):
        tensor1 = torch.from_numpy(matrix1).float().to(device)
    else:
        tensor1 = matrix1.float().to(device)
    
    if matrix2 is not None:
        if isinstance(matrix2, np.ndarray):
            tensor2 = torch.from_numpy(matrix2).float().to(device)
        else:
            tensor2 = matrix2.float().to(device)
    else:
        tensor2 = tensor1
    
    # Normalize vectors
    tensor1_norm = tensor1 / (tensor1.norm(dim=1, keepdim=True) + 1e-8)
    tensor2_norm = tensor2 / (tensor2.norm(dim=1, keepdim=True) + 1e-8)
    
    # Compute cosine similarity
    similarity = torch.mm(tensor1_norm, tensor2_norm.t())
    
    # Convert back to numpy
    if device.type == 'cuda':
        similarity_np = similarity.cpu().numpy()
    else:
        similarity_np = similarity.numpy()
    
    return similarity_np


def to_gpu(array, device=None):
    """Convert numpy array to GPU tensor."""
    if not TORCH_AVAILABLE:
        return array
    
    if device is None:
        device = get_device()
    
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array).float().to(device)
    return array


def from_gpu(tensor, device=None):
    """Convert GPU tensor to numpy array."""
    if not TORCH_AVAILABLE:
        return tensor
    
    if device is None:
        device = get_device()
    
    if isinstance(tensor, torch.Tensor):
        if device.type == 'cuda' and tensor.is_cuda:
            return tensor.cpu().numpy()
        else:
            return tensor.numpy()
    return tensor


def matrix_multiply_gpu(matrix1, matrix2, device=None):
    """Matrix multiplication on GPU."""
    if not TORCH_AVAILABLE:
        return np.dot(matrix1, matrix2)
    
    if device is None:
        device = get_device()
    
    tensor1 = to_gpu(matrix1, device)
    tensor2 = to_gpu(matrix2, device)
    
    result = torch.mm(tensor1, tensor2)
    return from_gpu(result, device)


def is_gpu_available():
    """Check if GPU is available."""
    if TORCH_AVAILABLE:
        return torch.cuda.is_available()
    return False


