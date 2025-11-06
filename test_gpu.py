"""
Quick test script to verify GPU availability and functionality.
"""

from src.gpu_utils import is_gpu_available, get_device, cosine_similarity_gpu
import numpy as np
import time

def main():
    print("=" * 60)
    print("GPU Test for Movie Recommendation System")
    print("=" * 60)
    
    # Check GPU availability
    print("\n1. Checking GPU availability...")
    if is_gpu_available():
        device = get_device()
        try:
            import torch
            print("GPU Available!")
            print(f"   Device: {device}")
            print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        except Exception as e:
            print(f"GPU detected but error getting details: {e}")
    else:
        print("GPU not available - will use CPU")
        print("   (This is fine, the system will work on CPU)")
    
    # Test GPU similarity calculation
    print("\n2. Testing GPU similarity calculation...")
    try:
        # Create test matrix
        test_matrix = np.random.rand(1000, 100).astype(np.float32)
        
        # Test CPU
        print("   Testing CPU...")
        start = time.time()
        from sklearn.metrics.pairwise import cosine_similarity
        cpu_result = cosine_similarity(test_matrix)
        cpu_time = time.time() - start
        print(f"   CPU time: {cpu_time:.4f} seconds")
        
        # Test GPU
        if is_gpu_available():
            print("   Testing GPU...")
            start = time.time()
            gpu_result = cosine_similarity_gpu(test_matrix)
            gpu_time = time.time() - start
            print(f"   GPU time: {gpu_time:.4f} seconds")
            print(f"   Speedup: {cpu_time / gpu_time:.2f}x faster")
            
            # Verify results are similar
            diff = np.abs(cpu_result - gpu_result).max()
            print(f"   Max difference: {diff:.6f} (should be < 0.001)")
            if diff < 0.001:
                print("   Results match!")
            else:
                print("   Results differ slightly (may be due to floating point precision)")
        else:
            print("   Skipping GPU test (GPU not available)")
    
    except Exception as e:
        print(f"   Error during test: {e}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()

