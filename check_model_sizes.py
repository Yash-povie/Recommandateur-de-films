"""
Quick script to check model file sizes and loading time.
"""
import os
import time
import pickle

models_dir = 'models'
files = {
    'data_loader.pkl': 'Data Loader',
    'content_based_model.pkl': 'Content-Based Model',
    'collaborative_model.pkl': 'Collaborative Model',
    'hybrid_engine.pkl': 'Hybrid Engine'
}

print("="*60)
print("Model File Sizes")
print("="*60)

total_size = 0
for filename, name in files.items():
    filepath = os.path.join(models_dir, filename)
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        total_size += size_mb
        print(f"{name:30s}: {size_mb:8.1f} MB")
    else:
        print(f"{name:30s}: NOT FOUND")

print("-"*60)
print(f"{'Total':30s}: {total_size:8.1f} MB")
print("="*60)

# Test loading time
print("\nTesting load times...")
for filename, name in files.items():
    filepath = os.path.join(models_dir, filename)
    if os.path.exists(filepath):
        start = time.time()
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            elapsed = time.time() - start
            print(f"{name:30s}: {elapsed:6.2f} seconds")
        except Exception as e:
            print(f"{name:30s}: ERROR - {e}")


