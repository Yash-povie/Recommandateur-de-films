"""
Test script to check actual loading speed of models.
"""
import time
import pickle
import os

models_dir = 'models'

print("="*60)
print("Testing Model Loading Speed")
print("="*60)

files = {
    'data_loader.pkl': 'Data Loader',
    'content_based_model.pkl': 'Content-Based Model',
    'collaborative_model.pkl': 'Collaborative Model',
    'hybrid_engine.pkl': 'Hybrid Engine'
}

total_start = time.time()

for filename, name in files.items():
    filepath = os.path.join(models_dir, filename)
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"\nLoading {name} ({size_mb:.1f} MB)...")
        start = time.time()
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            elapsed = time.time() - start
            print(f"  ✓ Loaded in {elapsed:.2f} seconds")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n{name}: FILE NOT FOUND")

total_elapsed = time.time() - total_start
print("\n" + "="*60)
print(f"Total loading time: {total_elapsed:.2f} seconds")
print("="*60)


