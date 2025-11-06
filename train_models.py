"""
Training script to pre-train and save recommendation models.
Run this once to train and save models, then the app will load them instantly.
"""

import os
import sys
from pathlib import Path
import pickle

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.content_based import ContentBasedFiltering
from src.collaborative import CollaborativeFiltering
from src.hybrid import HybridRecommendationEngine
from src.gpu_utils import is_gpu_available, get_device


def train_and_save_models():
    """Train all models and save them to disk."""
    
    print("=" * 60)
    print("Movie Recommendation System - Model Training")
    print("=" * 60)
    
    # Create models directory
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Check GPU
    if is_gpu_available():
        device = get_device()
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0)
            print(f"\nGPU Available: {gpu_name}")
            print(f"CUDA Version: {torch.version.cuda}")
        except:
            print("\nGPU Available")
    else:
        print("\nGPU not available - using CPU")
    
    # Step 1: Load data
    print("\n" + "=" * 60)
    print("Step 1: Loading Dataset")
    print("=" * 60)
    
    loader = DataLoader(data_dir='data')
    try:
        loader.load_data(
            movies_file='movies.csv',
            ratings_file='ratings.csv',
            subdirectory='ml-25m'
        )
        loader.preprocess_movies()
        # Limit to 50k users and 20k movies for memory efficiency
        # Adjust these numbers based on your available RAM
        loader.create_user_item_matrix(min_ratings=5, max_users=50000, max_movies=20000)
        print(f"Loaded {len(loader.movies_df):,} movies")
        print(f"Loaded {len(loader.ratings_df):,} ratings")
        print(f"Created user-item matrix: {loader.user_item_matrix.shape[0]:,} users x {loader.user_item_matrix.shape[1]:,} movies")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the MovieLens dataset files are in the 'data/ml-25m' directory.")
        return False
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Save data loader (for quick access to movies_df, etc.)
    print("\n" + "=" * 60)
    print("Step 2: Saving Data Loader")
    print("=" * 60)
    
    data_loader_path = os.path.join(models_dir, 'data_loader.pkl')
    with open(data_loader_path, 'wb') as f:
        pickle.dump({
            'movies_df': loader.movies_df,
            'movie_features': loader.movie_features,
            'ratings_df': loader.ratings_df,
            'user_ids': loader.user_ids,
            'movie_ids': loader.movie_ids
        }, f)
    print(f"Saved data loader to {data_loader_path}")
    
    # Step 3: Train content-based model
    print("\n" + "=" * 60)
    print("Step 3: Training Content-Based Model")
    print("=" * 60)
    
    print("Building TF-IDF and similarity matrix...")
    content_model = ContentBasedFiltering(loader.movie_features, use_gpu=True)
    content_model.build_model()
    
    # Save content-based model
    content_model_path = os.path.join(models_dir, 'content_based_model.pkl')
    content_model.save_model(content_model_path)
    print(f"Saved content-based model to {content_model_path}")
    
    # Step 4: Train collaborative filtering model
    print("\n" + "=" * 60)
    print("Step 4: Training Collaborative Filtering Model")
    print("=" * 60)
    
    print("Building user-user similarity matrix...")
    collab_model = CollaborativeFiltering(
        loader.ratings_df,
        loader.user_item_matrix,
        loader.user_ids,
        loader.movie_ids,
        use_gpu=True
    )
    collab_model.build_model()
    
    # Save collaborative filtering model
    collab_model_path = os.path.join(models_dir, 'collaborative_model.pkl')
    collab_model.save_model(collab_model_path)
    print(f"Saved collaborative filtering model to {collab_model_path}")
    
    # Step 5: Create and save hybrid engine
    print("\n" + "=" * 60)
    print("Step 5: Creating Hybrid Engine")
    print("=" * 60)
    
    hybrid_engine = HybridRecommendationEngine(
        content_model,
        collab_model,
        loader.movie_features
    )
    
    # Save hybrid engine
    hybrid_engine_path = os.path.join(models_dir, 'hybrid_engine.pkl')
    with open(hybrid_engine_path, 'wb') as f:
        pickle.dump({
            'content_model': content_model,
            'collab_model': collab_model,
            'movies_df': loader.movie_features
        }, f)
    print(f"Saved hybrid engine to {hybrid_engine_path}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nAll models saved to '{models_dir}/' directory")
    print("The app will now load these pre-trained models instantly!")
    print("\nYou can now run: streamlit run app.py")
    
    return True


if __name__ == "__main__":
    success = train_and_save_models()
    if not success:
        sys.exit(1)

