"""
Example usage script for the Movie Recommendation System.
This demonstrates how to use the system programmatically.
"""

import pandas as pd
from src.data_loader import DataLoader
from src.content_based import ContentBasedFiltering
from src.collaborative import CollaborativeFiltering
from src.hybrid import HybridRecommendationEngine


def main():
    """Example usage of the recommendation system."""
    
    print("=" * 60)
    print("Movie Recommendation System - Example Usage")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n1. Loading data...")
    loader = DataLoader(data_dir='data')
    
    try:
        loader.load_data(movies_file='movies.csv', ratings_file='ratings.csv', subdirectory='ml-25m')
        loader.preprocess_movies()
        loader.create_user_item_matrix(min_ratings=5)
        print("✓ Data loaded successfully!")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("Please ensure the MovieLens dataset files are in the 'data/ml-25m' directory.")
        return
    
    # Step 2: Build content-based model
    print("\n2. Building content-based model...")
    content_model = ContentBasedFiltering(loader.movie_features, use_gpu=True)
    content_model.build_model()
    print("✓ Content-based model built!")
    
    # Step 3: Build collaborative filtering model
    print("\n3. Building collaborative filtering model...")
    collab_model = CollaborativeFiltering(
        loader.ratings_df,
        loader.user_item_matrix,
        loader.user_ids,
        loader.movie_ids,
        use_gpu=True
    )
    collab_model.build_model()
    print("✓ Collaborative filtering model built!")
    
    # Step 4: Create hybrid engine
    print("\n4. Creating hybrid recommendation engine...")
    hybrid_engine = HybridRecommendationEngine(
        content_model,
        collab_model,
        loader.movie_features
    )
    print("✓ Hybrid engine created!")
    
    # Step 5: Example 1 - Content-based recommendations for a user
    print("\n" + "=" * 60)
    print("Example 1: Content-Based Recommendations")
    print("=" * 60)
    
    # Simulate user ratings
    sample_movie_ids = loader.movies_df['movieId'].head(5).values
    user_ratings = pd.DataFrame({
        'movieId': sample_movie_ids,
        'rating': [4.5, 4.0, 5.0, 3.5, 4.5]
    })
    
    print(f"\nUser has rated {len(user_ratings)} movies:")
    for _, row in user_ratings.iterrows():
        movie = loader.get_movie_by_id(row['movieId'])
        if movie:
            print(f"  - {movie['title']}: {row['rating']}/5.0")
    
    print("\nGetting content-based recommendations...")
    content_recs = content_model.recommend_for_user(user_ratings, top_n=5)
    
    if len(content_recs) > 0:
        print("\nTop 5 Content-Based Recommendations:")
        for idx, row in content_recs.iterrows():
            print(f"  {idx + 1}. {row['title']} (Score: {row['recommendation_score']:.3f})")
            print(f"     Genres: {row['genres']}")
    else:
        print("No recommendations found.")
    
    # Step 6: Example 2 - Collaborative filtering recommendations
    print("\n" + "=" * 60)
    print("Example 2: Collaborative Filtering Recommendations")
    print("=" * 60)
    
    # Use an existing user
    if len(loader.user_ids) > 0:
        sample_user_id = loader.user_ids[0]
        print(f"\nUsing user ID: {sample_user_id}")
        
        print("\nGetting collaborative filtering recommendations...")
        collab_recs = collab_model.recommend_for_user_fast(sample_user_id, top_n=5)
        
        if len(collab_recs) > 0:
            print("\nTop 5 Collaborative Filtering Recommendations:")
            for idx, row in collab_recs.iterrows():
                movie = loader.get_movie_by_id(row['movieId'])
                if movie:
                    print(f"  {idx + 1}. {movie['title']} (Predicted Rating: {row['predicted_rating']:.2f}/5.0)")
                    print(f"     Genres: {movie['genres']}")
        else:
            print("No recommendations found.")
    
    # Step 7: Example 3 - Hybrid recommendations
    print("\n" + "=" * 60)
    print("Example 3: Hybrid Recommendations")
    print("=" * 60)
    
    print("\nGetting hybrid recommendations...")
    hybrid_recs = hybrid_engine.recommend(
        user_id=sample_user_id if len(loader.user_ids) > 0 else None,
        user_ratings=user_ratings,
        top_n=5,
        content_weight=0.4,
        collaborative_weight=0.6
    )
    
    if len(hybrid_recs) > 0:
        print("\nTop 5 Hybrid Recommendations:")
        for idx, row in hybrid_recs.iterrows():
            print(f"  {idx + 1}. {row['title']} (Total Score: {row['total_score']:.3f})")
            print(f"     Genres: {row['genres']}")
            print(f"     Content Score: {row['content_score']:.3f}, Collaborative Score: {row['collaborative_score']:.3f}")
    else:
        print("No recommendations found.")
    
    # Step 8: Example 4 - Find similar movies
    print("\n" + "=" * 60)
    print("Example 4: Find Similar Movies")
    print("=" * 60)
    
    # Get a sample movie
    sample_movie = loader.movies_df.iloc[0]
    print(f"\nFinding movies similar to: {sample_movie['title']}")
    print(f"Genres: {sample_movie['genres']}")
    
    similar_movies = content_model.get_similar_movies(sample_movie['movieId'], top_n=5)
    
    if len(similar_movies) > 0:
        print("\nTop 5 Similar Movies:")
        for idx, row in similar_movies.iterrows():
            print(f"  {idx + 1}. {row['title']} (Similarity: {row['similarity_score']:.3f})")
            print(f"     Genres: {row['genres']}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

