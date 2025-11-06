"""
Content-based filtering implementation for Movie Recommendation System.
Uses movie features (genres) to recommend similar movies.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from src.gpu_utils import cosine_similarity_gpu, get_device, is_gpu_available


class ContentBasedFiltering:
    """Content-based recommendation system using movie features."""
    
    def __init__(self, movies_df, use_gpu=True):
        """
        Initialize ContentBasedFiltering.
        
        Args:
            movies_df: DataFrame containing movie information with genres
            use_gpu: Whether to use GPU acceleration if available
        """
        self.movies_df = movies_df.copy()
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.vectorizer = None
        self.use_gpu = use_gpu and is_gpu_available()
        self.device = get_device() if self.use_gpu else None
        self.use_sparse_similarity = False
        
    def build_model(self):
        """Build TF-IDF model and similarity matrix."""
        # Prepare genres data - combine all genres for each movie
        # Replace NaN with empty string
        self.movies_df['genres'] = self.movies_df['genres'].fillna('')
        
        # Initialize TF-IDF Vectorizer
        # Using ngram_range=(1, 2) to capture single and two-word genres
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            min_df=1,  # Changed from 0 to 1 (minimum document frequency)
            stop_words='english'
        )
        
        # Create TF-IDF matrix
        print("Building TF-IDF matrix...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.movies_df['genres'])
        
        # Calculate cosine similarity matrix
        # For large datasets, we'll compute similarities on-demand instead of storing full matrix
        print("Calculating similarity matrix...")
        
        # Check if matrix is too large for full similarity matrix
        num_movies = self.tfidf_matrix.shape[0]
        matrix_size_gb = (num_movies * num_movies * 8) / (1024**3)  # 8 bytes per float64
        
        if matrix_size_gb > 10:  # If larger than 10GB, use sparse/on-demand approach
            print(f"Matrix too large ({matrix_size_gb:.2f} GB), using sparse similarity computation...")
            self.similarity_matrix = None  # Don't store full matrix
            self.use_sparse_similarity = True
        else:
            if self.use_gpu:
                print("Using GPU acceleration for similarity calculation...")
                # Convert sparse matrix to dense for GPU
                tfidf_dense = self.tfidf_matrix.toarray().astype(np.float32)
                similarity_gpu = cosine_similarity_gpu(tfidf_dense, device=self.device)
                # Keep as float32 to save memory
                self.similarity_matrix = similarity_gpu.astype(np.float32)
            else:
                print("Using CPU for similarity calculation...")
                self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix).astype(np.float32)
            self.use_sparse_similarity = False
            print(f"Built similarity matrix: {self.similarity_matrix.shape}")
        
        return self.similarity_matrix
    
    def get_similar_movies(self, movie_id, top_n=10):
        """
        Get top N similar movies to a given movie.
        
        Args:
            movie_id: ID of the movie
            top_n: Number of similar movies to return
            
        Returns:
            DataFrame with similar movies and similarity scores
        """
        if self.tfidf_matrix is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Find movie index
        movie_idx = self.movies_df[self.movies_df['movieId'] == movie_id].index
        
        if len(movie_idx) == 0:
            return pd.DataFrame()
        
        movie_idx = movie_idx[0]
        
        # If using sparse similarity, compute on-demand
        if self.use_sparse_similarity:
            # Get the movie's TF-IDF vector
            movie_vector = self.tfidf_matrix[movie_idx:movie_idx+1]
            
            # Compute cosine similarity with all movies
            if self.use_gpu:
                # Convert to dense for this movie only
                movie_vector_dense = movie_vector.toarray().astype(np.float32)
                all_vectors_dense = self.tfidf_matrix.toarray().astype(np.float32)
                similarity_scores = cosine_similarity_gpu(movie_vector_dense, all_vectors_dense, device=self.device)[0]
            else:
                similarity_scores = cosine_similarity(movie_vector, self.tfidf_matrix).flatten()
        else:
            # Use pre-computed similarity matrix
            similarity_scores = self.similarity_matrix[movie_idx]
        
        # Get top N similar movies (excluding the movie itself)
        top_indices = np.argsort(similarity_scores)[::-1][1:top_n + 1]
        top_scores = similarity_scores[top_indices]
        
        # Get movie information
        similar_movies = self.movies_df.iloc[top_indices].copy()
        similar_movies['similarity_score'] = top_scores
        
        return similar_movies[['movieId', 'title', 'genres', 'similarity_score']]
    
    def recommend_for_user(self, user_ratings, top_n=10, min_rating=3.5):
        """
        Recommend movies for a user based on their rated movies.
        
        Args:
            user_ratings: DataFrame with columns ['movieId', 'rating']
            top_n: Number of recommendations to return
            min_rating: Minimum rating threshold to consider a movie as "liked"
            
        Returns:
            DataFrame with recommended movies and scores
        """
        if self.tfidf_matrix is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Get movies the user liked (rated >= min_rating)
        liked_movies = user_ratings[user_ratings['rating'] >= min_rating]['movieId'].values
        
        if len(liked_movies) == 0:
            # If no liked movies, return empty
            return pd.DataFrame()
        
        # Get movies already rated (to exclude from recommendations)
        rated_movies = set(user_ratings['movieId'].values)
        
        # Calculate recommendation scores for all movies
        recommendation_scores = {}
        
        for liked_movie_id in liked_movies:
            # Get similar movies for this liked movie
            similar = self.get_similar_movies(liked_movie_id, top_n=100)
            
            # Accumulate scores
            for _, row in similar.iterrows():
                movie_id = row['movieId']
                if movie_id not in rated_movies:
                    if movie_id not in recommendation_scores:
                        recommendation_scores[movie_id] = 0
                    recommendation_scores[movie_id] += row['similarity_score']
        
        # Sort by score
        sorted_recommendations = sorted(
            recommendation_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        # Create result DataFrame
        if len(sorted_recommendations) == 0:
            return pd.DataFrame()
        
        movie_ids = [item[0] for item in sorted_recommendations]
        scores = [item[1] for item in sorted_recommendations]
        
        # Get movie information
        recommendations = self.movies_df[
            self.movies_df['movieId'].isin(movie_ids)
        ].copy()
        
        # Add scores
        score_dict = dict(zip(movie_ids, scores))
        recommendations['recommendation_score'] = recommendations['movieId'].map(score_dict)
        
        # Sort by score
        recommendations = recommendations.sort_values(
            'recommendation_score',
            ascending=False
        )
        
        return recommendations[['movieId', 'title', 'genres', 'recommendation_score']]
    
    def save_model(self, filepath='models/content_based_model.pkl'):
        """Save the model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'similarity_matrix': self.similarity_matrix,
            'tfidf_matrix': self.tfidf_matrix,
            'vectorizer': self.vectorizer,
            'movies_df': self.movies_df,
            'use_sparse_similarity': self.use_sparse_similarity,
            'use_gpu': self.use_gpu
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/content_based_model.pkl'):
        """Load the model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.similarity_matrix = model_data.get('similarity_matrix')
        self.tfidf_matrix = model_data.get('tfidf_matrix')
        self.vectorizer = model_data['vectorizer']
        self.movies_df = model_data['movies_df']
        self.use_sparse_similarity = model_data.get('use_sparse_similarity', False)
        self.use_gpu = model_data.get('use_gpu', False)
        if self.use_gpu:
            self.device = get_device()
        
        print(f"Model loaded from {filepath}")

