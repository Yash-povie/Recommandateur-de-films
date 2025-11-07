"""
Collaborative filtering implementation for Movie Recommendation System.
Uses user-based collaborative filtering to recommend movies.
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from src.gpu_utils import cosine_similarity_gpu, get_device, is_gpu_available


class CollaborativeFiltering:
    """User-based collaborative filtering recommendation system."""
    
    def __init__(self, ratings_df, user_item_matrix=None, user_ids=None, movie_ids=None, use_gpu=True):
        """
        Initialize CollaborativeFiltering.
        
        Args:
            ratings_df: DataFrame with columns ['userId', 'movieId', 'rating']
            user_item_matrix: Sparse user-item matrix (optional)
            user_ids: Array of user IDs corresponding to matrix rows
            movie_ids: Array of movie IDs corresponding to matrix columns
            use_gpu: Whether to use GPU acceleration if available
        """
        self.ratings_df = ratings_df.copy()
        self.user_item_matrix = user_item_matrix
        self.user_ids = user_ids
        self.movie_ids = movie_ids
        self.user_similarity_matrix = None
        self.user_means = None
        self.use_gpu = use_gpu and is_gpu_available()
        self.device = get_device() if self.use_gpu else None
        
    def build_model(self, min_similarity=0.0, chunk_size=1000, top_k_similar=100):
        """
        Build user-user similarity matrix using chunked computation for memory efficiency.
        
        Args:
            min_similarity: Minimum similarity threshold to store
            chunk_size: Size of chunks for batch processing
            top_k_similar: Number of top similar users to store per user (for memory efficiency)
        """
        if self.user_item_matrix is None:
            raise ValueError("User-item matrix not provided. Cannot build model.")
        
        print("Building user-user similarity matrix...")
        
        num_users = self.user_item_matrix.shape[0]
        matrix_size_gb = (num_users * num_users * 8) / (1024**3)  # 8 bytes per float64
        
        # If matrix is too large, use chunked computation with sparse storage
        if matrix_size_gb > 5.0:  # If larger than 5GB, use chunked approach
            print(f"Matrix too large ({matrix_size_gb:.2f} GB), using chunked computation with sparse storage...")
            self._build_model_chunked(chunk_size=chunk_size, top_k_similar=top_k_similar, min_similarity=min_similarity)
        else:
            # Calculate user-user similarity using cosine similarity
            # Use sparse matrix directly for memory efficiency
            if self.use_gpu:
                print("Using GPU acceleration for similarity calculation...")
                # For GPU, we need to convert to dense, but only if matrix is small enough
                matrix_size = self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]
                if matrix_size > 50_000_000:  # If matrix is too large (>50M elements)
                    print("Matrix too large for GPU, using CPU with sparse matrices...")
                    self.user_similarity_matrix = cosine_similarity(self.user_item_matrix)
                else:
                    user_item_dense = self.user_item_matrix.toarray().astype(np.float32)
                    self.user_similarity_matrix = cosine_similarity_gpu(user_item_dense, device=self.device)
            else:
                print("Using CPU for similarity calculation...")
                # Use sparse matrix directly (much more memory efficient)
                self.user_similarity_matrix = cosine_similarity(self.user_item_matrix)
        
        # Calculate mean rating for each user (for rating prediction)
        # Use sparse matrix mean calculation
        self.user_means = np.array(self.user_item_matrix.mean(axis=1)).flatten()
        
        if hasattr(self, 'user_similarity_matrix') and self.user_similarity_matrix is not None:
            print(f"Built similarity matrix: {self.user_similarity_matrix.shape}")
        else:
            print(f"Built sparse similarity storage for {num_users} users")
        
        return self.user_similarity_matrix
    
    def _build_model_chunked(self, chunk_size=500, top_k_similar=100, min_similarity=0.0):
        """
        Build similarity matrix using chunked computation to save memory.
        Stores only top-k similar users per user in a sparse format.
        """
        from scipy.sparse import csr_matrix, lil_matrix, dok_matrix
        from sklearn.metrics.pairwise import cosine_similarity
        
        num_users = self.user_item_matrix.shape[0]
        print(f"Computing similarities in chunks of {chunk_size} users...")
        print(f"Storing top {top_k_similar} similar users per user...")
        
        # Use DOK format for efficient incremental building
        # This is more memory efficient than LIL for sparse matrices
        similarity_dict = {}
        
        # Process in smaller chunks to reduce memory usage
        for i in range(0, num_users, chunk_size):
            end_i = min(i + chunk_size, num_users)
            chunk_i = self.user_item_matrix[i:end_i]
            
            chunk_num = i//chunk_size + 1
            total_chunks = (num_users + chunk_size - 1)//chunk_size
            print(f"Processing chunk {chunk_num}/{total_chunks} (users {i} to {end_i-1})...")
            
            # Compute similarity of this chunk with all users
            # This creates a chunk_size x num_users matrix
            chunk_similarity = cosine_similarity(chunk_i, self.user_item_matrix)
            
            # For each user in chunk, store only top-k similar users
            for local_idx, global_idx in enumerate(range(i, end_i)):
                similarities = chunk_similarity[local_idx]
                
                # Get top-k similar users (excluding self)
                top_indices = np.argsort(similarities)[::-1]
                # Remove self-similarity (should be 1.0)
                top_indices = top_indices[top_indices != global_idx][:top_k_similar]
                
                # Store only similarities above threshold
                for similar_idx in top_indices:
                    sim_score = float(similarities[similar_idx])
                    if sim_score >= min_similarity:
                        similarity_dict[(global_idx, similar_idx)] = sim_score
                
                # Also store symmetric similarity (if not already stored)
                for similar_idx in top_indices:
                    sim_score = float(similarities[similar_idx])
                    if sim_score >= min_similarity:
                        if (similar_idx, global_idx) not in similarity_dict:
                            similarity_dict[(similar_idx, global_idx)] = sim_score
            
            # Clear chunk_similarity to free memory
            del chunk_similarity
        
        # Convert dictionary to sparse matrix
        print("Converting to sparse matrix format...")
        if len(similarity_dict) > 0:
            rows, cols = zip(*similarity_dict.keys())
            data = list(similarity_dict.values())
            self.user_similarity_matrix = csr_matrix(
                (data, (rows, cols)),
                shape=(num_users, num_users),
                dtype=np.float32
            )
        else:
            # Create empty sparse matrix
            self.user_similarity_matrix = csr_matrix((num_users, num_users), dtype=np.float32)
        
        print(f"Chunked computation complete! Stored {len(similarity_dict)} similarity pairs.")
    
    def get_similar_users(self, user_id, top_n=10):
        """
        Get top N similar users to a given user.
        
        Args:
            user_id: ID of the user
            top_n: Number of similar users to return
            
        Returns:
            Array of (user_id, similarity_score) tuples
        """
        if self.user_similarity_matrix is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Find user index
        user_idx = np.where(self.user_ids == user_id)[0]
        
        if len(user_idx) == 0:
            return []
        
        user_idx = user_idx[0]
        
        # Get similarity scores for this user
        similarity_scores = self.user_similarity_matrix[user_idx]
        
        # Get top N similar users (excluding the user itself)
        top_indices = np.argsort(similarity_scores)[::-1][1:top_n + 1]
        
        # Create list of (user_id, similarity_score)
        similar_users = [
            (self.user_ids[idx], similarity_scores[idx])
            for idx in top_indices
            if similarity_scores[idx] > 0
        ]
        
        return similar_users
    
    def predict_rating(self, user_id, movie_id, top_k=50):
        """
        Predict rating for a user-movie pair.
        
        Args:
            user_id: ID of the user
            movie_id: ID of the movie
            top_k: Number of similar users to consider
            
        Returns:
            Predicted rating (float)
        """
        if self.user_similarity_matrix is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Find user and movie indices
        user_idx = np.where(self.user_ids == user_id)[0]
        movie_idx = np.where(self.movie_ids == movie_id)[0]
        
        if len(user_idx) == 0 or len(movie_idx) == 0:
            # Cold start: return average rating
            return self.ratings_df['rating'].mean()
        
        user_idx = user_idx[0]
        movie_idx = movie_idx[0]
        
        # Get user's mean rating
        user_mean = self.user_means[user_idx]
        
        # Get similarity scores for this user
        user_similarities = self.user_similarity_matrix[user_idx]
        
        # Get ratings for this movie from all users
        movie_ratings = self.user_item_matrix[:, movie_idx].toarray().flatten()
        
        # Find users who rated this movie
        rated_mask = movie_ratings > 0
        
        if not rated_mask.any():
            # No one rated this movie - return user's mean
            return user_mean
        
        # Get similarities and ratings for users who rated the movie
        similarities = user_similarities[rated_mask]
        ratings = movie_ratings[rated_mask]
        other_means = self.user_means[rated_mask]
        
        # Weight similarities (only positive similarities)
        positive_mask = similarities > 0
        if not positive_mask.any():
            return user_mean
        
        similarities = similarities[positive_mask]
        ratings = ratings[positive_mask]
        other_means = other_means[positive_mask]
        
        # Get top K similar users
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        similarities = similarities[top_indices]
        ratings = ratings[top_indices]
        other_means = other_means[top_indices]
        
        # Calculate weighted average
        # Rating prediction: user_mean + weighted sum of (rating - other_mean)
        numerator = np.sum(similarities * (ratings - other_means))
        denominator = np.sum(np.abs(similarities))
        
        if denominator == 0:
            return user_mean
        
        predicted_rating = user_mean + (numerator / denominator)
        
        # Clamp to valid rating range (typically 0.5 to 5.0)
        predicted_rating = max(0.5, min(5.0, predicted_rating))
        
        return predicted_rating
    
    def recommend_for_user(self, user_id, top_n=10, min_rating=3.0):
        """
        Recommend movies for a user.
        
        Args:
            user_id: ID of the user
            top_n: Number of recommendations to return
            min_rating: Minimum predicted rating threshold
            
        Returns:
            DataFrame with recommended movies and predicted ratings
        """
        if self.user_similarity_matrix is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Get movies already rated by user
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        rated_movie_ids = set(user_ratings['movieId'].values)
        
        # Get all movies
        all_movie_ids = set(self.movie_ids)
        
        # Get unrated movies
        unrated_movies = all_movie_ids - rated_movie_ids
        
        if len(unrated_movies) == 0:
            return pd.DataFrame()
        
        # Predict ratings for unrated movies
        predictions = []
        for movie_id in unrated_movies:
            predicted_rating = self.predict_rating(user_id, movie_id)
            if predicted_rating >= min_rating:
                predictions.append({
                    'movieId': movie_id,
                    'predicted_rating': predicted_rating
                })
        
        if len(predictions) == 0:
            return pd.DataFrame()
        
        # Create DataFrame and sort by predicted rating
        recommendations_df = pd.DataFrame(predictions)
        recommendations_df = recommendations_df.sort_values(
            'predicted_rating',
            ascending=False
        ).head(top_n)
        
        return recommendations_df
    
    def recommend_for_user_fast(self, user_id, top_n=10, min_rating=3.0):
        """
        Faster recommendation using matrix operations.
        
        Args:
            user_id: ID of the user
            top_n: Number of recommendations to return
            min_rating: Minimum predicted rating threshold
            
        Returns:
            DataFrame with recommended movies and predicted ratings
        """
        if self.user_similarity_matrix is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Find user index
        user_idx = np.where(self.user_ids == user_id)[0]
        if len(user_idx) == 0:
            return pd.DataFrame()
        user_idx = user_idx[0]
        
        # Get user's mean rating
        user_mean = self.user_means[user_idx]
        
        # Get similarity scores for this user
        user_similarities = self.user_similarity_matrix[user_idx]
        
        # Get movies already rated by user
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        rated_movie_ids = set(user_ratings['movieId'].values)
        
        # Convert user-item matrix to dense for this calculation
        user_item_dense = self.user_item_matrix.toarray()
        
        # Calculate predictions for all movies
        predictions = []
        
        for movie_idx, movie_id in enumerate(self.movie_ids):
            if movie_id in rated_movie_ids:
                continue
            
            # Get ratings for this movie
            movie_ratings = user_item_dense[:, movie_idx]
            
            # Find users who rated this movie
            rated_mask = movie_ratings > 0
            
            if not rated_mask.any():
                continue
            
            # Get similarities and ratings
            similarities = user_similarities[rated_mask]
            ratings = movie_ratings[rated_mask]
            other_means = self.user_means[rated_mask]
            
            # Only consider positive similarities
            positive_mask = similarities > 0
            if not positive_mask.any():
                continue
            
            similarities = similarities[positive_mask]
            ratings = ratings[positive_mask]
            other_means = other_means[positive_mask]
            
            # Calculate weighted average
            numerator = np.sum(similarities * (ratings - other_means))
            denominator = np.sum(np.abs(similarities))
            
            if denominator == 0:
                continue
            
            predicted_rating = user_mean + (numerator / denominator)
            predicted_rating = max(0.5, min(5.0, predicted_rating))
            
            if predicted_rating >= min_rating:
                predictions.append({
                    'movieId': movie_id,
                    'predicted_rating': predicted_rating
                })
        
        if len(predictions) == 0:
            return pd.DataFrame()
        
        # Create DataFrame and sort
        recommendations_df = pd.DataFrame(predictions)
        recommendations_df = recommendations_df.sort_values(
            'predicted_rating',
            ascending=False
        ).head(top_n)
        
        return recommendations_df
    
    def save_model(self, filepath='models/collaborative_model.pkl'):
        """Save the model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'user_similarity_matrix': self.user_similarity_matrix,
            'user_means': self.user_means,
            'user_ids': self.user_ids,
            'movie_ids': self.movie_ids,
            'ratings_df': self.ratings_df
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/collaborative_model.pkl'):
        """Load the model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.user_similarity_matrix = model_data['user_similarity_matrix']
        self.user_means = model_data['user_means']
        self.user_ids = model_data['user_ids']
        self.movie_ids = model_data['movie_ids']
        self.ratings_df = model_data['ratings_df']
        
        # Model loaded silently for faster startup

