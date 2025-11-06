"""
Data loading and preprocessing module for Movie Recommendation System.
Handles loading MovieLens dataset and creating necessary data structures.
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import os


class DataLoader:
    """Class to load and preprocess MovieLens dataset."""
    
    def __init__(self, data_dir='data'):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing the dataset files
        """
        self.data_dir = data_dir
        self.movies_df = None
        self.ratings_df = None
        self.user_item_matrix = None
        self.movie_features = None
        
    def load_data(self, movies_file='movies.csv', ratings_file='ratings.csv', subdirectory=None):
        """
        Load movies and ratings data from CSV files.
        
        Args:
            movies_file: Name of the movies CSV file
            ratings_file: Name of the ratings CSV file
            subdirectory: Subdirectory within data_dir (e.g., 'ml-25m')
        """
        if subdirectory:
            base_path = os.path.join(self.data_dir, subdirectory)
        else:
            base_path = self.data_dir
        
        movies_path = os.path.join(base_path, movies_file)
        ratings_path = os.path.join(base_path, ratings_file)
        
        # Check if files exist
        if not os.path.exists(movies_path):
            raise FileNotFoundError(f"Movies file not found: {movies_path}")
        if not os.path.exists(ratings_path):
            raise FileNotFoundError(f"Ratings file not found: {ratings_path}")
        
        # Load movies data
        print(f"Loading movies from {movies_path}...")
        self.movies_df = pd.read_csv(movies_path)
        
        # Handle different MovieLens formats
        if 'movieId' not in self.movies_df.columns:
            # Try .dat format
            if movies_file.endswith('.dat'):
                self.movies_df = pd.read_csv(
                    movies_path,
                    sep='::',
                    engine='python',
                    names=['movieId', 'title', 'genres'],
                    encoding='latin-1'
                )
            else:
                raise ValueError("Movies file must contain 'movieId' column")
        
        # Load ratings data
        print(f"Loading ratings from {ratings_path}...")
        self.ratings_df = pd.read_csv(ratings_path)
        
        # Handle different MovieLens formats
        if 'userId' not in self.ratings_df.columns:
            # Try .dat format
            if ratings_file.endswith('.dat'):
                self.ratings_df = pd.read_csv(
                    ratings_path,
                    sep='::',
                    engine='python',
                    names=['userId', 'movieId', 'rating', 'timestamp'],
                    encoding='latin-1'
                )
        
        print(f"Loaded {len(self.movies_df)} movies and {len(self.ratings_df)} ratings")
        
    def preprocess_movies(self):
        """Preprocess movies data and extract features."""
        if self.movies_df is None:
            raise ValueError("Movies data not loaded. Call load_data() first.")
        
        # Create a copy to avoid modifying original
        movies = self.movies_df.copy()
        
        # Extract year from title if present (format: "Title (Year)")
        movies['year'] = movies['title'].str.extract(r'\((\d{4})\)', expand=False)
        movies['year'] = pd.to_numeric(movies['year'], errors='coerce')
        
        # Extract clean title (without year)
        movies['title_clean'] = movies['title'].str.replace(r'\s*\(\d{4}\)\s*', '', regex=True)
        
        # Process genres - split by | and create genre list
        movies['genres_list'] = movies['genres'].str.split('|')
        
        # Get all unique genres
        all_genres = set()
        for genres in movies['genres_list'].dropna():
            all_genres.update(genres)
        
        # Create binary features for each genre
        for genre in all_genres:
            movies[f'genre_{genre}'] = movies['genres'].str.contains(genre, na=False).astype(int)
        
        self.movie_features = movies
        print(f"Preprocessed {len(movies)} movies with {len(all_genres)} genres")
        
        return movies
    
    def create_user_item_matrix(self, min_ratings=5, max_users=None, max_movies=None):
        """
        Create user-item rating matrix using sparse matrix for memory efficiency.
        
        Args:
            min_ratings: Minimum number of ratings per user to include
            max_users: Maximum number of users to include (for memory efficiency)
            max_movies: Maximum number of movies to include (for memory efficiency)
        """
        if self.ratings_df is None:
            raise ValueError("Ratings data not loaded. Call load_data() first.")
        
        # Filter users with minimum ratings
        user_rating_counts = self.ratings_df['userId'].value_counts()
        valid_users = user_rating_counts[user_rating_counts >= min_ratings].index
        
        # Limit users if specified
        if max_users and len(valid_users) > max_users:
            # Take top users by rating count
            top_users = user_rating_counts[user_rating_counts >= min_ratings].head(max_users).index
            valid_users = top_users
            print(f"Limiting to top {max_users} users for memory efficiency")
        
        filtered_ratings = self.ratings_df[self.ratings_df['userId'].isin(valid_users)].copy()
        
        print(f"Filtering to {len(valid_users)} users with at least {min_ratings} ratings")
        
        # Get unique movie IDs
        unique_movies = filtered_ratings['movieId'].unique()
        
        # Limit movies if specified
        if max_movies and len(unique_movies) > max_movies:
            # Get most rated movies
            movie_counts = filtered_ratings['movieId'].value_counts()
            top_movies = movie_counts.head(max_movies).index
            unique_movies = top_movies
            filtered_ratings = filtered_ratings[filtered_ratings['movieId'].isin(unique_movies)]
            print(f"Limiting to top {max_movies} movies for memory efficiency")
        
        # Create mappings
        user_to_idx = {user_id: idx for idx, user_id in enumerate(sorted(valid_users))}
        movie_to_idx = {movie_id: idx for idx, movie_id in enumerate(sorted(unique_movies))}
        
        # Create sparse matrix directly (much more memory efficient)
        rows = [user_to_idx[user_id] for user_id in filtered_ratings['userId']]
        cols = [movie_to_idx[movie_id] for movie_id in filtered_ratings['movieId']]
        values = filtered_ratings['rating'].values.astype(np.float32)  # Use float32 to save memory
        
        # Create sparse matrix
        self.user_item_matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(len(valid_users), len(unique_movies)),
            dtype=np.float32
        )
        
        # Store user and movie mappings
        self.user_ids = np.array(sorted(valid_users))
        self.movie_ids = np.array(sorted(unique_movies))
        
        print(f"Created sparse user-item matrix: {self.user_item_matrix.shape[0]} users x {self.user_item_matrix.shape[1]} movies")
        print(f"Matrix density: {self.user_item_matrix.nnz / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]):.4%}")
        
        return self.user_item_matrix
    
    def get_movie_by_id(self, movie_id):
        """Get movie information by movie ID."""
        if self.movies_df is None:
            raise ValueError("Movies data not loaded.")
        
        movie = self.movies_df[self.movies_df['movieId'] == movie_id]
        if len(movie) == 0:
            return None
        return movie.iloc[0].to_dict()
    
    def get_movie_by_title(self, title, partial=True):
        """Get movie(s) by title (supports partial matching)."""
        if self.movies_df is None:
            raise ValueError("Movies data not loaded.")
        
        if partial:
            mask = self.movies_df['title'].str.contains(title, case=False, na=False)
        else:
            mask = self.movies_df['title'].str.lower() == title.lower()
        
        return self.movies_df[mask]
    
    def get_user_ratings(self, user_id):
        """Get all ratings for a specific user."""
        if self.ratings_df is None:
            raise ValueError("Ratings data not loaded.")
        
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        return user_ratings.merge(
            self.movies_df[['movieId', 'title', 'genres']],
            on='movieId',
            how='left'
        )
    
    def get_popular_movies(self, top_n=20, min_ratings=50):
        """Get most popular movies by average rating."""
        if self.ratings_df is None or self.movies_df is None:
            raise ValueError("Data not loaded.")
        
        # Calculate average ratings
        movie_stats = self.ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        
        movie_stats.columns = ['movieId', 'avg_rating', 'num_ratings']
        
        # Filter by minimum ratings
        movie_stats = movie_stats[movie_stats['num_ratings'] >= min_ratings]
        
        # Sort by average rating
        movie_stats = movie_stats.sort_values('avg_rating', ascending=False)
        
        # Merge with movie info
        popular = movie_stats.head(top_n).merge(
            self.movies_df[['movieId', 'title', 'genres']],
            on='movieId',
            how='left'
        )
        
        return popular

