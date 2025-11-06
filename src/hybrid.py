"""
Hybrid recommendation engine combining content-based and collaborative filtering.
"""

import pandas as pd
import numpy as np
from src.content_based import ContentBasedFiltering
from src.collaborative import CollaborativeFiltering


class HybridRecommendationEngine:
    """Hybrid recommendation system combining multiple approaches."""
    
    def __init__(self, content_based_model, collaborative_model, movies_df):
        """
        Initialize HybridRecommendationEngine.
        
        Args:
            content_based_model: ContentBasedFiltering instance
            collaborative_model: CollaborativeFiltering instance
            movies_df: DataFrame with movie information
        """
        self.content_based = content_based_model
        self.collaborative = collaborative_model
        self.movies_df = movies_df
        
    def recommend(
        self,
        user_id=None,
        user_ratings=None,
        top_n=10,
        content_weight=0.4,
        collaborative_weight=0.6,
        min_rating=3.0
    ):
        """
        Get hybrid recommendations for a user.
        
        Args:
            user_id: User ID (for collaborative filtering)
            user_ratings: DataFrame with columns ['movieId', 'rating'] (for content-based)
            top_n: Number of recommendations to return
            content_weight: Weight for content-based recommendations (0-1)
            collaborative_weight: Weight for collaborative recommendations (0-1)
            min_rating: Minimum rating threshold
            
        Returns:
            DataFrame with recommended movies and scores
        """
        # Normalize weights
        total_weight = content_weight + collaborative_weight
        if total_weight > 0:
            content_weight = content_weight / total_weight
            collaborative_weight = collaborative_weight / total_weight
        else:
            content_weight = 0.5
            collaborative_weight = 0.5
        
        recommendations = {}
        
        # Get content-based recommendations
        if user_ratings is not None and len(user_ratings) > 0:
            try:
                content_recs = self.content_based.recommend_for_user(
                    user_ratings,
                    top_n=top_n * 2,  # Get more to have better options
                    min_rating=min_rating
                )
                
                if len(content_recs) > 0:
                    # Normalize content-based scores (0-1 range)
                    max_score = content_recs['recommendation_score'].max()
                    if max_score > 0:
                        for _, row in content_recs.iterrows():
                            movie_id = row['movieId']
                            score = (row['recommendation_score'] / max_score) * content_weight
                            recommendations[movie_id] = {
                                'movieId': movie_id,
                                'content_score': score,
                                'collaborative_score': 0,
                                'total_score': score
                            }
            except Exception as e:
                print(f"Error in content-based recommendations: {e}")
        
        # Get collaborative filtering recommendations
        if user_id is not None:
            try:
                collab_recs = self.collaborative.recommend_for_user_fast(
                    user_id,
                    top_n=top_n * 2,  # Get more to have better options
                    min_rating=min_rating
                )
                
                if len(collab_recs) > 0:
                    # Normalize collaborative scores (0-1 range)
                    max_rating = collab_recs['predicted_rating'].max()
                    min_rating_val = collab_recs['predicted_rating'].min()
                    rating_range = max_rating - min_rating_val if max_rating > min_rating_val else 1
                    
                    for _, row in collab_recs.iterrows():
                        movie_id = row['movieId']
                        # Normalize predicted rating to 0-1
                        normalized_score = (
                            (row['predicted_rating'] - min_rating_val) / rating_range
                        ) * collaborative_weight
                        
                        if movie_id in recommendations:
                            recommendations[movie_id]['collaborative_score'] = normalized_score
                            recommendations[movie_id]['total_score'] += normalized_score
                        else:
                            recommendations[movie_id] = {
                                'movieId': movie_id,
                                'content_score': 0,
                                'collaborative_score': normalized_score,
                                'total_score': normalized_score
                            }
            except Exception as e:
                print(f"Error in collaborative recommendations: {e}")
        
        if len(recommendations) == 0:
            return pd.DataFrame()
        
        # Convert to DataFrame
        recs_df = pd.DataFrame(list(recommendations.values()))
        
        # Sort by total score
        recs_df = recs_df.sort_values('total_score', ascending=False)
        
        # Get top N
        recs_df = recs_df.head(top_n)
        
        # Merge with movie information
        recs_df = recs_df.merge(
            self.movies_df[['movieId', 'title', 'genres']],
            on='movieId',
            how='left'
        )
        
        # Reorder columns
        recs_df = recs_df[[
            'movieId', 'title', 'genres',
            'content_score', 'collaborative_score', 'total_score'
        ]]
        
        return recs_df
    
    def recommend_for_new_user(self, user_ratings, top_n=10):
        """
        Recommend for a new user (cold start problem).
        Uses only content-based filtering since collaborative needs user history.
        
        Args:
            user_ratings: DataFrame with columns ['movieId', 'rating']
            top_n: Number of recommendations to return
            
        Returns:
            DataFrame with recommended movies
        """
        if user_ratings is None or len(user_ratings) == 0:
            # Return popular movies if no ratings
            return self.movies_df.head(top_n)
        
        return self.content_based.recommend_for_user(
            user_ratings,
            top_n=top_n,
            min_rating=3.5
        )
    
    def get_recommendation_explanation(self, movie_id, user_id=None, user_ratings=None):
        """
        Get explanation for why a movie was recommended.
        
        Args:
            movie_id: ID of the movie
            user_id: User ID (optional)
            user_ratings: User ratings DataFrame (optional)
            
        Returns:
            Dictionary with explanation details
        """
        explanation = {
            'movie_id': movie_id,
            'reasons': []
        }
        
        # Get movie info
        movie = self.movies_df[self.movies_df['movieId'] == movie_id]
        if len(movie) == 0:
            return explanation
        
        movie = movie.iloc[0]
        explanation['movie_title'] = movie['title']
        explanation['movie_genres'] = movie['genres']
        
        # Content-based explanation
        if user_ratings is not None and len(user_ratings) > 0:
            liked_movies = user_ratings[user_ratings['rating'] >= 3.5]['movieId'].values
            for liked_id in liked_movies[:3]:  # Check top 3 liked movies
                similar = self.content_based.get_similar_movies(liked_id, top_n=20)
                if movie_id in similar['movieId'].values:
                    liked_movie = self.movies_df[self.movies_df['movieId'] == liked_id].iloc[0]
                    explanation['reasons'].append(
                        f"Similar to '{liked_movie['title']}' which you rated highly"
                    )
        
        # Collaborative explanation
        if user_id is not None:
            try:
                similar_users = self.collaborative.get_similar_users(user_id, top_n=5)
                for similar_user_id, similarity in similar_users[:3]:
                    user_ratings_check = self.collaborative.ratings_df[
                        self.collaborative.ratings_df['userId'] == similar_user_id
                    ]
                    if movie_id in user_ratings_check['movieId'].values:
                        rating = user_ratings_check[
                            user_ratings_check['movieId'] == movie_id
                        ]['rating'].iloc[0]
                        explanation['reasons'].append(
                            f"Users similar to you rated this {rating:.1f}/5.0"
                        )
            except:
                pass
        
        return explanation


