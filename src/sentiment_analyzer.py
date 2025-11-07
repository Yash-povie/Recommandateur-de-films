"""
Sentiment Analysis module for Movie Recommendation System.
Analyzes sentiment of movie reviews and ratings.
"""

import pandas as pd
import numpy as np
from collections import Counter
import re

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False


class MovieSentimentAnalyzer:
    """Sentiment analyzer for movies using reviews and ratings."""
    
    def __init__(self, ratings_df, movies_df, compute_sentiments=True):
        """
        Initialize sentiment analyzer.
        
        Args:
            ratings_df: DataFrame with movie ratings
            movies_df: DataFrame with movie information
            compute_sentiments: Whether to pre-compute all movie sentiments (can be slow for large datasets)
        """
        self.ratings_df = ratings_df.copy()
        self.movies_df = movies_df.copy()
        
        # Initialize sentiment analyzers
        self.vader_analyzer = None
        if VADER_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Pre-compute movie sentiment scores (lazy - compute on demand if disabled)
        if compute_sentiments:
            self.movie_sentiments = self._compute_movie_sentiments()
        else:
            self.movie_sentiments = {}  # Compute on-demand
    
    def _compute_movie_sentiments(self):
        """Pre-compute sentiment scores for all movies based on ratings - OPTIMIZED."""
        movie_sentiments = {}
        
        # Use vectorized operations for much faster computation
        # Group by movieId and compute statistics in one pass
        movie_stats = self.ratings_df.groupby('movieId')['rating'].agg([
            ('avg_rating', 'mean'),
            ('num_ratings', 'count'),
            ('std_rating', 'std')
        ]).reset_index()
        
        # Fill NaN std values with 0
        movie_stats['std_rating'] = movie_stats['std_rating'].fillna(0)
        
        # Calculate sentiment distribution using vectorized operations
        rating_counts = self.ratings_df.groupby('movieId')['rating'].apply(
            lambda x: pd.Series({
                'positive': len(x[x >= 4.0]),
                'negative': len(x[x <= 2.0]),
                'neutral': len(x[(x > 2.0) & (x < 4.0)])
            })
        ).reset_index()
        
        # Merge statistics
        movie_stats = movie_stats.merge(rating_counts, on='movieId', how='left')
        
        # Convert to sentiment scores (vectorized)
        movie_stats['avg_sentiment'] = (movie_stats['avg_rating'] - 3.0) / 2.0
        movie_stats['sentiment_std'] = movie_stats['std_rating'] / 2.0
        
        # Calculate percentages
        movie_stats['positive_pct'] = (movie_stats['positive'] / movie_stats['num_ratings'] * 100).fillna(0)
        movie_stats['negative_pct'] = (movie_stats['negative'] / movie_stats['num_ratings'] * 100).fillna(0)
        movie_stats['neutral_pct'] = (movie_stats['neutral'] / movie_stats['num_ratings'] * 100).fillna(0)
        
        # Get sentiment labels
        movie_stats['sentiment_label'] = movie_stats['avg_sentiment'].apply(self._get_sentiment_label)
        
        # Convert to dictionary format
        for _, row in movie_stats.iterrows():
            movie_sentiments[row['movieId']] = {
                'avg_sentiment': float(row['avg_sentiment']),
                'sentiment_std': float(row['sentiment_std']),
                'num_ratings': int(row['num_ratings']),
                'avg_rating': float(row['avg_rating']),
                'positive_pct': float(row['positive_pct']),
                'negative_pct': float(row['negative_pct']),
                'neutral_pct': float(row['neutral_pct']),
                'sentiment_label': row['sentiment_label']
            }
        
        return movie_sentiments
    
    def _compute_single_movie_sentiment(self, movie_id):
        """Compute sentiment for a single movie on-demand."""
        movie_ratings = self.ratings_df[self.ratings_df['movieId'] == movie_id]['rating']
        
        if len(movie_ratings) > 0:
            normalized_ratings = (movie_ratings - 3.0) / 2.0
            avg_sentiment = normalized_ratings.mean()
            num_ratings = len(movie_ratings)
            
            positive_count = len(movie_ratings[movie_ratings >= 4.0])
            negative_count = len(movie_ratings[movie_ratings <= 2.0])
            neutral_count = len(movie_ratings[(movie_ratings > 2.0) & (movie_ratings < 4.0)])
            
            self.movie_sentiments[movie_id] = {
                'avg_sentiment': float(avg_sentiment),
                'sentiment_std': float(movie_ratings.std() / 2.0),
                'num_ratings': int(num_ratings),
                'avg_rating': float(movie_ratings.mean()),
                'positive_pct': float(positive_count / num_ratings * 100) if num_ratings > 0 else 0,
                'negative_pct': float(negative_count / num_ratings * 100) if num_ratings > 0 else 0,
                'neutral_pct': float(neutral_count / num_ratings * 100) if num_ratings > 0 else 0,
                'sentiment_label': self._get_sentiment_label(avg_sentiment)
            }
    
    def _get_sentiment_label(self, sentiment_score):
        """Convert sentiment score to label."""
        if sentiment_score >= 0.5:
            return "Very Positive"
        elif sentiment_score >= 0.2:
            return "Positive"
        elif sentiment_score >= -0.2:
            return "Neutral"
        elif sentiment_score >= -0.5:
            return "Negative"
        else:
            return "Very Negative"
    
    def analyze_movie_sentiment(self, movie_id):
        """
        Analyze sentiment for a specific movie.
        
        Args:
            movie_id: ID of the movie
            
        Returns:
            Dictionary with sentiment analysis results
        """
        # Compute on-demand if not pre-computed
        if movie_id not in self.movie_sentiments:
            self._compute_single_movie_sentiment(movie_id)
        
        if movie_id not in self.movie_sentiments:
            return None
        
        sentiment_data = self.movie_sentiments[movie_id].copy()
        
        # Get movie info
        movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
        if len(movie_info) > 0:
            sentiment_data['title'] = movie_info.iloc[0]['title']
            sentiment_data['genres'] = movie_info.iloc[0]['genres']
        
        return sentiment_data
    
    def analyze_text_sentiment(self, text):
        """
        Analyze sentiment of a text string.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not text or len(text.strip()) == 0:
            return None
        
        results = {
            'text': text,
            'methods': {}
        }
        
        # VADER Sentiment Analysis (best for social media/text)
        if self.vader_analyzer:
            vader_scores = self.vader_analyzer.polarity_scores(text)
            results['methods']['vader'] = {
                'compound': vader_scores['compound'],
                'positive': vader_scores['pos'],
                'neutral': vader_scores['neu'],
                'negative': vader_scores['neg'],
                'label': self._get_vader_label(vader_scores['compound'])
            }
        
        # TextBlob Sentiment Analysis
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                results['methods']['textblob'] = {
                    'polarity': float(polarity),
                    'subjectivity': float(subjectivity),
                    'label': self._get_textblob_label(polarity)
                }
            except:
                pass
        
        # Overall sentiment (combine methods if available)
        if 'vader' in results['methods']:
            results['overall_sentiment'] = results['methods']['vader']['compound']
            results['overall_label'] = results['methods']['vader']['label']
        elif 'textblob' in results['methods']:
            results['overall_sentiment'] = results['methods']['textblob']['polarity']
            results['overall_label'] = results['methods']['textblob']['label']
        else:
            results['overall_sentiment'] = 0.0
            results['overall_label'] = "Neutral"
        
        return results
    
    def _get_vader_label(self, compound_score):
        """Convert VADER compound score to label."""
        if compound_score >= 0.05:
            return "Positive"
        elif compound_score <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    
    def _get_textblob_label(self, polarity):
        """Convert TextBlob polarity to label."""
        if polarity > 0.1:
            return "Positive"
        elif polarity < -0.1:
            return "Negative"
        else:
            return "Neutral"
    
    def get_top_sentiment_movies(self, top_n=10, sentiment_type='positive'):
        """
        Get top movies by sentiment.
        
        Args:
            top_n: Number of movies to return
            sentiment_type: 'positive', 'negative', or 'neutral'
            
        Returns:
            DataFrame with top movies
        """
        # If sentiments not pre-computed, compute for top-rated movies only
        if len(self.movie_sentiments) == 0:
            # Get top-rated movies and compute their sentiments
            top_movies = self.ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
            top_movies = top_movies[top_movies['count'] >= 10].sort_values('mean', ascending=False).head(100)
            for movie_id in top_movies['movieId']:
                if movie_id not in self.movie_sentiments:
                    self._compute_single_movie_sentiment(movie_id)
        
        sentiment_list = []
        
        for movie_id, sentiment_data in self.movie_sentiments.items():
            if sentiment_type == 'positive' and sentiment_data['avg_sentiment'] > 0:
                sentiment_list.append({
                    'movieId': movie_id,
                    'avg_sentiment': sentiment_data['avg_sentiment'],
                    'num_ratings': sentiment_data['num_ratings'],
                    'avg_rating': sentiment_data['avg_rating'],
                    'sentiment_label': sentiment_data['sentiment_label']
                })
            elif sentiment_type == 'negative' and sentiment_data['avg_sentiment'] < 0:
                sentiment_list.append({
                    'movieId': movie_id,
                    'avg_sentiment': sentiment_data['avg_sentiment'],
                    'num_ratings': sentiment_data['num_ratings'],
                    'avg_rating': sentiment_data['avg_rating'],
                    'sentiment_label': sentiment_data['sentiment_label']
                })
            elif sentiment_type == 'neutral' and -0.2 <= sentiment_data['avg_sentiment'] <= 0.2:
                sentiment_list.append({
                    'movieId': movie_id,
                    'avg_sentiment': sentiment_data['avg_sentiment'],
                    'num_ratings': sentiment_data['num_ratings'],
                    'avg_rating': sentiment_data['avg_rating'],
                    'sentiment_label': sentiment_data['sentiment_label']
                })
        
        if len(sentiment_list) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(sentiment_list)
        
        # Sort by sentiment score
        if sentiment_type == 'positive':
            df = df.sort_values('avg_sentiment', ascending=False)
        elif sentiment_type == 'negative':
            df = df.sort_values('avg_sentiment', ascending=True)
        else:
            df = df.sort_values('num_ratings', ascending=False)
        
        # Merge with movie info
        df = df.merge(
            self.movies_df[['movieId', 'title', 'genres']],
            on='movieId',
            how='left'
        )
        
        return df.head(top_n)
    
    def compare_movies_sentiment(self, movie_ids):
        """
        Compare sentiment of multiple movies.
        
        Args:
            movie_ids: List of movie IDs to compare
            
        Returns:
            DataFrame with comparison
        """
        comparison = []
        
        for movie_id in movie_ids:
            sentiment_data = self.analyze_movie_sentiment(movie_id)
            if sentiment_data:
                comparison.append({
                    'movieId': movie_id,
                    'title': sentiment_data.get('title', 'Unknown'),
                    'avg_sentiment': sentiment_data['avg_sentiment'],
                    'avg_rating': sentiment_data['avg_rating'],
                    'num_ratings': sentiment_data['num_ratings'],
                    'sentiment_label': sentiment_data['sentiment_label'],
                    'positive_pct': sentiment_data['positive_pct'],
                    'negative_pct': sentiment_data['negative_pct']
                })
        
        return pd.DataFrame(comparison)


