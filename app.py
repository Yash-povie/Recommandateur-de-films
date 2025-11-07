"""
Movie Recommendation System - Streamlit Web Application
Main application file for the hybrid movie recommendation system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.content_based import ContentBasedFiltering
from src.collaborative import CollaborativeFiltering
from src.hybrid import HybridRecommendationEngine
from src.gpu_utils import is_gpu_available, get_device
from src.sentiment_analyzer import MovieSentimentAnalyzer

# Page configuration
st.set_page_config(
    page_title="Recommandateur de films",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .movie-card {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_built' not in st.session_state:
    st.session_state.models_built = False
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = pd.DataFrame(columns=['movieId', 'rating', 'title'])
if 'current_user_id' not in st.session_state:
    st.session_state.current_user_id = None
if 'loading_status' not in st.session_state:
    st.session_state.loading_status = "Initializing..."
if 'loading_started' not in st.session_state:
    st.session_state.loading_started = False


@st.cache_data
def load_data(data_dir='data', movies_file='movies.csv', ratings_file='ratings.csv', subdirectory='ml-25m'):
    """Load and cache data."""
    loader = DataLoader(data_dir)
    try:
        loader.load_data(movies_file, ratings_file, subdirectory=subdirectory)
        loader.preprocess_movies()
        loader.create_user_item_matrix(min_ratings=5)
        return loader
    except FileNotFoundError as e:
        return None
    except Exception as e:
        return None


@st.cache_resource
def build_models(data_loader):
    """Build and cache recommendation models."""
    if data_loader is None:
        return None, None, None
    
    # Build content-based model
    content_model = ContentBasedFiltering(data_loader.movie_features, use_gpu=True)
    content_model.build_model()
    
    # Build collaborative filtering model
    collab_model = CollaborativeFiltering(
        data_loader.ratings_df,
        data_loader.user_item_matrix,
        data_loader.user_ids,
        data_loader.movie_ids,
        use_gpu=True
    )
    collab_model.build_model()
    
    # Create hybrid engine
    hybrid_engine = HybridRecommendationEngine(
        content_model,
        collab_model,
        data_loader.movie_features
    )
    
    return content_model, collab_model, hybrid_engine


@st.cache_resource
def _load_models_cached():
    """Cached function to load models - only runs once."""
    models_dir = 'models'
    
    data_loader_path = os.path.join(models_dir, 'data_loader.pkl')
    content_model_path = os.path.join(models_dir, 'content_based_model.pkl')
    collab_model_path = os.path.join(models_dir, 'collaborative_model.pkl')
    hybrid_engine_path = os.path.join(models_dir, 'hybrid_engine.pkl')
    
    import pickle
    
    # Load all models
    with open(data_loader_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    # Load content-based model
    content_model = ContentBasedFiltering(data_dict['movie_features'], use_gpu=False)
    content_model.load_model(content_model_path)
    
    # Create minimal data loader
    class MinimalDataLoader:
        def __init__(self, data_dict):
            self.movies_df = data_dict['movies_df']
            self.movie_features = data_dict['movie_features']
            self.ratings_df = data_dict['ratings_df']
            self.user_ids = data_dict['user_ids']
            self.movie_ids = data_dict['movie_ids']
        
        def get_movie_by_title(self, title, partial=True):
            """Get movie(s) by title (supports partial matching)."""
            if partial:
                mask = self.movies_df['title'].str.contains(title, case=False, na=False)
            else:
                mask = self.movies_df['title'].str.lower() == title.lower()
            return self.movies_df[mask]
        
        def get_popular_movies(self, top_n=20, min_ratings=50):
            """Get most popular movies by average rating."""
            movie_stats = self.ratings_df.groupby('movieId').agg({
                'rating': ['mean', 'count']
            }).reset_index()
            movie_stats.columns = ['movieId', 'avg_rating', 'num_ratings']
            movie_stats = movie_stats[movie_stats['num_ratings'] >= min_ratings]
            movie_stats = movie_stats.sort_values('avg_rating', ascending=False)
            popular = movie_stats.head(top_n).merge(
                self.movies_df[['movieId', 'title', 'genres']],
                on='movieId',
                how='left'
            )
            return popular
    
    data_loader = MinimalDataLoader(data_dict)
    
    collab_model = CollaborativeFiltering(
        data_loader.ratings_df,
        None,
        data_loader.user_ids,
        data_loader.movie_ids,
        use_gpu=False
    )
    collab_model.load_model(collab_model_path)
    
    # Load hybrid engine
    with open(hybrid_engine_path, 'rb') as f:
        hybrid_dict = pickle.load(f)
    
    hybrid_engine = HybridRecommendationEngine(
        content_model,
        collab_model,
        data_dict['movie_features']
    )
    
    return {
        'data_dict': data_dict,
        'data_loader': data_loader,
        'content_model': content_model,
        'collab_model': collab_model,
        'hybrid_engine': hybrid_engine
    }


def load_pre_trained_models():
    """Load pre-trained models from disk."""
    models_dir = 'models'
    
    # Check if models exist
    data_loader_path = os.path.join(models_dir, 'data_loader.pkl')
    content_model_path = os.path.join(models_dir, 'content_based_model.pkl')
    collab_model_path = os.path.join(models_dir, 'collaborative_model.pkl')
    hybrid_engine_path = os.path.join(models_dir, 'hybrid_engine.pkl')
    
    if not all(os.path.exists(p) for p in [data_loader_path, content_model_path, collab_model_path, hybrid_engine_path]):
        return False, "Models not found. Please run 'python train_models.py' first to train the models."
    
    try:
        # Use cached loading function
        st.session_state.loading_status = "Loading models (5-10 seconds)..."
        models_data = _load_models_cached()
        
        data_dict = models_data['data_dict']
        data_loader = models_data['data_loader']
        
        st.session_state.data_loader = data_loader
        st.session_state.data_loaded = True
        st.session_state.content_model = models_data['content_model']
        st.session_state.collab_model = models_data['collab_model']
        st.session_state.hybrid_engine = models_data['hybrid_engine']
        
        # Initialize sentiment analyzer (lazy loading - will load when needed)
        # Don't initialize here to avoid blocking app startup
        st.session_state.sentiment_analyzer = None
        st.session_state.sentiment_data_ready = True  # Flag that data is ready for sentiment analysis
        
        st.session_state.models_built = True
        st.session_state.loading_status = "Ready!"
        
        return True, "Models loaded successfully!"
        
    except Exception as e:
        return False, f"Error loading models: {str(e)}"


def initialize_system():
    """Initialize the system by loading pre-trained models."""
    if not st.session_state.models_built:
        st.session_state.loading_status = "Loading pre-trained models..."
        success, message = load_pre_trained_models()
        
        if success:
            st.session_state.loading_status = "Ready!"
            return True
        else:
            st.session_state.loading_status = message
            return False
    
    return st.session_state.models_built


def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">🎬 Recommandateur de films</h1>', unsafe_allow_html=True)
    
    # Automatically initialize system (load data and train models)
    if not st.session_state.models_built:
        # Show loading status
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        with status_placeholder.container():
            st.info(f"🔄 {st.session_state.loading_status}")
            st.caption("💡 **Tip:** Model loading takes 10-30 seconds. This is normal for large models!")
        
        # Initialize system (this will load data and train models)
        is_ready = initialize_system()
        
        if not is_ready:
            if "not found" in st.session_state.loading_status.lower() or "error" in st.session_state.loading_status.lower():
                st.error(st.session_state.loading_status)
                st.markdown("""
                ### How to fix:
                1. Run the training script to train and save models:
                   ```bash
                   python train_models.py
                   ```
                2. This will train all models using GPU acceleration and save them to the `models/` directory
                3. Once training is complete, restart this app
                
                **Note:** Training takes time but only needs to be done once!
                """)
            else:
                status_placeholder.info(f"🔄 {st.session_state.loading_status}")
            return
        
        # Clear loading placeholders once ready
        status_placeholder.empty()
        progress_placeholder.empty()
        st.success("✅ System ready! Pre-trained models loaded.")
        # Don't rerun immediately - let user interact
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Status")
        
        # System Status
        st.subheader("📊 System Status")
        if st.session_state.data_loaded:
            st.success("✅ Data loaded")
            if 'data_loader' in st.session_state:
                st.info(f"Movies: {len(st.session_state.data_loader.movies_df):,}")
                st.info(f"Ratings: {len(st.session_state.data_loader.ratings_df):,}")
                st.info(f"Users: {st.session_state.data_loader.ratings_df['userId'].nunique():,}")
        
        if st.session_state.models_built:
            st.success("✅ Models loaded")
            st.info("Ready for recommendations!")
    
    data_loader = st.session_state.data_loader
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎬 Find Similar Movies", "👤 User Recommendations", "📊 Popular Movies", "💭 Sentiment Analysis", "ℹ️ About"])
    
    with tab1:
        st.header("🎬 Find Movies Similar to Your Favorite")
        
        st.markdown("### How it works:")
        st.info("""
        🔍 **Search for a movie you like** → Get recommendations for similar movies based on genres and content!
        """)
        
        st.divider()
        
        if not st.session_state.models_built:
            st.warning("⚠️ Models are still loading. Please wait...")
            return
        
        content_model = st.session_state.content_model
        
        # Movie search
        search_query = st.text_input(
            "🔍 Enter a movie you like:", 
            placeholder="e.g., 'The Matrix', 'Inception', 'Titanic', 'Toy Story'...",
            key="similar_movie_search"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            top_n = st.slider("Number of recommendations", min_value=5, max_value=30, value=10, key="similar_top_n")
        
        if search_query:
            with st.spinner("🔍 Searching for movies..."):
                search_results = data_loader.get_movie_by_title(search_query, partial=True)
                
                if len(search_results) > 0:
                    st.success(f"✅ Found {len(search_results)} movie(s) matching '{search_query}'")
                    
                    # Show search results and let user select one
                    if len(search_results) == 1:
                        selected_movie = search_results.iloc[0]
                        st.info(f"📽️ Selected: **{selected_movie['title']}** ({selected_movie['genres']})")
                    else:
                        st.markdown("### Select a movie:")
                        selected_idx = st.radio(
                            "Choose the movie you want recommendations for:",
                            options=search_results.index,
                            format_func=lambda idx: f"{search_results.loc[idx, 'title']} ({search_results.loc[idx, 'genres']})",
                            key="movie_selection"
                        )
                        selected_movie = search_results.loc[selected_idx]
                    
                    st.divider()
                    
                    # Get similar movies
                    if st.button("🎯 Get Similar Movies", type="primary", use_container_width=True):
                        with st.spinner("🤖 Finding similar movies based on content..."):
                            similar_movies = content_model.get_similar_movies(
                                selected_movie['movieId'], 
                                top_n=top_n
                            )
                            
                            if len(similar_movies) > 0:
                                st.success(f"🎉 Found {len(similar_movies)} similar movies to **{selected_movie['title']}**!")
                                st.divider()
                                
                                for idx, (_, movie) in enumerate(similar_movies.iterrows(), 1):
                                    with st.container():
                                        col1, col2 = st.columns([4, 1])
                                        with col1:
                                            st.markdown(f"### {idx}. {movie['title']}")
                                            st.markdown(f"**Genres:** {movie['genres']}")
                                            # Show similarity score
                                            similarity_pct = movie['similarity_score'] * 100
                                            st.progress(similarity_pct / 100, text=f"Similarity: {similarity_pct:.1f}%")
                                        with col2:
                                            st.metric("Match", f"{movie['similarity_score']:.3f}")
                                        st.divider()
                            else:
                                st.warning("No similar movies found.")
                else:
                    st.warning(f"❌ No movies found matching '{search_query}'. Try a different search term.")
    
    with tab2:
        st.header("👤 Get Recommendations for a User")
        
        if not st.session_state.models_built:
            st.warning("⚠️ Models are still loading. Please wait...")
            return
        
        hybrid_engine = st.session_state.hybrid_engine
        
        st.markdown("### How it works:")
        st.info("""
        👤 **Select a user ID** from the dataset → Get personalized movie recommendations based on that user's preferences!
        """)
        
        st.divider()
        
        # User selection
        if st.session_state.data_loaded:
            user_ids = st.session_state.data_loader.user_ids
            selected_user = st.selectbox(
                "👤 Select User ID:",
                options=list(user_ids[:5000]),  # Limit to first 5000 for performance
                format_func=lambda x: f"User {x}",
                key="user_recommendation_select"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                top_n = st.slider("Number of Recommendations", min_value=5, max_value=50, value=10, key="user_top_n")
            with col2:
                content_weight = st.slider("Content-Based Weight", min_value=0.0, max_value=1.0, value=0.5, step=0.1, key="user_content_weight")
                collaborative_weight = 1.0 - content_weight
                st.caption(f"Collaborative Weight: {collaborative_weight:.1f}")
            
            if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
                with st.spinner("🤖 Generating personalized recommendations..."):
                    # Get recommendations for this user
                    recommendations = hybrid_engine.recommend(
                        user_id=selected_user,
                        user_ratings=None,
                        top_n=top_n,
                        content_weight=content_weight,
                        collaborative_weight=collaborative_weight,
                        min_rating=3.0
                    )
                    
                    if len(recommendations) > 0:
                        st.success(f"🎉 Found {len(recommendations)} recommendations for User {selected_user}!")
                        st.divider()
                        
                        # Display recommendations
                        for idx, (_, movie) in enumerate(recommendations.iterrows(), 1):
                            with st.container():
                                col1, col2 = st.columns([4, 1])
                                with col1:
                                    st.markdown(f"### {idx}. {movie['title']}")
                                    st.markdown(f"**Genres:** {movie['genres']}")
                                    # Show score
                                    score_pct = (movie['total_score'] / recommendations['total_score'].max()) * 100
                                    st.progress(score_pct / 100, text=f"Recommendation Score: {movie['total_score']:.3f}")
                                with col2:
                                    st.metric("Score", f"{movie['total_score']:.3f}")
                                
                                # Show breakdown
                                with st.expander("📊 Why this recommendation?"):
                                    st.write(f"**Content-Based Score:** {movie['content_score']:.3f} (genre similarity)")
                                    st.write(f"**Collaborative Score:** {movie['collaborative_score']:.3f} (similar users liked this)")
                                    st.write(f"**Total Score:** {movie['total_score']:.3f}")
                                
                                st.divider()
                    else:
                        st.warning("No recommendations found for this user.")
    
    with tab3:
        st.header("📊 Popular & Top-Rated Movies")
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Data is still loading. Please wait...")
            return
        
        st.markdown("### Discover the most popular movies in the dataset!")
        
        col1, col2 = st.columns(2)
        with col1:
            top_n = st.slider("Number of movies", min_value=10, max_value=50, value=20, key="popular_top_n")
        with col2:
            min_ratings = st.slider("Minimum ratings required", min_value=10, max_value=500, value=100, step=10, key="popular_min_ratings")
        
        if st.button("📽️ Show Popular Movies", type="primary", use_container_width=True):
            with st.spinner("📊 Loading popular movies..."):
                popular = data_loader.get_popular_movies(top_n=top_n, min_ratings=min_ratings)
                
                if len(popular) > 0:
                    st.success(f"🎬 Showing top {len(popular)} popular movies!")
                    st.divider()
                    
                    for idx, (_, movie) in enumerate(popular.iterrows(), 1):
                        with st.container():
                            col1, col2, col3 = st.columns([4, 1, 1])
                            with col1:
                                st.markdown(f"### {idx}. {movie['title']}")
                                st.markdown(f"**Genres:** {movie['genres']}")
                            with col2:
                                stars = "⭐" * int(movie['avg_rating'])
                                st.metric("Rating", f"{movie['avg_rating']:.2f}/5.0")
                                st.caption(f"{stars}")
                            with col3:
                                st.metric("Votes", f"{int(movie['num_ratings']):,}")
                            st.divider()
                else:
                    st.warning("No popular movies found with the selected criteria.")
    
    with tab4:
        st.header("💭 Sentiment Analysis")
        
        if not st.session_state.models_built:
            st.warning("⚠️ Models are still loading. Please wait...")
        else:
            # Initialize sentiment analyzer on-demand (lazy loading)
            # Use compute_sentiments=False to avoid slow initialization
            if st.session_state.sentiment_analyzer is None and st.session_state.get('sentiment_data_ready', False):
                with st.spinner("Initializing sentiment analyzer (computing sentiment scores - this may take 30-60 seconds)..."):
                    try:
                        # Don't pre-compute all sentiments - compute on demand instead
                        sentiment_analyzer = MovieSentimentAnalyzer(
                            st.session_state.data_loader.ratings_df,
                            st.session_state.data_loader.movies_df,
                            compute_sentiments=False  # Compute on-demand instead of all at once
                        )
                        st.session_state.sentiment_analyzer = sentiment_analyzer
                        st.success("✅ Sentiment analyzer ready!")
                    except Exception as e:
                        st.error(f"❌ Error initializing sentiment analyzer: {str(e)}")
                        st.info("💡 Make sure to install: pip install textblob vaderSentiment")
                        st.session_state.sentiment_analyzer = None
            
            if st.session_state.sentiment_analyzer is None:
                st.warning("⚠️ Sentiment analyzer not initialized. Click below to initialize.")
                if st.button("🔧 Initialize Sentiment Analyzer", type="primary"):
                    st.rerun()
            else:
                sentiment_analyzer = st.session_state.sentiment_analyzer
            
            st.markdown("### How it works:")
            st.info("""
            Analyze sentiment of movies based on user ratings and reviews. 
            Get insights into how audiences feel about different movies!
            """)
            
            st.divider()
            
            # Sentiment analysis options
            analysis_type = st.radio(
                "Choose analysis type:",
                ["🔍 Analyze Single Movie", "📊 Top Sentiment Movies", "📝 Analyze Text Review", "⚖️ Compare Movies"],
                horizontal=True
            )
            
            st.divider()
            
            if analysis_type == "🔍 Analyze Single Movie":
                st.markdown("### Analyze Movie Sentiment")
                
                # Movie search
                movie_search = st.text_input(
                    "🔍 Search for a movie:",
                    placeholder="e.g., 'The Matrix', 'Inception'...",
                    key="sentiment_movie_search"
                )
                
                if movie_search:
                    search_results = data_loader.get_movie_by_title(movie_search, partial=True)
                    
                    if len(search_results) > 0:
                        if len(search_results) == 1:
                            selected_movie = search_results.iloc[0]
                        else:
                            selected_idx = st.selectbox(
                                "Select movie:",
                                options=search_results.index,
                                format_func=lambda idx: f"{search_results.loc[idx, 'title']} ({search_results.loc[idx, 'genres']})",
                                key="sentiment_movie_select"
                            )
                            selected_movie = search_results.loc[selected_idx]
                        
                        if st.button("📊 Analyze Sentiment", type="primary"):
                            sentiment_data = sentiment_analyzer.analyze_movie_sentiment(selected_movie['movieId'])
                            
                            if sentiment_data:
                                st.success(f"✅ Sentiment Analysis for: **{sentiment_data['title']}**")
                                st.divider()
                                
                                # Display sentiment metrics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Sentiment Score", f"{sentiment_data['avg_sentiment']:.3f}")
                                    st.caption(f"Label: {sentiment_data['sentiment_label']}")
                                
                                with col2:
                                    st.metric("Average Rating", f"{sentiment_data['avg_rating']:.2f}/5.0")
                                
                                with col3:
                                    st.metric("Total Ratings", f"{sentiment_data['num_ratings']:,}")
                                
                                with col4:
                                    st.metric("Genres", sentiment_data.get('genres', 'N/A'))
                                
                                st.divider()
                                
                                # Sentiment distribution
                                st.markdown("### 📊 Sentiment Distribution")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Positive", f"{sentiment_data['positive_pct']:.1f}%")
                                    st.progress(sentiment_data['positive_pct'] / 100)
                                
                                with col2:
                                    st.metric("Neutral", f"{sentiment_data['neutral_pct']:.1f}%")
                                    st.progress(sentiment_data['neutral_pct'] / 100)
                                
                                with col3:
                                    st.metric("Negative", f"{sentiment_data['negative_pct']:.1f}%")
                                    st.progress(sentiment_data['negative_pct'] / 100)
                                
                                # Visual representation
                                import matplotlib.pyplot as plt
                                
                                fig, ax = plt.subplots(figsize=(8, 4))
                                categories = ['Positive', 'Neutral', 'Negative']
                                values = [sentiment_data['positive_pct'], sentiment_data['neutral_pct'], sentiment_data['negative_pct']]
                                colors = ['#2ecc71', '#f39c12', '#e74c3c']
                                
                                bars = ax.bar(categories, values, color=colors)
                                ax.set_ylabel('Percentage (%)')
                                ax.set_title(f"Sentiment Distribution: {sentiment_data['title']}")
                                ax.set_ylim(0, 100)
                                
                                # Add value labels on bars
                                for bar, value in zip(bars, values):
                                    height = bar.get_height()
                                    ax.text(bar.get_x() + bar.get_width()/2., height,
                                           f'{value:.1f}%', ha='center', va='bottom')
                                
                                st.pyplot(fig)
                            else:
                                st.warning("No sentiment data available for this movie.")
                    else:
                        st.warning("No movies found matching your search.")
            
            elif analysis_type == "📊 Top Sentiment Movies":
                st.markdown("### Top Movies by Sentiment")
                
                col1, col2 = st.columns(2)
                with col1:
                    sentiment_filter = st.selectbox(
                        "Sentiment Type:",
                        ["Positive", "Negative", "Neutral"],
                        key="top_sentiment_filter"
                    )
                with col2:
                    top_n = st.slider("Number of movies", min_value=5, max_value=30, value=10, key="top_sentiment_n")
                
                if st.button("📊 Get Top Movies", type="primary"):
                    with st.spinner("Analyzing sentiment..."):
                        top_movies = sentiment_analyzer.get_top_sentiment_movies(
                            top_n=top_n,
                            sentiment_type=sentiment_filter.lower()
                        )
                        
                        if len(top_movies) > 0:
                            st.success(f"🎬 Top {len(top_movies)} {sentiment_filter} Sentiment Movies!")
                            st.divider()
                            
                            for idx, (_, movie) in enumerate(top_movies.iterrows(), 1):
                                with st.container():
                                    col1, col2, col3 = st.columns([4, 1, 1])
                                    with col1:
                                        st.markdown(f"### {idx}. {movie['title']}")
                                        st.markdown(f"**Genres:** {movie['genres']}")
                                        st.caption(f"Sentiment: {movie['sentiment_label']}")
                                    with col2:
                                        sentiment_score = movie['avg_sentiment']
                                        st.metric("Sentiment", f"{sentiment_score:.3f}")
                                    with col3:
                                        st.metric("Ratings", f"{int(movie['num_ratings']):,}")
                                    
                                    # Sentiment bar
                                    if sentiment_score > 0:
                                        st.progress(sentiment_score, text=f"Positive: {sentiment_score:.3f}")
                                    else:
                                        st.progress(abs(sentiment_score), text=f"Negative: {sentiment_score:.3f}")
                                    
                                    st.divider()
                        else:
                            st.warning(f"No {sentiment_filter.lower()} sentiment movies found.")
            
            elif analysis_type == "📝 Analyze Text Review":
                st.markdown("### Analyze Text Review Sentiment")
                
                review_text = st.text_area(
                    "Enter a movie review or comment:",
                    placeholder="e.g., 'This movie was absolutely amazing! Great acting and storyline.'",
                    height=100,
                    key="review_text_input"
                )
                
                if st.button("🔍 Analyze Text", type="primary"):
                    if review_text and len(review_text.strip()) > 0:
                        with st.spinner("Analyzing text sentiment..."):
                            text_sentiment = sentiment_analyzer.analyze_text_sentiment(review_text)
                            
                            if text_sentiment:
                                st.success("✅ Text Analysis Complete!")
                                st.divider()
                                
                                # Overall sentiment
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.markdown(f"**Text:** *{text_sentiment['text']}*")
                                with col2:
                                    sentiment_score = text_sentiment['overall_sentiment']
                                    label = text_sentiment['overall_label']
                                    
                                    if sentiment_score > 0:
                                        st.success(f"**{label}** ({sentiment_score:.3f})")
                                    elif sentiment_score < 0:
                                        st.error(f"**{label}** ({sentiment_score:.3f})")
                                    else:
                                        st.info(f"**{label}** ({sentiment_score:.3f})")
                                
                                st.divider()
                                
                                # Detailed analysis
                                if 'vader' in text_sentiment['methods']:
                                    st.markdown("### VADER Sentiment Analysis")
                                    vader = text_sentiment['methods']['vader']
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Compound", f"{vader['compound']:.3f}")
                                    with col2:
                                        st.metric("Positive", f"{vader['positive']:.3f}")
                                    with col3:
                                        st.metric("Neutral", f"{vader['neutral']:.3f}")
                                    with col4:
                                        st.metric("Negative", f"{vader['negative']:.3f}")
                                
                                if 'textblob' in text_sentiment['methods']:
                                    st.markdown("### TextBlob Sentiment Analysis")
                                    textblob = text_sentiment['methods']['textblob']
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Polarity", f"{textblob['polarity']:.3f}")
                                    with col2:
                                        st.metric("Subjectivity", f"{textblob['subjectivity']:.3f}")
                            else:
                                st.warning("Could not analyze text sentiment.")
                    else:
                        st.warning("Please enter some text to analyze.")
            
            elif analysis_type == "⚖️ Compare Movies":
                st.markdown("### Compare Movie Sentiments")
                
                st.info("Search and select up to 5 movies to compare their sentiment scores.")
                
                movie_comparisons = []
                for i in range(5):
                    movie_search = st.text_input(
                        f"Movie {i+1}:",
                        placeholder="Search movie title...",
                        key=f"compare_movie_{i}"
                    )
                    
                    if movie_search:
                        search_results = data_loader.get_movie_by_title(movie_search, partial=True)
                        if len(search_results) > 0:
                            if len(search_results) == 1:
                                selected_movie = search_results.iloc[0]
                            else:
                                selected_idx = st.selectbox(
                                    f"Select Movie {i+1}:",
                                    options=search_results.index,
                                    format_func=lambda idx: f"{search_results.loc[idx, 'title']}",
                                    key=f"compare_select_{i}"
                                )
                                selected_movie = search_results.loc[selected_idx]
                            
                            movie_comparisons.append(selected_movie['movieId'])
                
                if len(movie_comparisons) > 0 and st.button("⚖️ Compare Sentiments", type="primary"):
                    with st.spinner("Comparing movie sentiments..."):
                        comparison_df = sentiment_analyzer.compare_movies_sentiment(movie_comparisons)
                        
                        if len(comparison_df) > 0:
                            st.success(f"✅ Comparing {len(comparison_df)} movies!")
                            st.divider()
                            
                            # Display comparison table
                            st.dataframe(
                                comparison_df[['title', 'avg_sentiment', 'avg_rating', 'num_ratings', 'sentiment_label', 'positive_pct', 'negative_pct']],
                                use_container_width=True
                            )
                            
                            # Visual comparison
                            import matplotlib.pyplot as plt
                            
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                            
                            # Sentiment scores comparison
                            ax1.barh(comparison_df['title'], comparison_df['avg_sentiment'], color='steelblue')
                            ax1.set_xlabel('Sentiment Score')
                            ax1.set_title('Sentiment Score Comparison')
                            ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
                            
                            # Average ratings comparison
                            ax2.barh(comparison_df['title'], comparison_df['avg_rating'], color='orange')
                            ax2.set_xlabel('Average Rating')
                            ax2.set_title('Average Rating Comparison')
                            ax2.set_xlim(0, 5)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.warning("Could not compare movies.")
                elif len(movie_comparisons) == 0:
                    st.info("Search for at least one movie to compare.")
    
    with tab5:
        st.header("ℹ️ About This System")
        
        st.markdown("""
        ### Movie Recommendation System
        
        This is a hybrid movie recommendation system that combines:
        
        - **Content-Based Filtering**: Recommends movies based on similarity of movie features (genres)
        - **Collaborative Filtering**: Recommends movies based on similar users' preferences
        - **Hybrid Approach**: Combines both methods for better recommendations
        
        ### How It Works
        
        1. **Content-Based**: Uses TF-IDF vectorization on movie genres to find similar movies
        2. **Collaborative**: Uses user-user similarity to predict ratings
        3. **Hybrid**: Combines both approaches with configurable weights
        
        ### Dataset Statistics
        """)
        
        if st.session_state.data_loaded:
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            
            with stats_col1:
                st.metric("Total Movies", len(data_loader.movies_df))
            with stats_col2:
                st.metric("Total Ratings", len(data_loader.ratings_df))
            with stats_col3:
                st.metric("Total Users", data_loader.ratings_df['userId'].nunique())
            
            st.divider()
            
            # Genre distribution
            st.subheader("Genre Distribution")
            all_genres = []
            for genres in data_loader.movies_df['genres'].dropna():
                all_genres.extend(genres.split('|'))
            genre_counts = pd.Series(all_genres).value_counts().head(10)
            st.bar_chart(genre_counts)
            
            st.info(f"✅ Dataset loaded: {len(data_loader.movies_df)} movies, {len(data_loader.ratings_df)} ratings")
        
        st.divider()
        st.markdown("""
        ### Dataset
        
        This system uses the MovieLens dataset. Download it from:
        https://grouplens.org/datasets/movielens/
        """)


if __name__ == "__main__":
    main()

