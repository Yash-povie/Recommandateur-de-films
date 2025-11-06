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
        # Load data loader
        import pickle
        with open(data_loader_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        # Create a minimal data loader object
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
        st.session_state.data_loader = data_loader
        st.session_state.data_loaded = True
        
        # Load content-based model
        # Use movie_features from the data_dict
        content_model = ContentBasedFiltering(data_dict['movie_features'], use_gpu=False)
        content_model.load_model(content_model_path)
        
        # Load collaborative filtering model
        collab_model = CollaborativeFiltering(
            data_loader.ratings_df,
            None,  # user_item_matrix not needed for loaded model
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
        
        st.session_state.content_model = content_model
        st.session_state.collab_model = collab_model
        st.session_state.hybrid_engine = hybrid_engine
        st.session_state.models_built = True
        
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
        st.rerun()
    
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
    tab1, tab2, tab3, tab4 = st.tabs(["🎬 Find Similar Movies", "👤 User Recommendations", "📊 Popular Movies", "ℹ️ About"])
    
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

