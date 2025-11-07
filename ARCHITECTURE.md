# Movie Recommendation System - Architecture & Usage Guide

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Installation & Setup](#installation--setup)
4. [Usage Guide](#usage-guide)
5. [Deployment & GPU Requirements](#deployment--gpu-requirements)
6. [Model Details](#model-details)
7. [File Structure](#file-structure)

---

## 🎯 System Overview

This is a **Hybrid Movie Recommendation System** that combines:
- **Content-Based Filtering**: Recommends movies based on genre similarity
- **Collaborative Filtering**: Recommends movies based on similar users' preferences
- **Hybrid Approach**: Combines both methods for optimal recommendations
- **Sentiment Analysis**: Analyzes movie sentiment from ratings and text reviews

The system uses the **MovieLens 25M dataset** with 62,423 movies and 25 million ratings from 162,541 users.

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Web Application                  │
│                         (app.py)                             │
└───────────────────────┬───────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
│   Content-   │ │Collaborative│ │   Hybrid    │ │ Sentiment   │
│   Based      │ │ Filtering   │ │   Engine    │ │  Analyzer   │
│   Model      │ │   Model     │ │             │ │             │
└───────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
        │               │               │
        └───────────────┼───────────────┘
                        │
                ┌───────▼────────┐
                │  Pre-trained   │
                │  Model Files  │
                │  (.pkl files) │
                └───────────────┘
```

### Data Flow

1. **Training Phase** (One-time):
   - Load MovieLens dataset
   - Preprocess movies and ratings
   - Train Content-Based model (TF-IDF + Cosine Similarity)
   - Train Collaborative Filtering model (User-User Similarity)
   - Save all models to disk

2. **Inference Phase** (Runtime):
   - Load pre-trained models from disk
   - User searches for a movie or selects a user ID
   - System generates recommendations using loaded models
   - Display results to user

---

## 📦 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- 8GB+ RAM recommended
- **GPU is NOT required for inference** (only for training)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `scikit-learn>=1.3.0`
- `streamlit>=1.28.0`
- `torch>=2.0.0` (optional, for GPU training)

### Step 2: Download Dataset

1. Download MovieLens 25M dataset from: https://grouplens.org/datasets/movielens/
2. Extract the dataset
3. Place the `ml-25m` folder in the `data/` directory

Your directory structure should look like:
```
Movie Recommendation system/
├── data/
│   └── ml-25m/
│       ├── movies.csv
│       ├── ratings.csv
│       └── ...
├── models/          (created after training)
├── src/
├── app.py
├── train_models.py
└── requirements.txt
```

### Step 3: Train Models (One-time)

```bash
python train_models.py
```

This will:
- Load and preprocess the dataset
- Train Content-Based model
- Train Collaborative Filtering model
- Save all models to `models/` directory

**Training Time**: ~10-30 minutes depending on hardware
- With GPU: Faster (uses GPU acceleration)
- Without GPU: Slower but works fine (uses CPU)

### Step 4: Run the Application

```bash
streamlit run app.py
```

Or if streamlit is not in PATH:
```bash
python -m streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🚀 Usage Guide

### Tab 1: Find Similar Movies

**Purpose**: Get movie recommendations based on a movie you like

**Steps**:
1. Enter a movie title in the search box (e.g., "The Matrix", "Inception")
2. Select the correct movie if multiple results appear
3. Click "Get Similar Movies"
4. View recommendations with similarity scores

**How it works**: Uses Content-Based Filtering to find movies with similar genres

### Tab 2: User Recommendations

**Purpose**: Get personalized recommendations for a specific user from the dataset

**Steps**:
1. Select a User ID from the dropdown
2. Adjust settings (number of recommendations, content/collaborative weights)
3. Click "Get Recommendations"
4. View personalized movie recommendations

**How it works**: Uses Hybrid approach combining Content-Based and Collaborative Filtering

### Tab 3: Popular Movies

**Purpose**: Browse the most popular and highly-rated movies

**Steps**:
1. Adjust filters (number of movies, minimum ratings)
2. Click "Show Popular Movies"
3. Browse top-rated movies with ratings and vote counts

### Tab 4: About

**Purpose**: View system information and dataset statistics

---

## 💻 Deployment & GPU Requirements

### ✅ Can I use this on a computer with weak or no GPU?

**YES! Absolutely!**

The system is designed to work on **any computer**, regardless of GPU:

#### Training Phase (One-time)
- **With GPU**: Faster training (~10-15 minutes)
- **Without GPU**: Slower but works fine (~20-30 minutes on CPU)
- **Weak GPU**: Will automatically fall back to CPU if GPU memory is insufficient

#### Inference Phase (Runtime)
- **NO GPU REQUIRED** ✅
- Models are **pre-trained** and saved to disk
- Inference uses **CPU only** and is very fast
- Works perfectly on:
  - Laptops without dedicated GPU
  - Low-end computers
  - Cloud instances with CPU only
  - Raspberry Pi (with sufficient RAM)

### Model Files

After training, the following files are saved in `models/`:
- `content_based_model.pkl` - Content-based filtering model
- `collaborative_model.pkl` - Collaborative filtering model
- `hybrid_engine.pkl` - Hybrid recommendation engine
- `data_loader.pkl` - Preprocessed data

**Total size**: ~500MB - 2GB (depending on dataset size)

### Deployment Options

1. **Local Machine**: Just run `streamlit run app.py`
2. **Cloud Deployment**: 
   - Deploy to Streamlit Cloud (free)
   - Deploy to Heroku, AWS, etc.
   - No GPU required for deployment
3. **Docker**: Create a Docker container (CPU-only is fine)

### Performance

- **Model Loading**: ~5-10 seconds (one-time on app start)
- **Recommendation Generation**: <1 second per request
- **Memory Usage**: ~2-4GB RAM during inference
- **CPU Usage**: Low to moderate (single-threaded operations)

---

## 🔬 Model Details

### 1. Content-Based Filtering

**Algorithm**: TF-IDF Vectorization + Cosine Similarity

**Process**:
1. Extract genres from each movie
2. Create TF-IDF vectors for all movies
3. Compute cosine similarity matrix
4. For a given movie, find top-N most similar movies

**Advantages**:
- Works for new movies (no user ratings needed)
- Explains recommendations (genre-based)
- Fast inference

**Limitations**:
- Only considers genre similarity
- May recommend too similar movies

### 2. Collaborative Filtering

**Algorithm**: User-User Similarity + Rating Prediction

**Process**:
1. Create user-item rating matrix
2. Compute user-user similarity using cosine similarity
3. For a user, find similar users
4. Predict ratings for unrated movies
5. Recommend top-rated predictions

**Advantages**:
- Considers user preferences
- Can discover unexpected recommendations
- Learns from user behavior

**Limitations**:
- Cold start problem (new users/movies)
- Requires user rating history

### 3. Hybrid Approach

**Combination**: Weighted combination of both methods

**Formula**:
```
Final Score = (Content_Weight × Content_Score) + (Collaborative_Weight × Collaborative_Score)
```

**Benefits**:
- Combines strengths of both methods
- Handles cold start better
- More accurate recommendations

---

## 📁 File Structure

```
Movie Recommendation system/
│
├── app.py                          # Main Streamlit application
├── train_models.py                 # Training script (run once)
├── requirements.txt                 # Python dependencies
├── ARCHITECTURE.md                  # This file
├── README.md                        # Project README
│
├── data/                            # Dataset directory
│   └── ml-25m/
│       ├── movies.csv              # Movie metadata
│       ├── ratings.csv             # User ratings
│       └── README.txt               # Dataset info
│
├── models/                          # Pre-trained models (created after training)
│   ├── content_based_model.pkl     # Content-based model
│   ├── collaborative_model.pkl     # Collaborative filtering model
│   ├── hybrid_engine.pkl           # Hybrid engine
│   └── data_loader.pkl              # Preprocessed data
│
└── src/                             # Source code
    ├── __init__.py
    ├── data_loader.py              # Data loading and preprocessing
    ├── content_based.py             # Content-based filtering
    ├── collaborative.py             # Collaborative filtering
    ├── hybrid.py                    # Hybrid recommendation engine
    └── gpu_utils.py                 # GPU utilities (optional)
```

---

## 🔧 Configuration

### Training Configuration

Edit `train_models.py` to adjust:
- `max_users`: Limit number of users (default: 50,000)
- `max_movies`: Limit number of movies (default: 20,000)
- `min_ratings`: Minimum ratings per user (default: 5)
- `chunk_size`: Memory optimization for large datasets (default: 500)

### Application Configuration

Edit `app.py` to adjust:
- Number of recommendations shown
- Default weights for hybrid approach
- UI customization

---

## ❓ FAQ

**Q: Do I need to retrain models every time?**
A: No! Models are trained once and saved. Just load them when running the app.

**Q: Can I use a smaller dataset?**
A: Yes! The system works with any MovieLens dataset (1M, 10M, 25M). Just place it in `data/` directory.

**Q: How do I update recommendations?**
A: Retrain the models with `python train_models.py` to incorporate new data.

**Q: Can I add my own movies?**
A: Yes, but you'll need to retrain the models with your custom dataset.

**Q: Why are recommendations slow?**
A: First-time loading takes 5-10 seconds. Subsequent recommendations are <1 second.

---

## 📝 Notes

- Models are trained on a subset of data (50K users, 20K movies) for memory efficiency
- Full dataset can be used by adjusting parameters in `train_models.py`
- GPU is only used during training, not during inference
- All inference operations use CPU and are optimized for speed

---

## 📧 Support

For issues or questions:
1. Check this documentation
2. Review error messages
3. Ensure all dependencies are installed
4. Verify dataset is in correct location

---

**Last Updated**: 2024
**Version**: 1.0


