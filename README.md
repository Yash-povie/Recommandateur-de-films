# Movie Recommendation System

A hybrid movie recommendation system that combines content-based and collaborative filtering approaches to provide personalized movie recommendations.

## Features

- **Content-Based Filtering**: Recommends movies based on similarity of movie features (genres, etc.)
- **Collaborative Filtering**: Recommends movies based on similar users' preferences
- **Hybrid Approach**: Combines both methods for better recommendations
- **Sentiment Analysis**: Analyze movie sentiment based on user ratings and text reviews
- **Web Interface**: Interactive Streamlit web application
- **GPU Acceleration**: Automatic GPU detection and acceleration for faster similarity calculations (PyTorch)

## Dataset

This system uses the MovieLens dataset. To get started:

1. Download the MovieLens dataset from: https://grouplens.org/datasets/movielens/
2. Recommended: MovieLens Latest Small (25M) or MovieLens Latest (26M)
3. Extract the following files to the `data/` directory:
   - `movies.csv` (or `movies.dat`)
   - `ratings.csv` (or `ratings.dat`)
   - `links.csv` (optional)

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. For GPU support (optional but recommended):
   - Install PyTorch with CUDA support:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```
   - The system will automatically detect and use GPU if available

3. Place your MovieLens dataset files in the `data/ml-25m/` directory (or update the subdirectory path in the app)

4. Run the application:
```bash
streamlit run app.py
```

## Project Structure

```
Movie Recommendation system/
├── data/                    # Dataset files (movies.csv, ratings.csv)
├── src/
│   ├── __init__.py
│   ├── content_based.py     # Content-based filtering implementation
│   ├── collaborative.py    # Collaborative filtering implementation
│   ├── hybrid.py            # Hybrid recommendation engine
│   ├── data_loader.py       # Dataset loading and preprocessing
│   ├── sentiment_analyzer.py # Sentiment analysis module
│   └── gpu_utils.py         # GPU utilities
├── app.py                   # Main web application
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## Usage

### Step 1: Train Models (One-time setup)

First, train and save the models using GPU acceleration:

```bash
python train_models.py
```

This will:
- Load the MovieLens dataset
- Train content-based filtering model
- Train collaborative filtering model
- Create hybrid recommendation engine
- Save all models to the `models/` directory

**Note:** Training takes time (especially on CPU) but only needs to be done once. With GPU, it's much faster!

### Step 2: Run the Application

Once models are trained, start the Streamlit app:

```bash
streamlit run app.py
```

The app will automatically load the pre-trained models and be ready to use instantly!

### Step 3: Get Recommendations

The app has four main features:

1. **Find Similar Movies**: Search for a movie you like → Get similar movie recommendations
2. **User Recommendations**: Select a user ID from the dataset → Get personalized recommendations for that user (useful for testing/demo)
3. **Popular Movies**: Browse the most popular and highly-rated movies
4. **Sentiment Analysis**: Analyze movie sentiment, compare movies, and analyze text reviews

## GPU Support

### Training Phase (One-time)
- **With GPU**: Faster training (~10-15 minutes)
- **Without GPU**: Works fine on CPU (~20-30 minutes)
- GPU is optional but recommended for faster training

### Inference Phase (Runtime)
- **NO GPU REQUIRED** ✅
- Models are pre-trained and saved to disk
- Inference uses CPU only and is very fast (<1 second per recommendation)
- Works perfectly on any computer, including:
  - Laptops without dedicated GPU
  - Low-end computers
  - Cloud instances with CPU only

### Testing GPU Availability

To test GPU availability:
```bash
python test_gpu.py
```

**Note**: GPU is only used during training. Once models are trained, you can use them on any computer without GPU!

