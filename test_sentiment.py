"""
Quick test script for sentiment analyzer.
Run this to test if sentiment analysis works before adding to the app.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

try:
    from src.sentiment_analyzer import MovieSentimentAnalyzer
    from src.data_loader import DataLoader
    print("[OK] Successfully imported sentiment analyzer")
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    sys.exit(1)

# Check dependencies
print("\n[CHECK] Checking dependencies...")
try:
    from textblob import TextBlob
    print("[OK] TextBlob available")
except ImportError:
    print("[WARNING] TextBlob not installed - run: pip install textblob")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    print("[OK] VADER Sentiment available")
except ImportError:
    print("[WARNING] VADER Sentiment not installed - run: pip install vaderSentiment")

print("\n" + "="*60)
print("Testing Sentiment Analyzer")
print("="*60)

# Load data
print("\n[LOAD] Loading data...")
try:
    loader = DataLoader(data_dir='data')
    loader.load_data(
        movies_file='movies.csv',
        ratings_file='ratings.csv',
        subdirectory='ml-25m'
    )
    loader.preprocess_movies()
    print(f"[OK] Loaded {len(loader.movies_df)} movies and {len(loader.ratings_df)} ratings")
except Exception as e:
    print(f"[ERROR] Error loading data: {e}")
    print("Make sure the dataset is in data/ml-25m/ directory")
    sys.exit(1)

# Initialize sentiment analyzer
print("\n[INIT] Initializing sentiment analyzer...")
try:
    analyzer = MovieSentimentAnalyzer(loader.ratings_df, loader.movies_df)
    print("[OK] Sentiment analyzer initialized")
    print(f"[OK] Analyzed {len(analyzer.movie_sentiments)} movies")
except Exception as e:
    print(f"[ERROR] Error initializing analyzer: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 1: Analyze a specific movie
print("\n" + "="*60)
print("Test 1: Analyze Single Movie")
print("="*60)

# Find a popular movie
popular_movies = loader.movies_df.head(10)
test_movie = popular_movies.iloc[0]
print(f"\n[TEST] Testing with: {test_movie['title']} (ID: {test_movie['movieId']})")

try:
    sentiment = analyzer.analyze_movie_sentiment(test_movie['movieId'])
    if sentiment:
        print(f"[OK] Sentiment Analysis Results:")
        print(f"   - Title: {sentiment.get('title', 'N/A')}")
        print(f"   - Sentiment Score: {sentiment['avg_sentiment']:.3f}")
        print(f"   - Sentiment Label: {sentiment['sentiment_label']}")
        print(f"   - Average Rating: {sentiment['avg_rating']:.2f}/5.0")
        print(f"   - Total Ratings: {sentiment['num_ratings']:,}")
        print(f"   - Positive: {sentiment['positive_pct']:.1f}%")
        print(f"   - Neutral: {sentiment['neutral_pct']:.1f}%")
        print(f"   - Negative: {sentiment['negative_pct']:.1f}%")
    else:
        print("[WARNING] No sentiment data for this movie")
except Exception as e:
    print(f"[ERROR] Error analyzing movie: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Analyze text
print("\n" + "="*60)
print("Test 2: Analyze Text Review")
print("="*60)

test_texts = [
    "This movie was absolutely amazing! Great acting and storyline.",
    "Terrible movie, waste of time. Boring and poorly acted.",
    "It was okay, nothing special but not bad either."
]

for i, text in enumerate(test_texts, 1):
    print(f"\n[TEST] Test Text {i}: \"{text}\"")
    try:
        text_sentiment = analyzer.analyze_text_sentiment(text)
        if text_sentiment:
            print(f"   [OK] Overall Sentiment: {text_sentiment['overall_label']} ({text_sentiment['overall_sentiment']:.3f})")
            if 'vader' in text_sentiment['methods']:
                vader = text_sentiment['methods']['vader']
                print(f"   - VADER Compound: {vader['compound']:.3f}")
            if 'textblob' in text_sentiment['methods']:
                textblob = text_sentiment['methods']['textblob']
                print(f"   - TextBlob Polarity: {textblob['polarity']:.3f}")
        else:
            print("   [WARNING] Could not analyze text")
    except Exception as e:
        print(f"   [ERROR] Error: {e}")

# Test 3: Top sentiment movies
print("\n" + "="*60)
print("Test 3: Top Positive Sentiment Movies")
print("="*60)

try:
    top_positive = analyzer.get_top_sentiment_movies(top_n=5, sentiment_type='positive')
    if len(top_positive) > 0:
        print(f"[OK] Found {len(top_positive)} positive sentiment movies:")
        for idx, (_, movie) in enumerate(top_positive.iterrows(), 1):
            print(f"   {idx}. {movie['title']} - Sentiment: {movie['avg_sentiment']:.3f}")
    else:
        print("[WARNING] No positive sentiment movies found")
except Exception as e:
    print(f"[ERROR] Error getting top movies: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("[OK] All tests completed!")
print("="*60)
print("\nIf all tests passed, the sentiment analyzer is ready to use!")
print("You can now add it to the app and push to GitHub.")

