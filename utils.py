import re
from textblob import TextBlob
from collections import Counter
import numpy as np
from typing import List, Dict, Tuple
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def extract_hashtags(caption: str) -> List[str]:
    """Extract hashtags from a caption."""
    return re.findall(r'#\w+', caption)

def extract_mentions(caption: str) -> List[str]:
    """Extract @mentions from a caption."""
    return re.findall(r'@\w+', caption)

def extract_emojis(caption: str) -> List[str]:
    """Extract emojis from a caption."""
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emojis
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.findall(caption)

def analyze_sentiment(caption: str) -> Tuple[float, float]:
    """Analyze sentiment of a caption using TextBlob."""
    analysis = TextBlob(caption)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

def extract_keywords(caption: str, top_n: int = 10) -> List[str]:
    """Extract important keywords from a caption."""
    # Tokenize and lemmatize
    tokens = word_tokenize(caption.lower())
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in lemmatized_tokens 
                      if token.isalpha() and token not in stop_words]
    
    # Count frequencies
    word_freq = Counter(filtered_tokens)
    return [word for word, _ in word_freq.most_common(top_n)]

def extract_features_from_caption(caption: str) -> str:
    """Extract comprehensive features from a caption."""
    # Get basic features
    hashtags = extract_hashtags(caption)
    mentions = extract_mentions(caption)
    emojis = extract_emojis(caption)
    keywords = extract_keywords(caption)
    
    # Analyze sentiment
    sentiment_polarity, sentiment_subjectivity = analyze_sentiment(caption)
    
    # Combine all features
    features = []
    features.extend(hashtags)
    features.extend(mentions)
    features.extend(emojis)
    features.extend(keywords)
    features.append(f"sentiment_{sentiment_polarity:.2f}")
    features.append(f"subjectivity_{sentiment_subjectivity:.2f}")
    
    return ' '.join(features)

def analyze_hashtag_performance(hashtags: List[str], views: List[int]) -> Dict[str, float]:
    """Analyze the performance of hashtags based on views."""
    hashtag_performance = {}
    
    for hashtag, view_count in zip(hashtags, views):
        if hashtag not in hashtag_performance:
            hashtag_performance[hashtag] = []
        hashtag_performance[hashtag].append(view_count)
    
    # Calculate average views for each hashtag
    return {tag: np.mean(views) for tag, views in hashtag_performance.items()}

def suggest_optimal_posting_time(hour_views: List[Tuple[int, int]]) -> int:
    """Suggest optimal posting time based on historical view data."""
    # Group views by hour
    hourly_views = {}
    for hour, views in hour_views:
        if hour not in hourly_views:
            hourly_views[hour] = []
        hourly_views[hour].append(views)
    
    # Calculate average views for each hour
    hourly_averages = {hour: np.mean(views) for hour, views in hourly_views.items()}
    
    # Return hour with highest average views
    return max(hourly_averages.items(), key=lambda x: x[1])[0]

def calculate_engagement_rate(likes: int, comments: int, views: int) -> float:
    """Calculate engagement rate based on likes, comments, and views."""
    if views == 0:
        return 0.0
    return (likes + comments) / views * 100