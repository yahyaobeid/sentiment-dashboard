import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import extract_features_from_caption, analyze_sentiment

class InstagramPredictor:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.scaler = StandardScaler()
        self.tokenizer = Tokenizer(num_words=10000)
        self.caption_model = None
        
    def prepare_features(self, data: pd.DataFrame) -> tuple:
        """Prepare features for training."""
        # Text features
        data['features_text'] = data['caption'].apply(extract_features_from_caption)
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_features=1000)
            X_text = self.vectorizer.fit_transform(data['features_text'])
        else:
            X_text = self.vectorizer.transform(data['features_text'])
            
        # Numerical features
        numerical_features = ['likes', 'comments', 'hour', 'day_of_week']
        X_num = data[numerical_features].values
        
        # Scale numerical features
        if not hasattr(self.scaler, 'mean_'):
            X_num = self.scaler.fit_transform(X_num)
        else:
            X_num = self.scaler.transform(X_num)
            
        # Combine features
        X = np.hstack([X_text.toarray(), X_num])
        return X, data['views']
    
    def train(self, data: pd.DataFrame):
        """Train the model on Instagram data."""
        X, y = self.prepare_features(data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train main model
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Train caption generation model
        self._train_caption_model(data)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        print(f"Training R^2 Score: {train_score}")
        print(f"Testing R^2 Score: {test_score}")
        
    def _train_caption_model(self, data: pd.DataFrame):
        """Train a model for caption generation."""
        # Prepare caption data
        self.tokenizer.fit_on_texts(data['caption'])
        sequences = self.tokenizer.texts_to_sequences(data['caption'])
        padded_sequences = pad_sequences(sequences, maxlen=50)
        
        # Create and train caption model
        self.caption_model = tf.keras.Sequential([
            tf.keras.layers.Embedding(10000, 128),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(10000, activation='softmax')
        ])
        
        self.caption_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Prepare target data
        y_caption = np.array([seq[1:] for seq in sequences])
        X_caption = np.array([seq[:-1] for seq in sequences])
        
        # Train caption model
        self.caption_model.fit(
            X_caption, y_caption,
            epochs=10,
            batch_size=32,
            validation_split=0.2
        )
    
    def predict_views(self, caption: str, likes: int, comments: int, 
                     hour: int, day_of_week: int) -> float:
        """Predict views for a post."""
        # Prepare features
        features_text = extract_features_from_caption(caption)
        X_text = self.vectorizer.transform([features_text])
        
        # Prepare numerical features
        X_num = self.scaler.transform([[likes, comments, hour, day_of_week]])
        
        # Combine features
        X = np.hstack([X_text.toarray(), X_num])
        
        # Make prediction
        return self.model.predict(X)[0]
    
    def generate_caption(self, seed_text: str, max_length: int = 50) -> str:
        """Generate a caption based on seed text."""
        # Tokenize seed text
        seed_sequence = self.tokenizer.texts_to_sequences([seed_text])[0]
        
        # Generate caption
        generated_text = seed_text
        for _ in range(max_length):
            # Prepare input
            padded_sequence = pad_sequences([seed_sequence], maxlen=50)
            
            # Get prediction
            predicted = self.caption_model.predict(padded_sequence, verbose=0)
            predicted_word_idx = np.argmax(predicted[0])
            
            # Convert back to word
            for word, idx in self.tokenizer.word_index.items():
                if idx == predicted_word_idx:
                    generated_text += " " + word
                    break
            
            # Update seed sequence
            seed_sequence.append(predicted_word_idx)
            seed_sequence = seed_sequence[-50:]
            
        return generated_text
    
    def suggest_hashtags(self, caption: str, top_n: int = 5) -> list:
        """Suggest hashtags based on caption content."""
        # Extract keywords from caption
        keywords = extract_features_from_caption(caption).split()
        
        # Get hashtag importance from model
        hashtag_importance = {}
        for word in keywords:
            if word.startswith('#'):
                hashtag_importance[word] = self.model.feature_importances_[
                    self.vectorizer.vocabulary_.get(word, 0)
                ]
        
        # Sort and return top hashtags
        return sorted(hashtag_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def save(self, path: str):
        """Save the model to disk."""
        with open(f'{path}/rf_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open(f'{path}/vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(f'{path}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        self.caption_model.save(f'{path}/caption_model')
        
    def load(self, path: str):
        """Load the model from disk."""
        with open(f'{path}/rf_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        with open(f'{path}/vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(f'{path}/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        self.caption_model = tf.keras.models.load_model(f'{path}/caption_model')

if __name__ == '__main__':
    sample_data = {
        "caption": [
            "Loving the new vibe! #summer #fun",
            "This is a bad day, feeling sad.",
            "What an amazing workout session #fitness #health",
            "Check out this cool soundtrack #music #party",
            "Happy times with family #love #joy"
        ],
        "views": [1500, 200, 3000, 5000, 2500]
    }
    df = pd.DataFrame(sample_data)
    predictor = InstagramPredictor()
    predictor.train(df)

## NEXT STEPS:
## INTEGRATE AI AGENT
## IT SHOULD CREATE CAPTIONS, HASHTAGS, AND SOUNDS FOR VIRAL CONTENT
## IT SHOULD BE TRAINED ON DATA BASED ON THE USERS INSTAGRAM AND ALGORITHM
##
##
##