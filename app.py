from flask import Flask, render_template, request, jsonify
from instagram_client import InstagramClient
from model import InstagramPredictor
from utils import extract_features_from_caption, analyze_sentiment
import os
from datetime import datetime
import json

app = Flask(__name__)

# Initialize Instagram client and predictor
INSTAGRAM_ACCESS_TOKEN = os.getenv('INSTAGRAM_ACCESS_TOKEN')
instagram_client = InstagramClient(INSTAGRAM_ACCESS_TOKEN)
predictor = InstagramPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_post():
    data = request.json
    caption = data.get('caption', '')
    likes = int(data.get('likes', 0))
    comments = int(data.get('comments', 0))
    hour = datetime.now().hour
    day_of_week = datetime.now().weekday()
    
    # Get predictions and suggestions
    predicted_views = predictor.predict_views(caption, likes, comments, hour, day_of_week)
    suggested_hashtags = predictor.suggest_hashtags(caption)
    generated_caption = predictor.generate_caption(caption[:50])
    
    # Analyze sentiment
    sentiment_polarity, sentiment_subjectivity = analyze_sentiment(caption)
    
    return jsonify({
        'predicted_views': int(predicted_views),
        'suggested_hashtags': [tag for tag, _ in suggested_hashtags],
        'generated_caption': generated_caption,
        'sentiment': {
            'polarity': sentiment_polarity,
            'subjectivity': sentiment_subjectivity
        }
    })

@app.route('/train', methods=['POST'])
def train_model():
    data = request.json
    hashtags = data.get('hashtags', [])
    limit = int(data.get('limit', 100))
    
    # Collect training data
    training_data = instagram_client.collect_training_data(hashtags, limit)
    
    # Train the model
    predictor.train(training_data)
    
    return jsonify({
        'status': 'success',
        'message': f'Model trained on {len(training_data)} posts'
    })

@app.route('/insights', methods=['GET'])
def get_insights():
    # Get user insights
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'User ID is required'}), 400
        
    insights = instagram_client.get_user_insights(user_id)
    return jsonify(insights)

if __name__ == '__main__':
    app.run(debug=True)