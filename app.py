from flask import Flask, render_template, jsonify, request
from utils import extract_features_from_caption

import instagram_client, pickle, os


app = Flask(__name__)

model = None
vectorizer = None

if os.path.exists('rf_model.pkl') and os.path.exists('vectorizer.pkl'):
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

def analyze_sentiment(text):
    text_lower = text.lower()
    if "good" in text_lower or "happy" in text_lower:
        return "positive"
    elif "bad" in text_lower or "sad" in text_lower:
        return "negative"
    else:
        return "neutral"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    tweet = data.get("tweet", "")
    sentiment = "neutral"

    return jsonify({"sentiment": sentiment})

@app.route('/fetch_instagram', methods=['GET'])
def fetch_instagram():
    access_token = "INSTA_ACCESS_TOKEN"
    user_id = "INSTAGRAM USER ID"

    posts = instagram_client.fetch_instagram_posts(access_token, user_id, count=5)

    results = []
    for post in posts:
        caption = post.get("caption", "")
        sentiment = analyze_sentiment(caption)
        results.append({
            "id": post.get("id"),
            "caption": caption,
            "sentiment": sentiment,
            "timestamp": post.get("timestamp")
        })

    return jsonify(results)

@app.route('/predict_views', methods=['POST'])
def predict_views():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not available. Please train the model first."}), 500
    
    data = request.get_json()
    caption = data.get("caption", "")
    if not caption:
        return jsonify({"error": "No caption provideed"}), 400
    
    features_text = extract_features_from_caption(caption)
    features_vectorized = vectorizer.transform([features_text])

    predicted_views = model.predict(features_vectorized)[0]
    return jsonify({"predicted_views": predicted_views})

if __name__ == '__main__':
    app.run(debug=True)