from flask import Flask, render_template, jsonify, request
import instagram_client

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True)