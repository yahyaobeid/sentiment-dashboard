from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    tweet = data.get("tweet", "")
    sentiment = "neutral"

    return jsonify({"sentiment": sentiment})

if __name__ == '__main__':
    app.run(debug=True)