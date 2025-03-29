# Instagram Sentiment Dashboard

A powerful dashboard for analyzing Instagram content and optimizing for maximum engagement. This tool helps content creators understand what drives high views and engagement on their posts by analyzing various factors including captions, hashtags, posting times, and more.

## Features

- **Content Analysis**: Analyze your Instagram posts for sentiment and engagement potential
- **View Prediction**: Predict potential views based on content and metadata
- **Hashtag Suggestions**: Get AI-powered hashtag recommendations
- **Caption Generation**: Generate engaging captions based on your content
- **Sentiment Analysis**: Understand the emotional tone of your content
- **Model Training**: Train the model on your specific niche or target audience

## Prerequisites

- Python 3.8 or higher
- Instagram Graph API access token
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/instagram-sentiment-dashboard.git
cd instagram-sentiment-dashboard
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables:
Create a `.env` file in the project root and add your Instagram access token:
```
INSTAGRAM_ACCESS_TOKEN=your_access_token_here
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Use the dashboard to:
   - Train the model on your target hashtags
   - Analyze your content
   - Get predictions and suggestions
   - Generate optimized captions

## How It Works

The dashboard uses machine learning to analyze various aspects of Instagram content:

1. **Data Collection**: Gathers data from Instagram posts including:
   - Captions
   - Hashtags
   - Engagement metrics (likes, comments, views)
   - Posting times
   - User interactions

2. **Feature Analysis**: Processes the data to extract meaningful features:
   - Text sentiment
   - Hashtag effectiveness
   - Optimal posting times
   - Content patterns

3. **Model Training**: Uses the collected data to train models for:
   - View prediction
   - Caption generation
   - Hashtag recommendation

4. **Recommendations**: Provides actionable insights to improve content performance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Instagram Graph API
- TensorFlow
- scikit-learn
- TextBlob
- NLTK

