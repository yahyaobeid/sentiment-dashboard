import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
from utils import extract_features_from_caption

def train_model(data):
    data['features_text'] = data['caption'].apply(extract_features_from_caption)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['features_text'])
    y = data['views']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"Model R^2 Score: {score}")

    with open('rf_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

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
    train_model(df)

## NEXT STEPS:
## INTEGRATE AI AGENT
## IT SHOULD CREATE CAPTIONS, HASHTAGS, AND SOUNDS FOR VIRAL CONTENT
## IT SHOULD BE TRAINED ON DATA BASED ON THE USERS INSTAGRAM AND ALGORITHM
##
##
##