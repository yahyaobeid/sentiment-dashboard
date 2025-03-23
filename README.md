# High-Level Architecture for Real-Time Sentiment Dashboard

## Overview
The goal of this project is to build a lightweight web dashboard that:
- Pulls real-time social media data (e.g., tweets)
- Performs sentiment analysis on the incoming data
- Displays the sentiment data in real time on a web dashboard

## Components

### 1. Data Ingestion
- **Source:** Instagram Graph API
- **Responsibility:** Poll the Instagram API for recent posts (or comments) from an Instagram Business account.


### 2. Sentiment Analysis
- **Library:** TensorFlow or scikit-learn
- **Responsibility:** Process and analyze tweet text to classify sentiment as positive, neutral, or negative.

### 3. Backend Service
- **Framework:** Flask (or FastAPI)
- **Responsibility:**
  - Serve API endpoints for processing and retrieving data.
  - Optionally push updates to the frontend via WebSockets (using Flask-SocketIO).

### 4. Frontend Dashboard
- **Technology:** HTML, CSS, and JavaScript
- **Visualization:** Use a library like Chart.js to display sentiment counts (positive, neutral, negative) in real time.
- **Responsibility:** Poll or receive real-time updates from the backend and update the visualization.

## Data Flow
1. The backend establishes a connection with the Twitter API to ingest data.
2. Incoming tweets are processed through the sentiment analysis model.
3. Processed data is stored temporarily (or streamed directly) and exposed via API endpoints.
4. The frontend dashboard requests updated data (or receives pushes) and refreshes the charts.

