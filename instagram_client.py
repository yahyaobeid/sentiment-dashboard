import requests
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional
import json

class InstagramClient:
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.base_url = "https://graph.instagram.com/v12.0"
        
    def get_post_data(self, post_id: str) -> Dict:
        """Get comprehensive data about a specific post."""
        endpoint = f"{self.base_url}/{post_id}"
        params = {
            "fields": "id,caption,media_type,media_url,permalink,thumbnail_url,timestamp,username,like_count,comments_count,video_views,hashtags,location,children{media_url,media_type,video_views}",
            "access_token": self.access_token
        }
        
        response = requests.get(endpoint, params=params)
        return response.json()
    
    def get_hashtag_data(self, hashtag: str) -> Dict:
        """Get data about a specific hashtag."""
        endpoint = f"{self.base_url}/ig_hashtag_search"
        params = {
            "q": hashtag,
            "access_token": self.access_token
        }
        
        response = requests.get(endpoint, params=params)
        return response.json()
    
    def get_hashtag_top_posts(self, hashtag_id: str) -> List[Dict]:
        """Get top posts for a specific hashtag."""
        endpoint = f"{self.base_url}/{hashtag_id}/top_media"
        params = {
            "fields": "id,caption,media_type,media_url,permalink,thumbnail_url,timestamp,username,like_count,comments_count,video_views",
            "access_token": self.access_token
        }
        
        response = requests.get(endpoint, params=params)
        return response.json().get("data", [])
    
    def get_user_insights(self, user_id: str) -> Dict:
        """Get insights about a user's account."""
        endpoint = f"{self.base_url}/{user_id}/insights"
        params = {
            "metric": "engagement,impressions,reach,saved",
            "period": "lifetime",
            "access_token": self.access_token
        }
        
        response = requests.get(endpoint, params=params)
        return response.json()
    
    def collect_training_data(self, hashtags: List[str], limit: int = 100) -> pd.DataFrame:
        """Collect training data from multiple hashtags."""
        all_posts = []
        
        for hashtag in hashtags:
            hashtag_data = self.get_hashtag_data(hashtag)
            if "data" in hashtag_data and hashtag_data["data"]:
                hashtag_id = hashtag_data["data"][0]["id"]
                posts = self.get_hashtag_top_posts(hashtag_id)
                
                for post in posts:
                    if len(all_posts) >= limit:
                        break
                        
                    post_data = {
                        "post_id": post["id"],
                        "caption": post.get("caption", ""),
                        "timestamp": post["timestamp"],
                        "likes": post.get("like_count", 0),
                        "comments": post.get("comments_count", 0),
                        "views": post.get("video_views", 0),
                        "hashtags": self._extract_hashtags(post.get("caption", "")),
                        "hour": datetime.fromisoformat(post["timestamp"].replace("Z", "+00:00")).hour,
                        "day_of_week": datetime.fromisoformat(post["timestamp"].replace("Z", "+00:00")).weekday()
                    }
                    all_posts.append(post_data)
        
        return pd.DataFrame(all_posts)
    
    def _extract_hashtags(self, caption: str) -> List[str]:
        """Extract hashtags from a caption."""
        import re
        return re.findall(r'#\w+', caption)
