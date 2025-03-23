import requests

def fetch_instagram_posts(access_token, user_id, count=5):
    url = f"https://graph.instagram.com/{user_id}/media"
    params = {
        "fields": "id, caption, media_url, timestamp",
        "access_token": access_token,
        "limit": count
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get("data", [])
    else:
        return []
