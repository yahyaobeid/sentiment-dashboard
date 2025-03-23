import re

def extract_hashtags(caption):
    return re.findall(r'#\w+', caption)

def extract_keywords(caption):
    words = caption.split()
    stop_words = {'the', 'and', 'is', 'in', 'to', 'of'}
    return [word for word in words if word.lower() not in stop_words]

def extract_features_from_caption(caption):
    hashtags = extract_hashtags(caption)
    keywords = extract_keywords(caption)

    return ' '.join(hashtags + keywords)