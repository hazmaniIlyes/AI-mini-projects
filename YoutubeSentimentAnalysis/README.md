```python
from googleapiclient.discovery import build
from textblob import TextBlob

def get_youtube_commentes(video_id, api_key, max_results=20):

    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []

    request = youtube.commentThreads().list(
        part = 'snippet',
        videoId = video_id,
        maxResults = max_results,
        textFormat = 'plainText'
    )
    response = request.execute()

    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comments.append(comment)

    return comments

def analyze_comments(comments):
    results = []
    for text in comments:
        polarity = TextBlob(text).sentiment.polarity
        sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
        results.append({
            'comment': text,
            'polarity': polarity,
            'sentiment': sentiment
        })

    return results


comments = get_youtube_commentes('bo47JoSxl1s', 'AIzaSyCg-kRjqmn3aKK0IEhoA8ajwyQBykloSy8',20)
print(comments)
results = analyze_comments(comments)
print("-------------------------------------------------")
print(results)
```
