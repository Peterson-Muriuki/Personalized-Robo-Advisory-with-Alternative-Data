import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import subprocess
import json

analyzer = SentimentIntensityAnalyzer()
tweets_list = []

query = "crypto OR bitcoin OR ethereum since:2026-01-01"

# Use snscrape as a subprocess
cmd = f'snscrape --jsonl twitter-search "{query}"'
result = subprocess.run(cmd, capture_output=True, text=True, shell=True)

for i, line in enumerate(result.stdout.splitlines()):
    if i > 100:
        break
    tweet = json.loads(line)
    sentiment = analyzer.polarity_scores(tweet["content"])["compound"]
    tweets_list.append([tweet["date"], tweet["content"], sentiment])

df_tweets = pd.DataFrame(tweets_list, columns=["date", "tweet", "sentiment"])
df_tweets.to_csv("twitter_sentiment.csv", index=False)
