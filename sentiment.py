from textblob import TextBlob
import pandas as pd

def compute_sentiment(df, text_col="headline"):
    sentiments = []
    for text in df[text_col]:
        score = TextBlob(str(text)).sentiment.polarity
        sentiments.append(score)

    df["sentiment"] = sentiments
    return df

if __name__ == "__main__":
    sample = pd.DataFrame({"headline": ["Markets rally as inflation cools", "Stocks fall on recession fears"]})
    print(compute_sentiment(sample))
