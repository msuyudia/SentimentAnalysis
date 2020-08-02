import re
import tweepy

consumer_key = "your_consumer_key"
consumer_secret = "your_consumer_secret"
access_key = "your_access_key"
access_secret = "your_access_secret"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)

search_key = '@pln_123'

api = tweepy.API(auth)

tweets = tweepy.Cursor(api.search,
                       q=search_key,
                       tweet_mode="extended").items(1000)

with open("tweets_all_25.txt", "w+", encoding="utf-8") as file_tweets_txt:
    for tweet in tweets:
        if not re.search(r'\bRT @\S*\b', tweet.full_text):
            date = tweet.created_at
            text = tweet.full_text.replace('\n', ' ')
            file_tweets_txt.write("%s | %s\n" % (date, text))

file_tweets_txt.close()
