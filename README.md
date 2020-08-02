# SentimentAnalysis
Sentiment analysis to get people's sentiments about company services classified by dates, services and places. For this case from people in DKI Jakarta for services from PT. PLN regarding electricity in DKI Jakarta in the scope of Twitter social media, especially those with opinions on PLN's official Twitter account.

How is work?
You can use PyCharm for much easy to install all library that i use it. Include : tweepy, numpy, Sastrawi (Stemmer for Indonesian words), sklearn, and pandas.
You can download PyCharm in site :
https://www.jetbrains.com/pycharm/download/#section=windows
And don't forget to install python 3.8 and sync to your PyCharm. this project can't work at version 2.x.x or 4.x.x above (if python already release version 4)

Run Crawling.py to get data for new people's opinion about services (in this case regarding electricity in DKI Jakarta). Don't forget before crawling, you must get your consumer api keys and access token & access token secret from your app. You can get it in site :
https://developer.twitter.com/

Run SentimentAnalysis.py for get a few bar charts of infographic people's sentiments that classified by dates, services and places.

Run Evaluation.py for get results of total accuracy of SentimentAnalysis.py.

Notes :

-list_cleaned_tweets.txt is text for all cleaned tweets after results of function text_preprocessing

-list_daerah.txt is text for all places that want to be classified, in this case in DKI Jakarta, such as Kotamadya, Kecamatan and Kelurahan. Places divided by "|", which is format like this "Kelurahan | Kecamatan | Kotamadya"

-normalization_words.txt is text for all words that must be normalized, for example slank words, abbreviation, and another non-standard words.

-tweets_all.txt, tweets_all_2.txt, etc is text for all results text after crawling based on query search in Crawling.py this tweets divided by dates and tweets, which is format like this "Date | Tweet"

-tweets_predicted_labels.txt is results label after classified sentiment based on tweets_training.txt labels use K-Nearest Neighbor (K-NN) method

-tweets_testing.txt is a data testing to classify which data positive sentiment, netral sentiment and negative sentiment, which is format like this "Date | Tweet"

-tweets_training.txt is a data training as basis sentiment for classify, which is format like this "Label | Tweet"
