#%%

words = "(Lehrkräftebildung OR Lehrerbildung OR Lehrkräfte OR Lehrkräftefortbildung OR Seiteneinstieg OR Quereinstieg OR Lehramt)" # The words you want tweets to include
words = "(Lehrkräftebildung)" # The words you want tweets to include
query = words + " lang:de" # The query you want to search for
tweets_limit = 2000 # The number of tweets you want to return in total (must be divisible by 100 and cannot exceed 2000)

import tweepy
import pandas as pd
import re
import datetime

from config import bearer_token

# Initialize the Tweepy client
auth = tweepy.OAuth2BearerHandler(bearer_token)
api = tweepy.API(auth)

# get current time and date
now = datetime.datetime.now()
print("Current date and time: ")
print(now.strftime("%Y-%m-%d_%H-%M-%S"))

# tweets = api.search_30_day(label='dev', query=query)

# tweets = tweepy.Cursor(api.search_30_day,
#                        label='dev,'
#                        q=query,
#                        count=100).items(500)

# Define a function to query tweets
def get_retweets(query, num_tweets=2000):
    # Initialize an empty DataFrame to store user data and tweets
    tweets_df = pd.DataFrame()

    # Return the number of batches based on num_tweets
    # if num_tweets > 2000:
    #     raise ValueError("`num_tweets` must be less than or equal to 2000.")
    if num_tweets % 100 != 0:
        raise ValueError("`num_tweets` must be a multiple of 100.")
    max_results = 100
    limit = num_tweets / max_results

    output = []

    tweets_list = tweepy.Cursor(api.search_30_day,
                               label='dev,',
                               query=query,
                               count=100,
                               ).items(limit)

    # Iterate through batches of tweets
    for tweet in tweets_list:

        if tweet.text.startswith("RT"):

            text = tweet.text
            favourite_count = tweet.favorite_count
            retweet_count = tweet.retweet_count
            created_at = tweet.created_at
            tweeter = re.search("@(\w+)", tweet.text)
            retweeter = tweet.user.screen_name
            followers_count = tweet.user.followers_count

            line = {
                    'text' : text,
                    'favourite_count' : favourite_count,
                    'retweet_count' : retweet_count,
                    'created_at' : created_at,
                    'followers_count' : followers_count,
                    'tweeter' : tweeter,
                    'retweeter' : retweeter
                    }

            output.append(line)

    df = pd.DataFrame(output)
    return df


            # batch_data = pd.DataFrame(tweet_batch.data)
            # users = {u["id"]: u["username"] for u in tweet_batch.includes["users"]}
            # batch_data["retweeter"] = batch_data["author_id"].map(users)
            # # Concatenate temporary DataFrames to existing DataFrames
            # tweets_df = pd.concat([tweets_df, batch_data])



    # Merge user information to tweet information on author_id
    # Extract original tweeter from tweet text
    tweets_df["tweeter"] = tweets_df["text"].str.extract(r"@(\w+)")
    # Return DataFrame
    return tweets_df


# Create a DataFrame using the function defined above
df = get_retweets(query, tweets_limit)

df.to_csv(f'data/tweets_lehrkraeftebildung_v1_{now}.csv')


#%%

Lehrkräftebildung
Lehrerbildung
Professionalisierung Lehrkräfte
Lehrkräftefortbildung
Quer- und Seiteneinstieg
flexible Wege ins Lehramt

