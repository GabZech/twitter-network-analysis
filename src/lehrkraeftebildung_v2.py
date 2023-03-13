#%%

words = "(Lehrkräftebildung OR Lehrerbildung OR Lehrkräfte OR Lehrkräftefortbildung OR Seiteneinstieg OR Quereinstieg OR Lehramt)" # The words you want tweets to include
query = words + " (is:retweet OR is:quote) lang:de" # The query you want to search for
tweets_limit = 2000 # The number of tweets you want to return in total (must be divisible by 100 and cannot exceed 2000)

import tweepy
import pandas as pd
import datetime

from config import bearer_token

# Initialize the Tweepy client
client = tweepy.Client(bearer_token=bearer_token)


# Define a function to query tweets
def get_retweets_v2(query, num_tweets=1000):
    # Initialize an empty DataFrame to store user data and tweets
    tweets_df = pd.DataFrame()
    
    # Return the number of batches based on num_tweets
    if num_tweets > 2000:
        raise ValueError("`num_tweets` must be less than or equal to 2000.")
    elif num_tweets % 100 != 0:
        raise ValueError("`num_tweets` must be a multiple of 100.")
    max_results = 100
    limit = num_tweets / max_results

    # Iterate through batches of tweets
    for tweet_batch in tweepy.Paginator(client.search_recent_tweets, 
                                query=query,
                                expansions='author_id',
                                user_fields=["username", "id"],
                                tweet_fields=["public_metrics"],
                                max_results=100,
                                limit=limit
                                ):
        batch_data = pd.DataFrame(tweet_batch.data)
        users = {u["id"]: u["username"] for u in tweet_batch.includes["users"]}
        batch_data["retweeter"] = batch_data["author_id"].map(users)
        # Concatenate temporary DataFrames to existing DataFrames
        tweets_df = pd.concat([tweets_df, batch_data])

    # Merge user information to tweet information on author_id
    # Extract original tweeter from tweet text
    tweets_df["tweeter"] = tweets_df["text"].str.extract(r"@(\w+)")
    # Return DataFrame
    return tweets_df



