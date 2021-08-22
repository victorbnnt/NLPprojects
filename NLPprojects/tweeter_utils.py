import pandas as pd
import tweepy
import os

PATH_TO_CREDENTIALS = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/credentials/"


def tweeter_authenticate():
    """ authenticate to tweeter """

    # load credentials secrets
    twitter_credentials = pd.read_csv(os.path.join(PATH_TO_CREDENTIALS, "twitter_credentials.csv"))

    # authenticate
    authenticate = tweepy.OAuthHandler(twitter_credentials["api_key"][0], twitter_credentials["api_secret_key"][0])
    authenticate.set_access_token(twitter_credentials["access_token"][0], twitter_credentials["access_token_secret"][0])
    return tweepy.API(authenticate, wait_on_rate_limit=True)


def get_tweets_from_username(api, screen_name):
    """ get last 3200 tweets from specific user """

    # initialize a list to hold all the Tweets
    alltweets = []
    output = []

    # make initial request for most recent tweets
    # (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name=screen_name, count=200, tweet_mode="extended")

    # save most recent tweets
    alltweets.extend(new_tweets)

    # save the id of the oldest tweet less one to avoid duplication
    oldest = alltweets[-1].id - 1

    # keep grabbing tweets until there are no tweets left
    while len(new_tweets) > 0:
        print("Getting tweets before %s" % (oldest))

        # all subsequent requests use the max_id param to prevent
        # duplicates
        new_tweets = api.user_timeline(screen_name=screen_name, count=200, max_id=oldest, tweet_mode="extended")

        # save most recent tweets
        alltweets.extend(new_tweets)

        # update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
        print("... %s tweets downloaded so far" % (len(alltweets)))

    # transform the tweepy tweets into a 2D array that will
    for tweet in alltweets:
        output.append([tweet.id_str,
                       tweet.created_at,
                       tweet.full_text,
                       tweet.in_reply_to_screen_name,
                       tweet.user.name,
                       tweet.user.location,
                       tweet.user.followers_count,
                       tweet.user.friends_count,
                       tweet.geo,
                       tweet.coordinates,
                       tweet.retweet_count,
                       tweet.favorite_count,
                       tweet.lang,
                       tweet.retweeted])

    # Convert to dataframe
    df = pd.DataFrame.from_records(output, columns=["id_str",
                                                    "created_at",
                                                    "full_text",
                                                    "in_reply_to_screen_name",
                                                    "user_name",
                                                    "user_location",
                                                    "user_followers_count",
                                                    "user_friends_count",
                                                    "geo",
                                                    "coordinates",
                                                    "retweet_count",
                                                    "favorite_count",
                                                    "lang",
                                                    "retweeted"])
    return df


def get_tweets_from_search(api, search_string, parameters=" -filter:retweets", since="2021-08-09", lang="en", max_tweets=1000):
    """ get tweets from a specific search """

    tweet_list = []
    count = 0
    search = search_string
    params = parameters

    for tweet in tweepy.Cursor(api.search, q=search + params,
                               count=100,
                               tweet_mode="extended",
                               lang=lang,
                               since=since,
                               # until="2015-02-01",
                               ).items():
        tweet_list.append(tweet._json["full_text"])
        count += 1
        if count == max_tweets:
            break
        print(count)
    return pd.DataFrame({"text": tweet_list})
