from datetime import datetime
import pickle
import numpy as np
import tensorflow as tf
from twikit import Client
import pandas as pd
import asyncio
import requests
from UserData import USERNAME, PASSWORD, EMAIL 
def read_tokens(file_path):
    with open(file_path, 'r') as file:
        tokens = file.read().splitlines()
    return tokens
# Set up the headers for authentication


# Initialize Twikit Client
client = Client('en-US')

# Login to Twitter (if required)
async def login_to_twitter():
    await client.login(
        auth_info_1=USERNAME,
        auth_info_2=EMAIL,
        password=PASSWORD,
        cookies_file='cookies.json'
    )

# Fetch user data using twikit
async def login_to_twitter():
    USERNAME = 'aa11233ajjs'
    EMAIL = 'aminaa.abdagic@gmail.com'
    PASSWORD = 'String1!'
    await client.login(
        auth_info_1=USERNAME,
        auth_info_2=EMAIL,
        password=PASSWORD,
        cookies_file='cookies.json'
    )

# Fetch user data using twikit
async def get_data_for_user(username):
    url = f'https://api.twitter.com/2/users/by/username/{username}'
    # Fetch user profile
    user = await client.get_user_by_screen_name(username)
    user_fields = [
        'id', 'name', 'username', 'created_at', 'description', 'url', 'protected',
        'verified', 'profile_image_url', 'location', 'public_metrics'
    ]
    params = {'user.fields': ','.join(user_fields)}
    for i in read_tokens('bearerTokens.txt'):
        headers = {
        'Authorization': f'Bearer {i}',
        'Accept': 'application/json'
    }
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            break
    user_2 = response.json().get('data', {})
    print(user_2)
    # Extract user data
    user_id = user.id
    name_length = len(user.name)
    screen_name_length = len(user.screen_name)
    screen_name_length_name_length_ratio = screen_name_length / (name_length + 1)

    # Calculate account age
    created_at_str = user.created_at
    account_age = (datetime.utcnow() - datetime.strptime(created_at_str, "%a %b %d %H:%M:%S +0000 %Y")).days if created_at_str else 0
    public_metrics = user_2.get('public_metrics', {})
    description_length = len(user.description)
    profile_has_url = 1 if user.url else 0
    protected = int(user_2.get('protected', 0))
    verified = int(user_2.get('verified', 0))
    # Extract public metrics
    followers_count = user.followers_count
    # `friends_count` is not directly available in twikit, so we skip it
    friends_count = public_metrics.get('following_count', 0) # Placeholder, as twikit does not provide this
    listed_count = user.listed_count

    # Compute derived metrics
    friends_followers_ratio = friends_count / (followers_count + 1)
    friends_followers_ratio_beq_50 = int(friends_followers_ratio <= 50)
    friends_followers_square_ratio = (friends_count ** 2) / (followers_count + 1)
    friends_followers_plus_friends_ratio = (friends_count + followers_count) / (followers_count + 1)
    two_followers_minus_friends = (2 * followers_count) - friends_count
    two_followers_beq_100 = int((2 * followers_count) <= 100)
    lists_followers_ratio = listed_count / (followers_count + 1)

    # Fetch tweets
    tweets = await client.get_user_tweets(user_id, 'Tweets')
    tweet_lengths, retweets, favorites, hashtags_counts, mentions_counts = [], [], [], [], []
    num_reply, num_retweet = 0, 0
    print(tweets)
    for tweet in tweets:
        text = tweet.text
        tweet_lengths.append(len(text))

        retweet_count = tweet.retweet_count
        like_count = tweet.favorite_count
        reply_count = tweet.reply_count

        retweets.append(retweet_count)
        favorites.append(like_count)
        num_reply += reply_count
        num_retweet += retweet_count

        # Extract hashtags and mentions count
        hashtags = tweet.entities.get('hashtags', []) if hasattr(tweet, 'entities') else []
        mentions = tweet.entities.get('user_mentions', []) if hasattr(tweet, 'entities') else []

        hashtags_counts.append(len(hashtags))
        mentions_counts.append(len(mentions))

    # Calculate tweet statistics
    min_tweet_length = min(tweet_lengths, default=0)
    max_tweet_length = max(tweet_lengths, default=0)
    avg_tweet_length = sum(tweet_lengths) / len(tweet_lengths) if tweet_lengths else 0

    min_hashtags = min(hashtags_counts, default=0)
    max_hashtags = max(hashtags_counts, default=0)
    avg_hashtags = sum(hashtags_counts) / len(hashtags_counts) if hashtags_counts else 0

    max_mentions = max(mentions_counts, default=0)
    avg_mentions = sum(mentions_counts) / len(mentions_counts) if mentions_counts else 0

    max_retweets = max(retweets, default=0)
    avg_retweets = sum(retweets) / len(retweets) if retweets else 0

    min_favorite = min(favorites, default=0)
    max_favorite = max(favorites, default=0)
    avg_favorite = sum(favorites) / len(favorites) if favorites else 0


    # Return results
    results = {
        'id': user_id,
        'name_length': name_length,
        'screen_name_length': screen_name_length,
        'screen_name_length_name_length_ratio': screen_name_length_name_length_ratio,
        'account_age': account_age,
        'description_length': description_length,
        'protected': protected,
        'verified': verified,
        'profile_has_url': profile_has_url,
        'followers_count': followers_count,
        'friends_count': friends_count,  # Placeholder, as twikit does not provide this
        'listed_count': listed_count,
        'friends_followers_ratio': friends_followers_ratio,
        'friends_followers_ratio_beq_50': friends_followers_ratio_beq_50,
        'friends_followers_square_ratio': friends_followers_square_ratio,
        'friends_followers+friends_ratio': friends_followers_plus_friends_ratio,
        '2_followers_minus_friends': two_followers_minus_friends,
        '2_followers_beq_100': two_followers_beq_100,
        'lists_followers_ratio': lists_followers_ratio,
        'num_reply': num_reply,
        'num_retweet': num_retweet,
        'min_favorite': min_favorite,
        'max_favorite': max_favorite,
        'avg_favorite': avg_favorite,
        'max_retweets': max_retweets,
        'avg_retweets': avg_retweets,
        'max_mentions': max_mentions,
        'avg_mentions': avg_mentions,
        'min_hashtags': min_hashtags,
        'max_hashtags': max_hashtags,
        'avg_hashtags': avg_hashtags,
        'min_tweet_length': min_tweet_length,
        'max_tweet_length': max_tweet_length,
        'avg_tweet_length': avg_tweet_length,
    }
    return results

# Load the trained model and scaler
model_path = r"model\model2_twitter.h5"
scaler_path = r"model\scaler2.pkl"
model = tf.keras.models.load_model(model_path)
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)
# Predict profile
async def predict_profile(username):
    # Extract user data from Twitter
    profile = await get_data_for_user(username)
    # Feature mapping based on returned profile data
    data_for_model = {
    'name_length': profile.get('name_length', 0),
    'screen_name_length': profile.get('screen_name_length', 0),
    'screen_name_length_name_length_ratio': profile.get('screen_name_length_name_length_ratio', 0),
    'account_age': profile.get('account_age', 0),
    'description_length': profile.get('description_length', 0),
    'protected': profile.get('protected', 0),
    'verified': profile.get('verified', 0),
    'profile_has_url': profile.get('profile_has_url', 0),
    'followers_count': profile.get('followers_count', 0),
    'friends_count': profile.get('friends_count', 0),
    'listed_count': profile.get('listed_count', 0),
    'friends_followers_ratio': profile.get('friends_followers_ratio', 0),
    'friends_followers_ratio_beq_50': profile.get('friends_followers_ratio_beq_50', 0),
    'friends_followers_square_ratio': profile.get('friends_followers_square_ratio', 0),
    'friends_followers+friends_ratio': profile.get('friends_followers_friends_ratio', 0),  # Replaced '+' with '_'
    '2_followers_minus_friends': profile.get('2_followers_minus_friends', 0),
    '2_followers_beq_100': profile.get('2_followers_beq_100', 0),
    'lists_followers_ratio': profile.get('lists_followers_ratio', 0),
    'num_reply': profile.get('num_reply', 0),
    'num_retweet': profile.get('num_retweet', 0),
    'min_favorite': profile.get('min_favorite', 0),
    'max_favorite': profile.get('max_favorite', 0),
    'avg_favorite': profile.get('avg_favorite', 0),
    'max_retweets': profile.get('max_retweets', 0),
    'avg_retweets': profile.get('avg_retweets', 0),
    'max_mentions': profile.get('max_mentions', 0),
    'avg_mentions': profile.get('avg_mentions', 0),
    'min_hashtags': profile.get('min_hashtags', 0),
    'max_hashtags': profile.get('max_hashtags', 0),
    'avg_hashtags': profile.get('avg_hashtags', 0),
    'min_tweet_length': profile.get('min_tweet_length', 0),
    'max_tweet_length': profile.get('max_tweet_length', 0),
    'avg_tweet_length': profile.get('avg_tweet_length', 0)
}
    print(data_for_model)

    features = list(data_for_model.keys())
    desired_columns = ['name_length', 'screen_name_length', 'screen_name_length_name_length_ratio', 'account_age', 'description_length', 'protected',
    'verified', 'profile_has_url', 'followers_count', 'friends_count',
    'listed_count', 'friends_followers_ratio', 'friends_followers_ratio_beq_50',
    'friends_followers_square_ratio', 'friends_followers+friends_ratio',  # Replaced '+' with '_'
    '2_followers_minus_friends', '2_followers_beq_100', 'lists_followers_ratio', 'num_reply', 'num_retweet', 'min_favorite', 
    'max_favorite', 'avg_favorite', 'max_retweets', 'avg_retweets', 'max_mentions', 'avg_mentions', 'min_hashtags', 'max_hashtags', 'avg_hashtags', 'min_tweet_length', 'max_tweet_length', 'avg_tweet_length'
]
    X_input = np.array([[data_for_model[f] for f in desired_columns]])
    X_input = scaler.transform(X_input)
    X_input = X_input.reshape(1, 1, X_input.shape[1])

    fake_probability = model.predict(X_input)[0][0] * 100  
    print(model.predict(X_input))
    return {
        "profile_data": profile,
        "fake_probability": fake_probability
    }


