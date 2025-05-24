import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import zipfile
import os


def download_and_extract_data():
    if not os.path.exists('ml-100k'):
        try:
            import urllib.request
            url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
            urllib.request.urlretrieve(url, 'ml-100k.zip')
            with zipfile.ZipFile('ml-100k.zip', 'r') as zip_ref:
                zip_ref.extractall('.')
            print("Dataset downloaded and extracted successfully")
        except Exception as e:
            print(f"Download failed: {e}")
            print("Please manually download the ml-100k.zip dataset and extract it to the current directory")
            exit(1)

# Load rating data
def load_ratings_data():
    columns = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv('ml-100k/u.data', sep='\t', names=columns)
    return data
# Create utility matrix
def create_utility_matrix(data):
    utility_matrix = data.pivot(index='user_id', columns='item_id', values='rating')
    return utility_matrix

# Calculate user similarity
def calculate_similarity(utility_matrix):
    # Calculate user average ratings
    user_means = utility_matrix.mean(axis=1)
    # Center the ratings
    centered_matrix = utility_matrix.sub(user_means, axis=0)
    # Fill missing values with 0
    centered_matrix_filled = centered_matrix.fillna(0)
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(centered_matrix_filled)
    similarity_df = pd.DataFrame(similarity_matrix, index=utility_matrix.index, columns=utility_matrix.index)
    return similarity_df, user_means, centered_matrix

# Find most similar users
def find_similar_users(similarity_df, target_user_id, n=10):
    # Get the top n most similar users to the target user
    similar_users = similarity_df.loc[target_user_id].sort_values(ascending=False)[1:n + 1]
    return similar_users

# Predict rating
def predict_rating(utility_matrix, user_means, similar_users, item_id, target_user_id):
    # Get ratings of the item by similar users
    similar_ratings = utility_matrix.loc[similar_users.index, item_id]
    # Filter out users who didn't rate the item
    rated_users = similar_ratings[~similar_ratings.isna()].index

    if len(rated_users) == 0:
        print("None of the similar users rated item 508")
        return user_means[target_user_id]  # Return the target user's average rating as the prediction

    # Get the similarity weights and ratings of these users
    weights = similar_users[rated_users]
    ratings = utility_matrix.loc[rated_users, item_id]

    # Calculate weighted average rating
    if weights.sum() == 0:
        return user_means[target_user_id]

    # Predict based on the user's original ratings
    prediction = (ratings * weights).sum() / weights.sum()
    return prediction

def main():
    download_and_extract_data()
    data = load_ratings_data()
    utility_matrix = create_utility_matrix(data)
    # Calculate similarity
    similarity_df, user_means, centered_matrix = calculate_similarity(utility_matrix)
    # Target user ID and item ID
    target_user_id = 1
    target_item_id = 508

    # Find top 10 similar users
    similar_users = find_similar_users(similarity_df, target_user_id, 10)
    # Predict rating
    predicted_rating = predict_rating(utility_matrix, user_means, similar_users, target_item_id, target_user_id)
    print(f"The 10 most similar users to user {target_user_id}:")
    print(similar_users)
    print(f"\nThe predicted rating for user {target_user_id} on item {target_item_id} is: {predicted_rating:.4f}")

if __name__ == "__main__":
    main()