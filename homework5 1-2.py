import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import zipfile
import os
import requests
from io import BytesIO


def download_and_extract_data():
    if not os.path.exists('ml-100k'):
        try:
            print("Starting dataset download...")
            url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
            response = requests.get(url)
            with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
                zip_ref.extractall('.')
            print("Dataset downloaded and extracted successfully")
        except Exception as e:
            print(f"Download failed: {e}")
            print("Please manually download the ml-100k.zip dataset and extract it to the current directory")
            exit(1)

def load_ratings_data():
    columns = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv('ml-100k/u.data', sep='\t', names=columns)
    return data

def create_utility_matrix(data):
    utility_matrix = data.pivot(index='user_id', columns='item_id', values='rating')
    return utility_matrix

def center_ratings(utility_matrix):
    # Calculate user average ratings
    user_means = utility_matrix.mean(axis=1)
    # Center the ratings
    centered_matrix = utility_matrix.sub(user_means, axis=0)
    return centered_matrix, user_means

def build_user_profile(centered_matrix, user_id):
    return centered_matrix.loc[user_id].dropna()

def get_item_vector(centered_matrix, item_id):
    return centered_matrix[item_id].dropna()

def calculate_similarity(user_profile, item_vector):
    # Find common indices
    common_indices = user_profile.index.intersection(item_vector.index)
    if len(common_indices) == 0:
        print("No common ratings found, unable to calculate similarity")
        return 0, float('inf')

    # Extract common ratings
    user_common = user_profile[common_indices].values.reshape(1, -1)
    item_common = item_vector[common_indices].values.reshape(1, -1)
    # Calculate cosine similarity
    cos_sim = cosine_similarity(user_common, item_common)[0][0]
    # Calculate Euclidean distance
    distance = np.linalg.norm(user_common - item_common)
    return cos_sim, distance


def main():
    download_and_extract_data()
    data = load_ratings_data()
    # Create utility matrix
    utility_matrix = create_utility_matrix(data)
    # Center ratings
    centered_matrix, user_means = center_ratings(utility_matrix)
    # User IDs and item ID
    user_id_1 = 200
    user_id_2 = 15
    item_id = 95
    # Build user profiles
    user_200_profile = build_user_profile(centered_matrix, user_id_1)
    user_15_profile = build_user_profile(centered_matrix, user_id_2)
    # Get item vector
    item_95_vector = get_item_vector(centered_matrix, item_id)
    # Calculate similarity and distance
    cos_sim_200, dist_200 = calculate_similarity(user_200_profile, item_95_vector)
    cos_sim_15, dist_15 = calculate_similarity(user_15_profile, item_95_vector)
    # Predict ratings
    user_200_mean = user_means[user_id_1]
    user_15_mean = user_means[user_id_2]
    # Predicted rating based on similarity
    predicted_rating_200 = user_200_mean + (cos_sim_200 * np.std(utility_matrix.loc[user_id_1].dropna()))
    predicted_rating_15 = user_15_mean + (cos_sim_15 * np.std(utility_matrix.loc[user_id_2].dropna()))
    # Determine recommendation strategy
    recommendation_user = None
    recommendation_reason = ""

    if cos_sim_200 > cos_sim_15:
        recommendation_user = user_id_1
        recommendation_reason = f"User {user_id_1} has higher cosine similarity ({cos_sim_200:.4f} > {cos_sim_15:.4f})"
    elif cos_sim_15 > cos_sim_200:
        recommendation_user = user_id_2
        recommendation_reason = f"User {user_id_2} has higher cosine similarity ({cos_sim_15:.4f} > {cos_sim_200:.4f})"
    else:
        # If similarities are equal, choose the one with smaller distance
        if dist_200 < dist_15:
            recommendation_user = user_id_1
            recommendation_reason = f"Cosine similarities are equal, but user {user_id_1} has smaller Euclidean distance ({dist_200:.4f} < {dist_15:.4f})"
        else:
            recommendation_user = user_id_2
            recommendation_reason = f"Cosine similarities are equal, but user {user_id_2} has smaller Euclidean distance ({dist_15:.4f} < {dist_200:.4f})"

    # Output results
    print("\n--- Analysis Results ---")
    print(f"Cosine similarity between user {user_id_1} and item {item_id}: {cos_sim_200:.4f}")
    print(f"Euclidean distance between user {user_id_1} and item {item_id}: {dist_200:.4f}")
    print(f"Predicted rating for user {user_id_1}: {predicted_rating_200:.4f}")

    print(f"\nCosine similarity between user {user_id_2} and item {item_id}: {cos_sim_15:.4f}")
    print(f"Euclidean distance between user {user_id_2} and item {item_id}: {dist_15:.4f}")
    print(f"Predicted rating for user {user_id_2}: {predicted_rating_15:.4f}")

    print(f"\nThe recommendation system should recommend item {item_id} to user {recommendation_user}")
    print(f"Reason: {recommendation_reason}")

if __name__ == "__main__":
    main()