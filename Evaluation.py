import os
import json
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score

class VideoRecommendationSystem:
    def __init__(self, data_path='data/summary.json'):
        
        with open(data_path) as f:
            data = json.load(f)       
        
        posts = data.get('posts', [])        
        scaler = MinMaxScaler()        
        view_counts = np.array([post.get('view_count', 0) for post in posts])
        upvote_counts = np.array([post.get('upvote_count', 0) for post in posts])
        average_ratings = np.array([post.get('average_rating', 0) for post in posts])
        share_counts = np.array([post.get('share_count', 0) for post in posts])        
        normalized_view_counts = scaler.fit_transform(view_counts.reshape(-1, 1)).flatten()
        normalized_upvote_counts = scaler.fit_transform(upvote_counts.reshape(-1, 1)).flatten()
        normalized_average_ratings = scaler.fit_transform(average_ratings.reshape(-1, 1)).flatten()
        normalized_share_counts = scaler.fit_transform(share_counts.reshape(-1, 1)).flatten()
        
        # Combined scores (simple average of normalized metrics)
        self.combined_scores = (
            normalized_view_counts + 
            normalized_upvote_counts + 
            normalized_average_ratings + 
            normalized_share_counts
        ) / 4

def load_and_calculate_ground_truth_scores(data_path='data/summary.json'):
    with open(data_path) as f:
        ground_truth_data = json.load(f)
    
    posts = ground_truth_data.get('posts', [])
    
    # Convert the ground truth data into numpy arrays for each metric
    view_counts = np.array([post.get('view_count', 0) for post in posts])
    upvote_counts = np.array([post.get('upvote_count', 0) for post in posts])
    average_ratings = np.array([post.get('average_rating', 0) for post in posts])
    share_counts = np.array([post.get('share_count', 0) for post in posts])
    
    # Normalize the metrics
    scaler = MinMaxScaler()
    normalized_view_counts = scaler.fit_transform(view_counts.reshape(-1, 1)).flatten()
    normalized_upvote_counts = scaler.fit_transform(upvote_counts.reshape(-1, 1)).flatten()
    normalized_average_ratings = scaler.fit_transform(average_ratings.reshape(-1, 1)).flatten()
    normalized_share_counts = scaler.fit_transform(share_counts.reshape(-1, 1)).flatten()
    
    # Combine the normalized metrics into a single combined score 
    combined_scores = (
        normalized_view_counts + 
        normalized_upvote_counts + 
        normalized_average_ratings + 
        normalized_share_counts
    ) / 4
    
    return combined_scores

def evaluate_recommendation_system(data_path='data/summary.json', threshold=0.5):
    
    
    ground_truth_combined_scores = load_and_calculate_ground_truth_scores(data_path)   
    
    recommendation_system = VideoRecommendationSystem(data_path)
    predicted_scores = recommendation_system.combined_scores    
    
    
    mae = mean_absolute_error(ground_truth_combined_scores, predicted_scores)
    rmse = math.sqrt(mean_squared_error(ground_truth_combined_scores, predicted_scores))
    
    
    ground_truth_binary = (ground_truth_combined_scores >= threshold).astype(int)
    predicted_binary = (predicted_scores >= threshold).astype(int)
    
    # Calculate F1 score (with average='binary' for binary classification)
    f1 = f1_score(ground_truth_binary, predicted_binary, average='binary')
    
    # Calculate precision and recall
    true_positives = np.sum((ground_truth_binary == 1) & (predicted_binary == 1))
    false_positives = np.sum((ground_truth_binary == 0) & (predicted_binary == 1))
    false_negatives = np.sum((ground_truth_binary == 1) & (predicted_binary == 0))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return {
        "Mean Absolute Error (MAE)": mae,
        "Root Mean Square Error (RMSE)": rmse,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall,
        "Number of Videos": len(ground_truth_combined_scores),
        "Threshold": threshold
    }

def main():
    
    results = evaluate_recommendation_system()
    
    
    print("\n--- Recommendation System Evaluation ---")
    for metric, value in results.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main()