import os
import json
import numpy as np
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

class VideoRecommendationSystem:
    def __init__(self, data_path='data/summary.json'):
        
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')     
        self.data_path = data_path
        self.load_data()
        
        #Get embeddings and calculate matrix and normalize it 
        self.compute_embeddings()
        self.compute_similarity_matrix()       
        self.compute_normalized_scores()
    
    def extract_emotions(self, emotions_data):
        
        if not emotions_data:
            return []
        
        if isinstance(emotions_data, list):
            return [str(emotion) for emotion in emotions_data]
        
        if isinstance(emotions_data, dict):
            #extracting potential values from emotion list/dict in summary.json
            emotion_keys = [
                'conveyed_emotions', 'moods', 'conveyed', 'described_emotions', 
                'conveyed_feelings', 'initial_emotion', 'mid_emotion', 'final_emotion'
            ]
            
            for key in emotion_keys:
                if key in emotions_data:
                    # Handle both list and string cases
                    emotions = emotions_data[key]
                    if isinstance(emotions, list):
                        return [str(emotion) for emotion in emotions]
                    elif isinstance(emotions, str):
                        return [emotions]
        
        return []
    
    def load_data(self):
        with open(self.data_path, 'r') as file:
            self.raw_data = json.load(file)
        
        self.posts = self.raw_data.get('posts', [])
        

        self.video_texts = []
        self.video_emotions = []
        
        for post in self.posts:
            # Combine title and description for text
            text = f"{post.get('title', '')} {post.get('description', '')}"
            self.video_texts.append(text)
            
            # Extract emotions from post_summary if exists
            post_summary = post.get('post_summary', {})
            emotions = post_summary.get('emotions', post.get('emotions', []))
            
            # Extract emotions flexibly
            extracted_emotions = self.extract_emotions(emotions)
            self.video_emotions.append(extracted_emotions)
    
    def compute_embeddings(self):
        # Tokenize data
        tokenized_data = self.tokenizer(
            self.video_texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**tokenized_data)
        
        self.embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    def compute_similarity_matrix(self):
        self.similarity_matrix = cosine_similarity(self.embeddings)
    
    def compute_normalized_scores(self):
        # Normalize  metrics
        scaler = MinMaxScaler()
        #These metrics are the primary fields to consider for ground truth value as it reflects the engagement of the content.
        #Combining them offers a more strict evaluation for the predicted model. 
        view_counts = np.array([post.get('view_count', 0) for post in self.posts])
        upvote_counts = np.array([post.get('upvote_count', 0) for post in self.posts])
        average_ratings = np.array([post.get('average_rating', 0) for post in self.posts])
        share_counts = np.array([post.get('share_count', 0) for post in self.posts])
        
        # Combine normalized metrics
        self.combined_scores = (
            scaler.fit_transform(view_counts.reshape(-1, 1)) +
            scaler.fit_transform(upvote_counts.reshape(-1, 1)) +
            scaler.fit_transform(average_ratings.reshape(-1, 1)) +
            scaler.fit_transform(share_counts.reshape(-1, 1))
        ) / 4
    
    def get_mood_based_recommendations(self, mood=None, category_id=None, top_n=5):
        # If no mood is specified, return content-based recommendations
        if not mood:
            return self.get_content_based_recommendations(top_n)
        
        # Normalize mood input
        mood = mood.lower()
        
        # Find videos with matching mood
        mood_matched_videos = [
            idx for idx, emotions in enumerate(self.video_emotions) 
            if any(mood in str(emotion).lower() for emotion in emotions)
        ]
        
        # If no mood-matched videos, return content-based recommendations
        if not mood_matched_videos:
            return self.get_content_based_recommendations(top_n)
        
        # Get similarity scores for mood-matched videos
        mood_sim_scores = self.similarity_matrix[mood_matched_videos].mean(axis=0)
        
        # Sort and get top recommendations
        top_indices = np.argsort(mood_sim_scores)[::-1][:top_n]
        
        return [
            {
                'video_id': f"Video {idx+1}", 
                'title': str(self.posts[idx].get('title', '')),
                'emotions': [str(e) for e in self.video_emotions[idx]],
                'similarity_score': float(mood_sim_scores[idx])
            } for idx in top_indices
        ]
    
    def get_content_based_recommendations(self, top_n=5, base_video_index=0):
        # Top 5 similar content is provided.
        sim_scores = self.similarity_matrix[base_video_index]
        top_indices = np.argsort(sim_scores)[::-1][1:top_n+1]
        
        return [
            {
                'video_id': f"Video {idx+1}", 
                'title': str(self.posts[idx].get('title', '')),
                'emotions': [str(e) for e in self.video_emotions[idx]],
                'similarity_score': float(sim_scores[idx])
            } for idx in top_indices
        ]


recommendation_system = VideoRecommendationSystem()