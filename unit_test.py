import json
import sys
import os


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommendation_system import recommendation_system

def print_recommendations(recommendations, title):
    print(f"\n{title}")
    print("-" * 50)
    for idx, rec in enumerate(recommendations, 1):
        print(f"{idx}. Video ID: {rec['video_id']}")
        print(f"   Title: {rec['title']}")
        print(f"   Emotions: {rec['emotions']}")
        print(f"   Similarity Score: {rec['similarity_score']:.4f}")
        print()

def test_recommendation_system():
    
    print("Test 1: Content-based Recommendations")
    content_recs = recommendation_system.get_content_based_recommendations(top_n=3)
    print_recommendations(content_recs, "Content-Based Recommendations")

   
    print("Test 2: Mood 1")
    joy_recs = recommendation_system.get_mood_based_recommendations(mood='joy', top_n=3)
    print_recommendations(joy_recs, "Joy-based Recommendations")

    
    print("Test 3: Mood 2")
    determination_recs = recommendation_system.get_mood_based_recommendations(mood='determination', top_n=3)
    print_recommendations(determination_recs, "Determination-based Recommendations")

   
    print("Test 4: Mood 3")
    weird_mood_recs = recommendation_system.get_mood_based_recommendations(mood='excitement', top_n=3)
    print_recommendations(weird_mood_recs, "Excitement-based Recommendations (Fallback to Content-Based)")

    

if __name__ == '__main__':
    test_recommendation_system()