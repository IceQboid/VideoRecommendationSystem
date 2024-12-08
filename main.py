from flask import Flask, request, jsonify
from recommendation_system import recommendation_system  # Import the recommendation system

app = Flask(__name__)

@app.route('/feed', methods=['GET'])
def get_recommendations():
    username = request.args.get('username')
    category_id = request.args.get('category_id')
    mood = request.args.get('mood')

    if not username:
        return jsonify({"error": "Username is required"}), 400

    try:        
        if mood and category_id:            
            recommendations = recommendation_system.get_mood_based_recommendations(mood=mood, category_id=category_id, top_n=10)
        elif mood:            
            recommendations = recommendation_system.get_mood_based_recommendations(mood=mood, top_n=10)
        elif category_id:            
            recommendations = recommendation_system.get_content_based_recommendations(top_n=10)
        else:           
            recommendations = recommendation_system.get_content_based_recommendations(top_n=10)
                    
        return jsonify({
            "username": username,
            "category_id": category_id,
            "mood": mood,
            "recommendations": recommendations
        })
    
    except Exception as e:       
        print(f"An error occurred: {str(e)}")
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)