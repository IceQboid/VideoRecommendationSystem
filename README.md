# Video Recommendation System
A content-based recommendation system that can tackle cold start problem with mood preference. 

## Description

Text Embedding with DistilBERT:
Use DistilBERT to generate embeddings for video titles and descriptions.

Similarity-Based Recommendations:
Compute a cosine similarity matrix to recommend videos based on content similarity.

Mood-Based Filtering:
Recommend videos that match a specific mood using extracted emotions from the data.

Evaluation:
Evaluate the relevance of recommendations using F1 score (based on top-N recommendations and a threshold).

## Getting Started

### Dependencies

* Install the required dependencies
``` 
pip install -r requirements.txt 
```

* Postman API


### Executing program

* We need to run the data fetch api to get the necessary data

```
python fetch_data.py
```

*The next step is to run the recommendation model that features DistilBERT along with mood filtering and cosine similarity:

```
python recommendation_system.py
```

* Make sure to include only the necessary json file for the creation of the recommendation system.


* Check if the system is working as intended with:

```
python unit_test.py
```

* Evaluation of the model:
```
python evaluation.py
```

* Now run the model with: 

```
python main.py
```

This starts the Flask App 

* Run POSTMAN to test endpoint methods:
* Use GET Method. 
* Pass your endpoint and get the top 10 results.


