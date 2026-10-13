"""
HiveFlow — Meal Planning with AI Support
NLP Similarity Recommendation API (TF-IDF + Cosine Similarity)

Content-based recommender using Scikit-learn TF-IDF vectorization and
linear kernel similarity, with protein/exact-match boosting.

Lightweight alternative / complementary endpoint to the deep learning API.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re
import os
import time
import socket

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def configure_ngrok():
    """Configure Ngrok from NGROK_AUTH_TOKEN env var (optional for local-only runs)."""
    ngrok_token = os.environ.get("NGROK_AUTH_TOKEN")
    if not ngrok_token:
        return False
    ngrok.set_auth_token(ngrok_token)
    return True


# Meal catalog used by the TF-IDF recommender
meals = pd.DataFrame({
    'meal_id': list(range(1, 51)),
    'name': [
        'Vegetable Curry', 'Beef Burger', 'Greek Salad', 'Tofu Stir Fry', 'Grilled Chicken Breast',
        'Vegan Buddha Bowl', 'Spaghetti Bolognese', 'Quinoa Avocado Salad', 'Fish Tacos', 'Chickpea Stew',
        'Cheese Pizza', 'Lentil Soup', 'BBQ Ribs', 'Mushroom Risotto', 'Falafel Wrap',
        'Cauliflower Tacos', 'Turkey Sandwich', 'Vegan Chili', 'Chicken Caesar Salad', 'Stuffed Peppers',
        'Eggplant Parmesan', 'Shrimp Pad Thai', 'Black Bean Burrito', 'Beef Stroganoff', 'Caprese Salad',
        'Vegan Mac and Cheese', 'Grilled Salmon', 'Vegetarian Sushi', 'Pulled Pork Sandwich', 'Avocado Toast',
        'Zucchini Noodles', 'Meatball Sub', 'Tofu Curry', 'Cobb Salad', 'Vegan Pancakes',
        'Teriyaki Chicken Bowl', 'Sweet Potato Fries', 'Tomato Basil Soup', 'Chicken Tikka Masala', 'Veggie Lasagna',
        'Soba Noodle Salad', 'Hot Dog', 'Kale Smoothie Bowl', 'Salmon Poke Bowl', 'Egg Salad Sandwich',
        'Pesto Pasta', 'Beef Kebab', 'Miso Soup', 'Fried Rice', 'Veggie Omelette'
    ],
    'tags': [
        'vegetable curry vegetarian vegan healthy spicy indian',
        'beef burger meat fast-food burger beef patty bun',
        'greek salad vegetarian healthy salad feta olive',
        'tofu stir fry vegetarian vegan asian tofu',
        'grilled chicken breast meat healthy low-carb grilled chicken',
        'vegan buddha bowl vegan gluten-free healthy bowl grain',
        'spaghetti bolognese meat pasta italian beef minced',
        'quinoa avocado salad vegetarian healthy gluten-free avocado grain',
        'fish tacos meat seafood mexican spicy fish',
        'chickpea stew vegan vegetarian protein-rich stew chickpea',
        'cheese pizza vegetarian cheese italian fast-food pizza',
        'lentil soup vegetarian vegan protein-rich soup lentil',
        'bbq ribs meat bbq pork high-calorie ribs',
        'mushroom risotto vegetarian creamy italian rice mushroom',
        'falafel wrap vegetarian vegan middle-eastern wrap falafel',
        'cauliflower tacos vegetarian vegan gluten-free mexican cauliflower',
        'turkey sandwich meat sandwich low-fat lunch turkey',
        'vegan chili vegan healthy spicy beans chili',
        'chicken caesar salad meat salad healthy low-carb chicken caesar',
        'stuffed peppers vegetarian healthy stuffed spicy pepper',
        'eggplant parmesan vegetarian cheese italian baked eggplant',
        'shrimp pad thai seafood asian thai noodles shrimp',
        'black bean burrito vegetarian vegan mexican beans burrito',
        'beef stroganoff meat creamy pasta russian beef',
        'caprese salad vegetarian healthy italian salad mozzarella tomato',
        'vegan mac and cheese vegan creamy pasta comfort cheese',
        'grilled salmon seafood grilled healthy omega-3 salmon',
        'vegetarian sushi vegetarian sushi asian low-calorie',
        'pulled pork sandwich meat pork fast-food sandwich pulled',
        'avocado toast vegetarian healthy avocado toast bread',
        'zucchini noodles vegetarian low-carb gluten-free zucchini noodle',
        'meatball sub meat sandwich italian sub meatball',
        'tofu curry vegan vegetarian curry spicy tofu',
        'cobb salad meat salad egg bacon chicken',
        'vegan pancakes vegan breakfast dessert sweet pancake',
        'teriyaki chicken bowl meat asian grilled teriyaki chicken rice bowl',
        'sweet potato fries vegetarian snack baked sweet-potato fries',
        'tomato basil soup vegetarian vegan soup tomato herb',
        'chicken tikka masala meat indian spicy curry chicken',
        'veggie lasagna vegetarian cheese pasta italian lasagna',
        'soba noodle salad vegan noodles japanese salad soba',
        'hot dog meat fast-food grilled sausage hotdog',
        'kale smoothie bowl vegan smoothie healthy breakfast bowl kale',
        'salmon poke bowl seafood asian raw healthy bowl salmon',
        'egg salad sandwich vegetarian egg sandwich lunch salad',
        'pesto pasta vegetarian pasta italian basil pesto',
        'beef kebab meat grilled kebab middle-eastern beef',
        'miso soup vegan asian soup broth miso',
        'fried rice vegetarian asian rice soy-sauce',
        'veggie omelette vegetarian egg breakfast protein omelet'
    ],
    'main_protein': [
        None, 'beef', None, 'tofu', 'chicken',
        None, 'beef', None, 'fish', 'chickpea',
        'cheese', 'lentil', 'pork', None, 'falafel',
        'cauliflower', 'turkey', None, 'chicken', None,
        None, 'shrimp', 'black bean', 'beef', None,
        None, 'salmon', None, 'pork', None,
        None, 'meatball', 'tofu', 'chicken', None,
        'chicken', None, None, 'chicken', None,
        None, 'sausage', None, 'salmon', 'egg',
        None, 'beef', None, None, 'egg'
    ]
})

def get_boosted_recommendations(preference, n=5, protein_boost=2.0, exact_boost=1.5):
    """Get recommendations with protein and exact match boosting"""

    # Preprocess preference to identify main protein
    pref_protein = None
    proteins = ['chicken', 'beef', 'pork', 'fish', 'tofu', 'turkey', 'salmon', 'shrimp']
    for p in proteins:
        if p in preference.lower():
            pref_protein = p
            break

    # Create boosted tags
    boosted_tags = []
    for idx, row in meals.iterrows():
        tags = row['tags']

        # Add protein boost if matches preference protein
        if pref_protein and row['main_protein'] == pref_protein:
            tags += f" {pref_protein}" * 3  # Add extra weight to matching protein

        # Add exact name boost
        if preference.lower() in row['name'].lower():
            tags += f" {preference.lower()}" * 2

        boosted_tags.append(tags)

    # Vectorize
    tfidf = TfidfVectorizer(
        stop_words='english',
        analyzer='word',
        token_pattern=r'(?u)\b\w+\b',
        ngram_range=(1, 2)
    )

    tfidf_matrix = tfidf.fit_transform(boosted_tags)
    pref_vector = tfidf.transform([preference])

    # Calculate similarity
    cosine_sim = linear_kernel(pref_vector, tfidf_matrix)

    # Get top recommendations
    recommendations = list(enumerate(cosine_sim[0]))
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)

    # Prepare results
    results = []
    for i, score in recommendations[:n]:
        results.append({
            'name': meals.iloc[i]['name'],
            'score': round(score, 4),
            'protein': meals.iloc[i]['main_protein'],
            'tags': meals.iloc[i]['tags']
        })

    return results

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'HiveFlow NLP Similarity Meal Recommender API is running',
        'endpoints': {
            '/recommendations': 'POST request with preference and count'
        }
    })

@app.route('/recommendations', methods=['GET', 'POST'])
def get_recommendations():
    if request.method == 'GET':
        return jsonify({
            'status': 'Please send a POST request with preference and count'
        })

    data = request.get_json()
    if not data:
        preference = request.form.get('preference', '')
        count = int(request.form.get('count', 5))
    else:
        preference = data.get('preference', '')
        count = data.get('count', 5)

    if not preference:
        return jsonify({
            'success': False,
            'error': 'No preference specified'
        }), 400

    try:
        recommendations = get_boosted_recommendations(preference, n=count)
        return jsonify({
            'success': True,
            'query': preference,
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", find_free_port()))
    use_ngrok = configure_ngrok()

    if use_ngrok:
        public_url = ngrok.connect(port)
        print(f"Ngrok URL: {public_url}")
        print("Use this public URL from PHP/cPanel backend integration.")
    else:
        print("NGROK_AUTH_TOKEN not set — running locally only.")
        print(f"Local URL: http://127.0.0.1:{port}/recommendations")

    app.run(port=port, debug=False)

