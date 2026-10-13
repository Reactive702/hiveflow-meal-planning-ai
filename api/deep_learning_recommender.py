"""
HiveFlow — Meal Planning with AI Support
Deep Learning Recommendation API (Multi-input Neural Network)

Binary classification model that scores meal candidates for a user based on:
- nutrition history (calories, protein)
- meal category / dietary constraints
- tag embeddings

Exposes Flask-CORS REST endpoints for PHP/MySQL backend integration (via Ngrok).
"""
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from collections import defaultdict, Counter
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split # Needed for train_model
# matplotlib.pyplot is not strictly needed for the API
# import matplotlib.pyplot as plt
import os
import time
import socket

from flask import Flask, jsonify, request
from flask_cors import CORS
from pyngrok import ngrok

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- Data Definitions (from Model 2) ---
vegan_meal_plan = [
    # ... (rest of the vegan_meal_plan list) ...
    {'date': '01.03.2025', 'breakfast': 'Avocado Toast', 'lunch': 'Vegan Buddha Bowl', 'dinner': 'Tofu Stir Fry'},
    {'date': '02.03.2025', 'breakfast': 'Kale Smoothie Bowl', 'lunch': 'Chickpea Stew', 'dinner': 'Mushroom Risotto'},
    {'date': '03.03.2025', 'breakfast': 'Vegan Pancakes', 'lunch': 'Falafel Wrap', 'dinner': 'Vegetable Curry'},
    {'date': '04.03.2025', 'breakfast': 'Avocado Toast', 'lunch': 'Quinoa Avocado Salad', 'dinner': 'Vegan Chili'},
    {'date': '05.03.2025', 'breakfast': 'Kale Smoothie Bowl', 'lunch': 'Lentil Soup', 'dinner': 'Black Bean Burrito'},
    {'date': '06.03.2025', 'breakfast': 'Vegan Pancakes', 'lunch': 'Cauliflower Tacos', 'dinner': 'Vegan Mac and Cheese'},
    {'date': '07.03.2025', 'breakfast': 'Avocado Toast', 'lunch': 'Vegetarian Sushi', 'dinner': 'Stuffed Peppers'},
    {'date': '08.03.2025', 'breakfast': 'Kale Smoothie Bowl', 'lunch': 'Tomato Basil Soup', 'dinner': 'Zucchini Noodles'},
    {'date': '09.03.2025', 'breakfast': 'Vegan Pancakes', 'lunch': 'Veggie Lasagna', 'dinner': 'Tofu Curry'},
    {'date': '10.03.2025', 'breakfast': 'Avocado Toast', 'lunch': 'Soba Noodle Salad', 'dinner': 'Fried Rice'},
    {'date': '11.03.2025', 'breakfast': 'Kale Smoothie Bowl', 'lunch': 'Miso Soup', 'dinner': 'Vegan Buddha Bowl'},
    {'date': '12.03.2025', 'breakfast': 'Vegan Pancakes', 'lunch': 'Vegetable Curry', 'dinner': 'Chickpea Stew'},
    {'date': '13.03.2025', 'breakfast': 'Avocado Toast', 'lunch': 'Falafel Wrap', 'dinner': 'Mushroom Risotto'},
    {'date': '14.03.2025', 'breakfast': 'Kale Smoothie Bowl', 'lunch': 'Quinoa Avocado Salad', 'dinner': 'Vegan Chili'},
    {'date': '15.03.2025', 'breakfast': 'Vegan Pancakes', 'lunch': 'Lentil Soup', 'dinner': 'Black Bean Burrito'},
    {'date': '16.03.2025', 'breakfast': 'Avocado Toast', 'lunch': 'Cauliflower Tacos', 'dinner': 'Vegan Mac and Cheese'},
    {'date': '17.03.2025', 'breakfast': 'Kale Smoothie Bowl', 'lunch': 'Vegetarian Sushi', 'dinner': 'Stuffed Peppers'},
    {'date': '18.03.2025', 'breakfast': 'Vegan Pancakes', 'lunch': 'Tomato Basil Soup', 'dinner': 'Zucchini Noodles'},
    {'date': '19.03.2025', 'breakfast': 'Avocado Toast', 'lunch': 'Veggie Lasagna', 'dinner': 'Tofu Curry'},
    {'date': '20.03.2025', 'breakfast': 'Kale Smoothie Bowl', 'lunch': 'Soba Noodle Salad', 'dinner': 'Fried Rice'},
    {'date': '21.03.2025', 'breakfast': 'Vegan Pancakes', 'lunch': 'Miso Soup', 'dinner': 'Vegan Buddha Bowl'},
    {'date': '22.03.2025', 'breakfast': 'Avocado Toast', 'lunch': 'Vegetable Curry', 'dinner': 'Chickpea Stew'},
    {'date': '23.03.2025', 'breakfast': 'Kale Smoothie Bowl', 'lunch': 'Falafel Wrap', 'dinner': 'Mushroom Risotto'},
    {'date': '24.03.2025', 'breakfast': 'Vegan Pancakes', 'lunch': 'Quinoa Avocado Salad', 'dinner': 'Vegan Chili'},
    {'date': '25.03.2025', 'breakfast': 'Avocado Toast', 'lunch': 'Lentil Soup', 'dinner': 'Black Bean Burrito'},
    {'date': '26.03.2025', 'breakfast': 'Kale Smoothie Bowl', 'lunch': 'Cauliflower Tacos', 'dinner': 'Vegan Mac and Cheese'},
    {'date': '27.03.2025', 'breakfast': 'Vegan Pancakes', 'lunch': 'Vegetarian Sushi', 'dinner': 'Stuffed Peppers'},
    {'date': '28.03.2025', 'breakfast': 'Avocado Toast', 'lunch': 'Tomato Basil Soup', 'dinner': 'Zucchini Noodles'},
    {'date': '29.03.2025', 'breakfast': 'Kale Smoothie Bowl', 'lunch': 'Veggie Lasagna', 'dinner': 'Tofu Curry'},
    {'date': '30.03.2025', 'breakfast': 'Vegan Pancakes', 'lunch': 'Soba Noodle Salad', 'dinner': 'Fried Rice'},
    {'date': '31.03.2025', 'breakfast': 'Avocado Toast', 'lunch': 'Miso Soup', 'dinner': 'Vegan Buddha Bowl'},
    # Nisan 2025
    {'date': '01.04.2025', 'breakfast': 'Kale Smoothie Bowl', 'lunch': 'Vegetable Curry', 'dinner': 'Chickpea Stew'},
    {'date': '02.04.2025', 'breakfast': 'Vegan Pancakes', 'lunch': 'Falafel Wrap', 'dinner': 'Mushroom Risotto'},
    {'date': '03.04.2025', 'breakfast': 'Avocado Toast', 'lunch': 'Quinoa Avocado Salad', 'dinner': 'Vegan Chili'},
    {'date': '04.04.2025', 'breakfast': 'Kale Smoothie Bowl', 'lunch': 'Lentil Soup', 'dinner': 'Black Bean Burrito'},
    {'date': '05.04.2025', 'breakfast': 'Vegan Pancakes', 'lunch': 'Cauliflower Tacos', 'dinner': 'Vegan Mac and Cheese'},
    {'date': '06.04.2025', 'breakfast': 'Avocado Toast', 'lunch': 'Vegetarian Sushi', 'dinner': 'Stuffed Peppers'},
    {'date': '07.04.2025', 'breakfast': 'Kale Smoothie Bowl', 'lunch': 'Tomato Basil Soup', 'dinner': 'Zucchini Noodles'},
    {'date': '08.04.2025', 'breakfast': 'Vegan Pancakes', 'lunch': 'Veggie Lasagna', 'dinner': 'Tofu Curry'},
    {'date': '09.04.2025', 'breakfast': 'Avocado Toast', 'lunch': 'Soba Noodle Salad', 'dinner': 'Fried Rice'},
    {'date': '10.04.2025', 'breakfast': 'Kale Smoothie Bowl', 'lunch': 'Miso Soup', 'dinner': 'Vegan Buddha Bowl'},
    {'date': '11.04.2025', 'breakfast': 'Vegan Pancakes', 'lunch': 'Vegetable Curry', 'dinner': 'Chickpea Stew'},
    {'date': '12.04.2025', 'breakfast': 'Avocado Toast', 'lunch': 'Falafel Wrap', 'dinner': 'Mushroom Risotto'},
    {'date': '13.04.2025', 'breakfast': 'Kale Smoothie Bowl', 'lunch': 'Quinoa Avocado Salad', 'dinner': 'Vegan Chili'},
    {'date': '14.04.2025', 'breakfast': 'Vegan Pancakes', 'lunch': 'Lentil Soup', 'dinner': 'Black Bean Burrito'},
    {'date': '15.04.2025', 'breakfast': 'Avocado Toast', 'lunch': 'Cauliflower Tacos', 'dinner': 'Vegan Mac and Cheese'},
    {'date': '16.04.2025', 'breakfast': 'Kale Smoothie Bowl', 'lunch': 'Vegetarian Sushi', 'dinner': 'Stuffed Peppers'},
    {'date': '17.04.2025', 'breakfast': 'Vegan Pancakes', 'lunch': 'Tomato Basil Soup', 'dinner': 'Zucchini Noodles'},
    {'date': '18.04.2025', 'breakfast': 'Avocado Toast', 'lunch': 'Veggie Lasagna', 'dinner': 'Tofu Curry'},
    {'date': '19.04.2025', 'breakfast': 'Kale Smoothie Bowl', 'lunch': 'Soba Noodle Salad', 'dinner': 'Fried Rice'},
    {'date': '20.04.2025', 'breakfast': 'Vegan Pancakes', 'lunch': 'Miso Soup', 'dinner': 'Vegan Buddha Bowl'},
    {'date': '21.04.2025', 'breakfast': 'Avocado Toast', 'lunch': 'Vegetable Curry', 'dinner': 'Chickpea Stew'},
    {'date': '22.04.2025', 'breakfast': 'Kale Smoothie Bowl', 'lunch': 'Falafel Wrap', 'dinner': 'Mushroom Risotto'},
    {'date': '23.04.2025', 'breakfast': 'Vegan Pancakes', 'lunch': 'Quinoa Avocado Salad', 'dinner': 'Vegan Chili'},
    {'date': '24.04.2025', 'breakfast': 'Avocado Toast', 'lunch': 'Lentil Soup', 'dinner': 'Black Bean Burrito'},
    {'date': '25.04.2025', 'breakfast': 'Kale Smoothie Bowl', 'lunch': 'Cauliflower Tacos', 'dinner': 'Vegan Mac and Cheese'},
    {'date': '26.04.2025', 'breakfast': 'Vegan Pancakes', 'lunch': 'Vegetarian Sushi', 'dinner': 'Stuffed Peppers'},
    {'date': '27.04.2025', 'breakfast': 'Avocado Toast', 'lunch': 'Tomato Basil Soup', 'dinner': 'Zucchini Noodles'},
    {'date': '28.04.2025', 'breakfast': 'Kale Smoothie Bowl', 'lunch': 'Veggie Lasagna', 'dinner': 'Tofu Curry'},
    {'date': '29.04.2025', 'breakfast': 'Vegan Pancakes', 'lunch': 'Soba Noodle Salad', 'dinner': 'Fried Rice'},
    {'date': '30.04.2025', 'breakfast': 'Avocado Toast', 'lunch': 'Miso Soup', 'dinner': 'Vegan Buddha Bowl'}
]

meals_data = {
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
    'category': [
        'vegetarian', 'meat', 'vegetarian', 'vegan', 'meat', 'vegan', 'meat', 'vegetarian', 'meat', 'vegan',
        'vegetarian', 'vegan', 'meat', 'vegetarian', 'vegan', 'vegan', 'meat', 'vegan', 'meat', 'vegetarian',
        'vegetarian', 'meat', 'vegan', 'meat', 'vegetarian', 'vegan', 'meat', 'vegetarian', 'meat', 'vegetarian',
        'vegetarian', 'meat', 'vegan', 'meat', 'vegan', 'meat', 'vegetarian', 'vegetarian', 'meat', 'vegetarian',
        'vegan', 'meat', 'vegan', 'meat', 'vegetarian', 'vegetarian', 'meat', 'vegan', 'vegetarian', 'vegetarian'
    ],
    'calories': [
        350, 650, 250, 300, 400, 450, 600, 320, 380, 280, 550, 220, 800, 420, 380,
        320, 450, 350, 400, 280, 420, 580, 400, 650, 220, 450, 420, 300, 550, 280,
        180, 600, 350, 450, 380, 550, 320, 180, 600, 450, 320, 450, 280, 420, 350,
        450, 550, 180, 380, 320
    ],
    'protein': [
        12, 35, 8, 15, 40, 15, 30, 10, 25, 15, 20, 12, 45, 8, 12,
        8, 25, 15, 30, 8, 12, 25, 15, 35, 12, 15, 25, 8, 30, 8,
        5, 30, 15, 25, 8, 35, 3, 5, 35, 15, 8, 20, 5, 25, 15,
        12, 30, 5, 8, 15
    ],
    'carbs': [ 45, 40, 15, 25, 5, 60, 75, 35, 30, 35, 60, 30, 15, 55, 45, 30, 40, 40, 15, 20, 40, 65, 50, 45, 10, 50, 5, 55, 45, 35, 15, 55, 30, 10, 60, 50, 35, 20, 45, 50, 45, 30, 40, 25, 20, 50, 15, 20, 60, 10 ],
    'fat': [ 12, 35, 15, 10, 20, 15, 25, 15, 15, 8, 25, 5, 55, 15, 15, 15, 20, 10, 25, 15, 20, 20, 10, 40, 15, 20, 25, 5, 30, 10, 10, 30, 15, 35, 10, 20, 15, 8, 35, 20, 10, 25, 10, 15, 20, 20, 35, 5, 10, 20 ],
    'dietary': [
        'vegetarian', 'normal', 'vegetarian', 'vegan', 'low-carb', 'vegan', 'normal', 'gluten-free', 'normal', 'vegan',
        'vegetarian', 'vegan', 'normal', 'vegetarian', 'vegan', 'vegan gluten-free', 'normal', 'vegan', 'low-carb', 'vegetarian',
        'vegetarian', 'normal', 'vegan', 'normal', 'vegetarian gluten-free', 'vegan', 'low-carb', 'vegetarian', 'normal', 'vegetarian',
        'vegetarian low-carb', 'normal', 'vegan', 'low-carb', 'vegan', 'normal', 'vegetarian', 'vegetarian vegan', 'normal', 'vegetarian',
        'vegan gluten-free', 'normal', 'vegan gluten-free', 'low-carb', 'vegetarian', 'vegetarian', 'normal', 'vegan gluten-free', 'vegetarian', 'vegetarian low-carb'
    ],
    'tags': [
        'spicy indian curry vegetable', 'beef burger meat fast-food', 'salad feta olive greek', 'tofu stir-fry asian vegan', 'chicken grilled healthy',
        'bowl grain vegan healthy', 'pasta italian beef', 'quinoa salad healthy', 'fish mexican tacos', 'chickpea stew vegan',
        'pizza cheese italian', 'lentil soup vegan', 'bbq pork ribs', 'mushroom risotto italian', 'falafel wrap middle-eastern vegan',
        'cauliflower tacos vegan mexican', 'turkey sandwich lunch', 'chili vegan spicy', 'chicken salad caesar', 'peppers stuffed vegetable',
        'eggplant parmesan italian', 'shrimp thai noodles', 'black bean burrito mexican vegan', 'beef stroganoff russian', 'salad mozzarella tomato',
        'mac cheese vegan', 'salmon grilled healthy', 'sushi vegetarian asian', 'pork sandwich pulled', 'avocado toast healthy breakfast',
        'zucchini noodles low-carb healthy vegetarian', 'meatball sub italian', 'tofu curry spicy vegan', 'salad egg bacon',
        'pancakes vegan sweet breakfast', 'chicken teriyaki asian', 'sweet potato fries snack', 'tomato soup basil vegan',
        'chicken curry indian', 'lasagna vegetarian italian', 'soba noodles japanese vegan', 'hot dog fast-food',
        'smoothie bowl healthy breakfast vegan', 'salmon poke asian', 'egg sandwich salad', 'pasta pesto italian',
        'beef kebab grilled', 'miso soup japanese vegan', 'rice asian fried vegetarian', 'omelette egg breakfast'
    ]
}
user_preferences = { # This is the global user_preferences for the API
    'calorie_range': (300, 700),
    'protein_range': (10, 50),
    'allowed_categories': ['vegan', 'vegetarian']
}

# --- Global Variables for API ---
meals_df = None
user_history_df = None # This will be the fixed history for user_id = 1
calorie_scaler = MinMaxScaler()
protein_scaler = MinMaxScaler()
category_encoder = LabelEncoder()
tag_list = [] # Populated by prepare_meal_features
tag_index = {} # Populated by prepare_meal_features
num_tags = 0 # Populated by prepare_meal_features
num_user_features = 0 # Set in initialize_app_globals after prepare_meal_features
keras_model = None # Trained model will be stored here

# Globals for precomputed meal features (populated by prepare_meal_features)
meal_features_nutritional_global = None
categories_one_hot_global = None
tag_features_global = None


# --- Model Functions (from Model 2) ---
def prepare_meal_features(current_meals_df):
    global category_encoder, tag_list, tag_index, num_tags, meal_features_nutritional_global, categories_one_hot_global, tag_features_global

    meal_features_nutritional_global = current_meals_df[['calories_norm', 'protein_norm']].values

    categories_encoded = category_encoder.fit_transform(current_meals_df['category'])
    categories_one_hot_global = tf.keras.utils.to_categorical(categories_encoded, num_classes=len(category_encoder.classes_))

    all_tags_set = set()
    for tags_str in current_meals_df['tags']:
        if isinstance(tags_str, str):
             all_tags_set.update(tags_str.split())

    tag_list.clear() # Ensure it's fresh if called multiple times (though not in this setup)
    tag_list.extend(sorted(list(all_tags_set)))

    tag_index.clear()
    for idx, tag_item in enumerate(tag_list):
        tag_index[tag_item] = idx
    num_tags = len(tag_list)

    tag_features_list = []
    for tags_str in current_meals_df['tags']:
        feature = np.zeros(num_tags)
        if isinstance(tags_str, str):
            for tag_item in tags_str.split():
                if tag_item in tag_index:
                    feature[tag_index[tag_item]] = 1
        tag_features_list.append(feature)
    tag_features_global = np.array(tag_features_list)
    print(f"Meal features prepared: {num_tags} unique tags found.")

def calculate_user_features(history_subset_df, current_meals_df, current_tag_index):
    global num_tags # Uses global num_tags set by prepare_meal_features

    if history_subset_df.empty:
        return np.concatenate(([0.5, 0.5], np.zeros(num_tags)))

    consumed_meal_ids = history_subset_df['meal_id'].tolist()
    consumed_meals_features = current_meals_df[current_meals_df['meal_id'].isin(consumed_meal_ids)]

    if consumed_meals_features.empty:
         return np.concatenate(([0.5, 0.5], np.zeros(num_tags)))

    avg_calories_norm = consumed_meals_features['calories_norm'].mean()
    avg_protein_norm = consumed_meals_features['protein_norm'].mean()

    tag_preference_vector = np.zeros(num_tags)
    tag_counts_counter = Counter()
    total_tags_consumed = 0
    for tags_str in consumed_meals_features['tags']:
        if isinstance(tags_str, str):
            tags_items = tags_str.split()
            for tag_item in tags_items:
                if tag_item in current_tag_index:
                    tag_counts_counter[tag_item] += 1
                    total_tags_consumed += 1

    if total_tags_consumed > 0:
        for tag_item, count in tag_counts_counter.items():
            if tag_item in current_tag_index:
                tag_preference_vector[current_tag_index[tag_item]] = count / total_tags_consumed
    return np.concatenate(([avg_calories_norm, avg_protein_norm], tag_preference_vector))

def create_recommendation_model(num_categories_local, num_tags_local, num_user_features_local, embedding_dim=64, l2_reg=0.001):
    user_input = layers.Input(shape=(num_user_features_local,), name='user_input')
    meal_input = layers.Input(shape=(2,), name='meal_input') # calories_norm, protein_norm
    category_input = layers.Input(shape=(num_categories_local,), name='category_input')
    tag_input = layers.Input(shape=(num_tags_local,), name='tag_input')

    user_processed = layers.Dense(embedding_dim, activation='relu', kernel_regularizer=l2(l2_reg))(user_input)
    user_processed = layers.Dropout(0.3)(user_processed)
    meal_processed = layers.Dense(embedding_dim, activation='relu', kernel_regularizer=l2(l2_reg))(meal_input)
    meal_processed = layers.Dropout(0.3)(meal_processed)
    tag_processed = layers.Dense(embedding_dim, activation='relu', kernel_regularizer=l2(l2_reg))(tag_input)
    tag_processed = layers.Dropout(0.3)(tag_processed)

    concat = layers.Concatenate()([user_processed, meal_processed, category_input, tag_processed])
    dense1 = layers.Dense(128, activation='relu', kernel_regularizer=l2(l2_reg))(concat)
    dense1 = layers.Dropout(0.4)(dense1)
    dense2 = layers.Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(dense1)
    dense2 = layers.Dropout(0.3)(dense2)
    output = layers.Dense(1, activation='sigmoid', name='output')(dense2)

    model = Model(inputs=[user_input, meal_input, category_input, tag_input], outputs=output)
    return model

def train_model_on_startup(current_user_history_df, current_meals_df, current_user_preferences):
    global num_user_features, tag_index, num_tags # Use already prepared globals
    global meal_features_nutritional_global, categories_one_hot_global, tag_features_global # Use precomputed features

    # These globals are set by prepare_meal_features called in initialize_app_globals
    # So, meal_features_nutritional, categories_one_hot, tag_features are already available globally.

    num_categories_local = categories_one_hot_global.shape[1]
    num_meals_local = len(current_meals_df)
    # num_user_features is already global and set

    overall_user_features = calculate_user_features(current_user_history_df, current_meals_df, tag_index)

    positive_samples = []
    for t in range(len(current_user_history_df)):
        history_so_far = current_user_history_df.iloc[:t]
        current_interaction = current_user_history_df.iloc[t]
        # meals_df meal_id is 1-based, indices are 0-based
        meal_idx = int(current_interaction['meal_id']) - 1

        if not (0 <= meal_idx < num_meals_local):
            print(f"Skipping invalid meal_idx: {meal_idx} from meal_id {current_interaction['meal_id']}")
            continue

        user_features_for_sample = calculate_user_features(history_so_far, current_meals_df, tag_index)
        positive_samples.append({
            'user': user_features_for_sample,
            'meal': meal_features_nutritional_global[meal_idx],
            'category': categories_one_hot_global[meal_idx],
            'tags': tag_features_global[meal_idx],
            'label': 1
        })

    negative_samples = []
    user_meal_indices_eaten = set(current_user_history_df['meal_id'] -1) # 0-based
    allowed_categories_local = current_user_preferences.get('allowed_categories', ['vegan', 'vegetarian'])

    candidate_negative_meals = current_meals_df[
        (current_meals_df['category'].isin(allowed_categories_local)) &
        (~current_meals_df.index.isin(user_meal_indices_eaten)) # meals_df.index is 0-based
    ]
    possible_negative_meal_indices = candidate_negative_meals.index.tolist()

    num_positive = len(positive_samples)
    num_negative_to_generate = 0
    if not possible_negative_meal_indices:
        print("Warning: User has eaten all meals within allowed categories. Cannot generate suitable negative samples.")
    elif num_positive == 0:
        print("Warning: No positive samples generated. Cannot generate negative samples.")
    else:
        num_negative_to_generate = num_positive

    if possible_negative_meal_indices and num_negative_to_generate > 0:
        num_negative_to_generate = min(num_negative_to_generate, len(possible_negative_meal_indices))
        selected_negative_indices = random.sample(possible_negative_meal_indices, num_negative_to_generate)
        for neg_meal_idx in selected_negative_indices:
            negative_samples.append({
                'user': overall_user_features,
                'meal': meal_features_nutritional_global[neg_meal_idx],
                'category': categories_one_hot_global[neg_meal_idx],
                'tags': tag_features_global[neg_meal_idx],
                'label': 0
            })
    else:
        print("Skipping negative sample generation or no candidates.")

    all_samples = positive_samples + negative_samples
    if not all_samples:
        print("Error: No training samples generated. Cannot train model.")
        return None, None

    random.shuffle(all_samples)
    X_user = np.array([sample['user'] for sample in all_samples])
    X_meal = np.array([sample['meal'] for sample in all_samples])
    X_category = np.array([sample['category'] for sample in all_samples])
    X_tags = np.array([sample['tags'] for sample in all_samples])
    y = np.array([sample['label'] for sample in all_samples])

    # Use global num_user_features, num_tags which are set based on prepare_meal_features
    model_to_train = create_recommendation_model(num_categories_local, num_tags, num_user_features)
    model_to_train.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    validation_data_for_fit = None
    X_user_train, X_meal_train, X_category_train, X_tags_train, y_train = X_user, X_meal, X_category, X_tags, y

    if len(all_samples) >= 10 and len(np.unique(y)) > 1: # Ensure enough samples and at least two classes for stratification
        X_user_train, X_user_test, X_meal_train, X_meal_test, X_category_train, X_category_test, X_tags_train, X_tags_test, y_train, y_test = train_test_split(
            X_user, X_meal, X_category, X_tags, y, test_size=0.2, random_state=42, stratify=y
        )
        validation_data_for_fit = ([X_user_test, X_meal_test, X_category_test, X_tags_test], y_test)
    else:
        print("Warning: Too few samples or only one class for train-test split. Training on all data without validation split.")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss' if validation_data_for_fit else 'loss',
        patience=10, restore_best_weights=True, verbose=1
    )

    print(f"--- Starting Model Training (on API startup) ---")
    print(f"Number of user features: {num_user_features}, Num tags: {num_tags}, Num categories: {num_categories_local}")
    print(f"Training samples: {len(y_train)}, Validation samples: {len(y_test) if validation_data_for_fit else 0}")

    training_history = model_to_train.fit(
        [X_user_train, X_meal_train, X_category_train, X_tags_train],
        y_train,
        epochs=50, # Reduced epochs for faster startup, can be tuned
        batch_size=32,
        validation_data=validation_data_for_fit,
        callbacks=[early_stopping],
        verbose=1 # Set to 0 or 2 for less output during startup if desired
    )
    print("--- Model Training Finished (on API startup) ---")
    return model_to_train, training_history


def get_recommendations_api(loaded_model, current_user_history_df, current_meals_df, current_user_preferences, n_recommendations=5):
    global meal_features_nutritional_global, categories_one_hot_global, tag_features_global, tag_index

    if current_user_history_df.empty:
        return []

    num_all_meals = len(current_meals_df)
    overall_user_features = calculate_user_features(current_user_history_df, current_meals_df, tag_index)
    user_input_for_prediction = overall_user_features.reshape(1, -1)
    user_input_repeated = np.repeat(user_input_for_prediction, num_all_meals, axis=0)

    predictions_all = loaded_model.predict([
        user_input_repeated, meal_features_nutritional_global,
        categories_one_hot_global, tag_features_global
    ])

    predictions_list = [(i, predictions_all[i][0]) for i in range(num_all_meals)]
    predictions_list.sort(key=lambda x: x[1], reverse=True)

    recommendations_output = []
    added_meal_indices = set()
    allowed_categories_rec = current_user_preferences.get('allowed_categories', ['vegan', 'vegetarian'])
    candidate_indices_for_rec = current_meals_df[current_meals_df['category'].isin(allowed_categories_rec)].index.tolist()
    filtered_predictions_by_category = [(i, score) for i, score in predictions_list if i in candidate_indices_for_rec]

    for idx, score in filtered_predictions_by_category:
        if len(recommendations_output) >= n_recommendations: break
        if idx in added_meal_indices: continue
        meal = current_meals_df.iloc[idx]
        # Apply calorie and protein filters
        if 'calorie_range' in current_user_preferences:
            cal_min, cal_max = current_user_preferences['calorie_range']
            if not (cal_min <= meal['calories'] <= cal_max): continue
        if 'protein_range' in current_user_preferences:
            prot_min, prot_max = current_user_preferences['protein_range']
            if not (prot_min <= meal['protein'] <= prot_max): continue
        recommendations_output.append({
            'meal_id': int(meal['meal_id']), 'name': meal['name'], 'category': meal['category'],
            'calories': int(meal['calories']), 'protein': int(meal['protein']),
            'tags': meal['tags'], 'score': float(score)
        })
        added_meal_indices.add(idx)
    return recommendations_output

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def initialize_app_globals():
    print("Initializing HiveFlow Deep Learning API (with training on startup)...")
    global meals_df, user_history_df, keras_model, num_user_features, tag_list, tag_index, num_tags
    global calorie_scaler, protein_scaler, category_encoder # ensure these are global for prepare_meal_features

    meals_df = pd.DataFrame(meals_data)
    meals_df['calories_norm'] = calorie_scaler.fit_transform(meals_df[['calories']])
    meals_df['protein_norm'] = protein_scaler.fit_transform(meals_df[['protein']])

    meal_name_to_id = {row['name']: row['meal_id'] for _, row in meals_df.iterrows()}
    current_user_history_data = []
    user_id_fixed = 1
    for day_plan in vegan_meal_plan:
        date = pd.to_datetime(day_plan['date'], format='%d.%m.%Y')
        for meal_type_slot in ['breakfast', 'lunch', 'dinner']:
            meal_name = day_plan[meal_type_slot]
            if meal_name in meal_name_to_id:
                meal_id = meal_name_to_id[meal_name]
                current_user_history_data.append({'user_id': user_id_fixed, 'meal_id': meal_id, 'timestamp': date})
            else:
                print(f"API Init Warning: Meal '{meal_name}' from vegan_meal_plan not found in meals_df.")
    user_history_df = pd.DataFrame(current_user_history_data)
    if not user_history_df.empty:
        user_history_df = user_history_df.sort_values(by='timestamp').reset_index(drop=True)

    prepare_meal_features(meals_df) # Populates tag_list, tag_index, num_tags, and global feature arrays
    num_user_features = 2 + num_tags # Set global num_user_features based on num_tags from prepare_meal_features

    print("Starting model training as part of API initialization...")
    trained_model, training_history = train_model_on_startup(user_history_df, meals_df, user_preferences) # user_preferences is global

    if trained_model:
        keras_model = trained_model
        print("Keras model trained and loaded successfully.")
        if training_history:
            print(f"Training completed. Final validation loss: {training_history.history['val_loss'][-1] if 'val_loss' in training_history.history and training_history.history['val_loss'] else 'N/A'}")
            print(f"Training completed. Final validation accuracy: {training_history.history['val_accuracy'][-1] if 'val_accuracy' in training_history.history and training_history.history['val_accuracy'] else 'N/A'}")
    else:
        print("Error: Model training failed. API might not function correctly.")
        keras_model = None

    print("Application globals initialized.")

@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'HiveFlow Deep Learning Meal Recommender API is running'})

@app.route('/smart-recommendations', methods=['GET', 'POST'])
def smart_recommend_api_endpoint():
    global keras_model, user_history_df, meals_df # user_preferences is also global

    if keras_model is None:
        return jsonify({"success": False, "error": "Keras model not available or training failed"}), 500
    if user_history_df is None or meals_df is None:
        return jsonify({"success": False, "error": "Server data (history or meals) not initialized"}), 500

    try:
        recommendations = get_recommendations_api(keras_model, user_history_df, meals_df, user_preferences, n_recommendations=5)
        return jsonify({"success": True, "recommendations": recommendations})
    except Exception as e:
        import traceback
        print(f"Error during smart recommendation: {str(e)}")
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    initialize_app_globals()

    if keras_model is None:
        print("Exiting: Keras model could not be trained/initialized. API cannot start.")
    else:
        port = int(os.environ.get("PORT", find_free_port()))
        print(f"Flask app will run on port: {port}")

        ngrok_token = os.environ.get("NGROK_AUTH_TOKEN")
        if ngrok_token:
            try:
                ngrok.set_auth_token(ngrok_token)
                public_url = ngrok.connect(port)
                print(f"Ngrok Tunnel URL: {public_url}")
                print(f"Smart recommendations endpoint: {public_url}/smart-recommendations")
            except Exception as e:
                print(f"Ngrok connection failed ({e}). Continuing with local-only mode.")
        else:
            print("NGROK_AUTH_TOKEN not set — running locally only.")
            print(f"Local endpoint: http://127.0.0.1:{port}/smart-recommendations")

        app.run(port=port, debug=False)

