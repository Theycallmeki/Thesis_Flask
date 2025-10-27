Project Overview

This project implements a machine learning pipeline to analyze customer purchase behavior and provide item recommendations.
It covers:

Data preprocessing

User behavior modeling

Machine learning training

Item recommendation generation

The system uses raw_sales.csv as input and outputs:

Categorized sales dataset (trained_sales.csv)

Machine learning model (buying_frequency_model.pkl)

Label encoder (label_encoder.pkl)

User behavior summary and top-selling items (behavior_analysis_ml.csv)

1. Data Preparation

Loads raw_sales.csv.

Automatically categorizes items into predefined categories (Fruits, Vegetables, Meat, Dairy, etc.) using keyword matching.

Fills uncategorized items as 'Uncategorized'.

Saves the cleaned dataset as trained_sales.csv.

2. User Behavior Analysis

Computes per-user metrics:

Total purchases (count)

Days active (days_active)

Average purchases per week (avg_per_week)

Classifies users into frequency categories:

Occasional, Monthly, Weekly, Daily

Saves user behavior summary to behavior_analysis_ml.csv.

3. Machine Learning Model

Trains a RandomForestClassifier to predict user purchase frequency based on behavioral features.

Evaluates the model:

Accuracy: 100%

Generates a classification report

Saves trained model and label encoder for future use:

buying_frequency_model.pkl

label_encoder.pkl

4. Recommendation System

Builds a user-item matrix.

Computes item similarity using cosine similarity.

Generates top 5 similar item recommendations for a sample item.

Example Output:

Sample item: 'canned beer'
Recommendations: ['whole milk', 'other vegetables', 'rolls/buns', 'root vegetables', 'soda']

5. Top-selling Items

The 10 most frequently purchased items are:

whole milk
other vegetables
rolls/buns
soda
yogurt
root vegetables
tropical fruit
bottled water
sausage
citrus fruit

Outputs
File	Description
trained_sales.csv	Categorized sales data ready for analysis
behavior_analysis_ml.csv	User behavior summary with frequency labels
buying_frequency_model.pkl	Trained RandomForest model
label_encoder.pkl	Label encoder for frequency labels
Next Steps / Recommendations

Improve recommendations by considering item categories for more realistic suggestions.

Add an interactive recommendation function for dynamic item queries.

Test and validate the ML model on larger datasets to prevent overfitting.

Optionally deploy as a Flask API for web-based predictions and recommendations.

How to Run

Place raw_sales.csv in the ml/ folder.

Run the script:

python -m ml.analyze_behavior


Outputs will be saved automatically in the ml/ folder.
