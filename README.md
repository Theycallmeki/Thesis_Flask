<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ML Behavioral Analysis Project</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
            background-color: #f9f9f9;
            color: #333;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        pre {
            background-color: #eef0f1;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
        table {
            border-collapse: collapse;
            width: 80%;
            margin-bottom: 20px;
        }
        table, th, td {
            border: 1px solid #ccc;
        }
        th, td {
            padding: 10px;
            text-align: left;
        }
        ul {
            margin-bottom: 20px;
        }
        code {
            background-color: #eef0f1;
            padding: 2px 5px;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <h1>Machine Learning Behavioral Analysis Project</h1>

    <h2>Project Overview</h2>
    <p>This project implements a machine learning pipeline to analyze customer purchase behavior and provide item recommendations. It covers:</p>
    <ul>
        <li>Data preprocessing</li>
        <li>User behavior modeling</li>
        <li>Machine learning training</li>
        <li>Item recommendation generation</li>
    </ul>
    <p>The system uses <code>raw_sales.csv</code> as input and outputs:</p>
    <ul>
        <li>Categorized sales dataset: <code>trained_sales.csv</code></li>
        <li>Machine learning model: <code>buying_frequency_model.pkl</code></li>
        <li>Label encoder: <code>label_encoder.pkl</code></li>
        <li>User behavior summary and top-selling items: <code>behavior_analysis_ml.csv</code></li>
    </ul>

    <h2>1. Data Preparation</h2>
    <ul>
        <li>Loads <code>raw_sales.csv</code>.</li>
        <li>Automatically categorizes items into predefined categories (Fruits, Vegetables, Meat, Dairy, etc.) using keyword matching.</li>
        <li>Fills uncategorized items as 'Uncategorized'.</li>
        <li>Saves the cleaned dataset as <code>trained_sales.csv</code>.</li>
    </ul>

    <h2>2. User Behavior Analysis</h2>
    <ul>
        <li>Computes per-user metrics:
            <ul>
                <li>Total purchases (<code>count</code>)</li>
                <li>Days active (<code>days_active</code>)</li>
                <li>Average purchases per week (<code>avg_per_week</code>)</li>
            </ul>
        </li>
        <li>Classifies users into frequency categories: Occasional, Monthly, Weekly, Daily</li>
        <li>Saves user behavior summary to <code>behavior_analysis_ml.csv</code></li>
    </ul>

    <h2>3. Machine Learning Model</h2>
    <ul>
        <li>Trains a <code>RandomForestClassifier</code> to predict user purchase frequency based on behavioral features.</li>
        <li>Evaluates the model:
            <ul>
                <li>Accuracy: 100%</li>
                <li>Generates a classification report</li>
            </ul>
        </li>
        <li>Saves trained model and label encoder for future use:
            <ul>
                <li><code>buying_frequency_model.pkl</code></li>
                <li><code>label_encoder.pkl</code></li>
            </ul>
        </li>
    </ul>

    <h2>4. Recommendation System</h2>
    <ul>
        <li>Builds a user-item matrix.</li>
        <li>Computes item similarity using cosine similarity.</li>
        <li>Generates top 5 similar item recommendations for a sample item.</li>
        <li>Example Output:
            <pre>
Sample item: 'canned beer'
Recommendations: ['whole milk', 'other vegetables', 'rolls/buns', 'root vegetables', 'soda']
            </pre>
        </li>
    </ul>

    <h2>5. Top-selling Items</h2>
    <p>The 10 most frequently purchased items are:</p>
    <ul>
        <li>whole milk</li>
        <li>other vegetables</li>
        <li>rolls/buns</li>
        <li>soda</li>
        <li>yogurt</li>
        <li>root vegetables</li>
        <li>tropical fruit</li>
        <li>bottled water</li>
        <li>sausage</li>
        <li>citrus fruit</li>
    </ul>

    <h2>Outputs</h2>
    <table>
        <tr>
            <th>File</th>
            <th>Description</th>
        </tr>
        <tr>
            <td>trained_sales.csv</td>
            <td>Categorized sales data ready for analysis</td>
        </tr>
        <tr>
            <td>behavior_analysis_ml.csv</td>
            <td>User behavior summary with frequency labels</td>
        </tr>
        <tr>
            <td>buying_frequency_model.pkl</td>
            <td>Trained RandomForest model</td>
        </tr>
        <tr>
            <td>label_encoder.pkl</td>
            <td>Label encoder for frequency labels</td>
        </tr>
    </table>

    <h2>Next Steps / Recommendations</h2>
    <ul>
        <li>Improve recommendations by considering item categories for more realistic suggestions.</li>
        <li>Add an interactive recommendation function for dynamic item queries.</li>
        <li>Test and validate the ML model on larger datasets to prevent overfitting.</li>
        <li>Optionally deploy as a Flask API for web-based predictions and recommendations.</li>
    </ul>

    <h2>How to Run</h2>
    <ol>
        <li>Place <code>raw_sales.csv</code> in the <code>ml/</code> folder.</li>
        <li>Run the script:
            <pre>python -m ml.analyze_behavior</pre>
        </li>
    </ol>
    <p>Outputs will be saved automatically in the <code>ml/</code> folder.</p>
</body>
</html>
