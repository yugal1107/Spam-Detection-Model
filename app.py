from flask import Flask, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

# Load the trained model and the LabelEncoder object from the saved files
classifier = joblib.load('spam_classifier_model.pkl')
le = joblib.load('label_encoder.pkl')
cv= joblib.load('countvectorizer.pkl')

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spam Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background-color: #f4f4f4;
        }
        .container {
            width: 400px;
            padding: 20px;
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            text-align: center;
        }
        label {
            display: block;
            margin-bottom: 8px;
        }
        input[type="text"] {
            width: 100%;
            padding: 8px;
            margin-bottom: 16px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        button {
            width: 100%;
            padding: 10px;
            background-color: #007bff;
            color: #fff;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #0056b3;
        }
        #result {
            margin-top: 20px;
            font-weight: bold;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Spam Detection</h1>
        <label for="message">Enter your message:</label>
        <input type="text" id="message" placeholder="Type your message here...">
        <button onclick="classify()">Classify</button>
        <div id="result"></div>
    </div>

    <script>
        function classify() {
            var message = document.getElementById('message').value;
            
            // You would send the 'message' variable to your backend for classification
            // and receive the result back. For now, let's just display a dummy result.
            var result = Math.random() < 0.5 ? "Not Spam" : "Spam";

            document.getElementById('result').innerText = "Result: " + result;
        }
    </script>
</body>
</html>

    """

@app.route('/classify', methods=['POST'])
def classify_message():
    # Receive the message from the request
    message = request.form['message']

    # Preprocess the message
    sp = re.sub('[^a-zA-Z]',' ',message)
    sp = sp.lower()
    sp = sp.split()
    all_stopwords = stopwords.words('english')
    ps = PorterStemmer()
    sp = [ps.stem(word) for word in sp if not word in all_stopwords]
    sp = ' '.join(sp)

    # Use the loaded model and LabelEncoder object to make predictions
    new = [sp]
    x_new = cv.transform(new).toarray()
    prediction = classifier.predict(x_new)
    prediction_label = le.inverse_transform(prediction)[0]

    # Return the prediction result as JSON
    return jsonify({"result": prediction_label})

if __name__ == '__main__':
    app.run(debug=True)
