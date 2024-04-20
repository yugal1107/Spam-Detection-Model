from flask import Flask, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

app = Flask(__name__)

# Load the trained model and the LabelEncoder object from the saved files
classifier = joblib.load('spam_classifier_model.pkl')
le = joblib.load('label_encoder.pkl')
cv= joblib.load('countvectorizer.pkl')

def classify_message(message):
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
    prediction_label = le.inverse_transform(prediction)

    return prediction_label

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
            background-image: url(https://th.bing.com/th/id/OIG2.V9eyzs9bLA998aHUOjkO?pid=ImgGn);
            background-position: center;
            background-repeat: no-repeat;
            background-size: cover;   }
        .container {
            display: flex;
            flex-direction: column;
            gap: 10px;
            width: 400px;
            padding: 30px;
            background-color: rgba(255, 255, 255, .15);  
            backdrop-filter: blur(10px);
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            text-align: center;
        }
        label {
            display: block;
            align-items: center;
            justify-content: center;
            margin-bottom: 8px;
        }
        input[type="text"] {
            padding: 8px;
            margin-bottom: 16px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        button {
            align-items: center;
            justify-content: center;
            padding: 10px;
            display: inline-block;
            background-color: #007bff;
            color: #fff;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            width: 25%;
            align-self: center;
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
        <button onclick="classify()">Predict</button>
        <div id="result"></div>
    </div>

    <script>
        function classify() {
            var message = document.getElementById('message').value;
            
            // Send the message to the backend for classification
            fetch('/classify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: message })
            })
            .then(response => response.json())
            .then(data => {
                // Display the classification result
                document.getElementById('result').innerText = "Result: " + data.result;

                // Hide the result after 5 seconds
                    setTimeout(function() {
                        document.getElementById('result').innerText = '';
                    }, 5000);
            })
            .catch(error => {
                console.error('Error:', error);
            });
        }
    </script>
</body>
</html>

    """

@app.route('/classify', methods=['POST'])
def classify():
    message = request.json['message']
    prediction_label = classify_message(message)
    if prediction_label[0]== 'ham':
        x= 'Not Spam'
    else:
        x='Spam'
    return jsonify({"result": x})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, port=port)
