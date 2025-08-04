from flask import Flask, request, render_template
import pickle
from datetime import datetime

# Load the saved model and vectorizer
with open("svm_fake_news_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', current_year=datetime.now().year)

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news']
    vect_text = vectorizer.transform([news_text])
    prediction = model.predict(vect_text)[0]
    result = "Real News" if prediction == 1 else "Fake News"
    return render_template('index.html', 
                         prediction=result, 
                         text=news_text,
                         current_year=datetime.now().year)

if __name__ == "__main__":
    app.run(debug=True)