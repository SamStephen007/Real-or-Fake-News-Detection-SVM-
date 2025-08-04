# 📰 Fake News Detector using SVM

This project is a machine learning web application that classifies news content as **Real** or **Fake** using a Support Vector Machine (SVM) classifier. It uses text data from news articles and leverages TF-IDF for feature extraction.

---

## 🚀 Features

- Classifies text-based news as Real or Fake
- SVM model trained using TF-IDF features
- Flask web interface for real-time predictions
- Simple and clean HTML UI

---

## 📂 Project Structure

```
fake_news_app/
│
├── Datasets/
│   └── Fake.csv
|   └── True.csv
├── svm_fake_news_model.pkl          # Trained SVM model
├── tfidf_vectorizer.pkl             # TF-IDF vectorizer used during training
├── app.py                           # Flask app
├── templates/
│   └── index.html                   # Frontend HTML form
└── README.md                        # Project documentation
```

---

## ⚙️ Requirements

- Python 3.7+
- Flask
- scikit-learn
- pandas

Install using:

```bash
pip install -r requirements.txt
```

---

## 🧪 How to Run

1. Clone or download this project.
2. Ensure you have the following files:
   - `svm_fake_news_model.pkl`
   - `tfidf_vectorizer.pkl`
3. Run the Flask app:
   ```bash
   python app.py
   ```
4. Open your browser and go to [http://localhost:5000](http://localhost:5000)

---

## 📊 Model Training

The SVM model was trained on a balanced dataset of real and fake news articles using TF-IDF for feature extraction:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
```

---

## ✨ Example Inputs

**Real:**
> The Senate passed a $1.2 trillion infrastructure bill on Tuesday.

**Fake:**
> BREAKING: Scientists discover banana cures COVID-19!

---
## Images

![Fake News Prediction](Images/(image.png))
![Fake News Prediction](Images/(image-1.png))

## 🧠 Future Improvements

- Use more advanced models like BERT or LSTM
- Add headline + text combination as input
- Provide confidence scores in predictions
- Deploy with Docker or Streamlit

---

## 📬 Author

**Sam Stephen**  
GitHub: [@SamStephen007](https://github.com/SamStephen007)