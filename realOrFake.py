import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import pickle

# 1. Load datasets
df_fake = pd.read_csv("Datasets/Fake.csv")
df_true = pd.read_csv("Datasets/True.csv")

# 2. Add labels
df_fake["label"] = 0  # Fake
df_true["label"] = 1  # Real

# 3. Combine and shuffle
df = pd.concat([df_fake, df_true], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# 4. Features and labels
X = df['text']
y = df['label']

# 5. Convert text to TF-IDF features
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_tfidf = tfidf.fit_transform(X)

# 6. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# 7. Train SVM model
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)

# 8. Evaluate
y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 9. Save model and vectorizer
with open("svm_fake_news_model.pkl", "wb") as model_file:
    pickle.dump(svm_model, model_file)

with open("tfidf_vectorizer.pkl", "wb") as vec_file:
    pickle.dump(tfidf, vec_file)

print("Model and vectorizer saved.")
