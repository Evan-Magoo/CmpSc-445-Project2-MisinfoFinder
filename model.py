import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# --- Data Loading ---
real = pd.read_csv('data/True.csv')
fake = pd.read_csv('data/Fake.csv')

real['label'] = 1
fake['label'] = 0

df = pd.concat([real, fake], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# --- Train-Test Split ---
X = df['text']
y = df['label'] 

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42
)

# --- TF-IDf Vectorization ---
vectorizer = TfidfVectorizer(
    stop_words='english', 
    max_df=0.7,
    min_df=5
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test) 

# --- Model Training ---
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# --- Model Evaluation ---
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print('Classification Report:', report)
print('Confusion Matrix:', conf_matrix)

# --- Save Model and Vectorizer ---
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("Model and vectorizer saved successfully.")
