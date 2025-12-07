import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# -----------------------------------------------------------
# ORIGINAL TEAM MODEL
# -----------------------------------------------------------

real = pd.read_csv('data/True.csv')
fake = pd.read_csv('data/Fake.csv')

real['label'] = 1
fake['label'] = 0

df = pd.concat([real, fake], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    min_df=5
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("---- ORIGINAL MODEL RESULTS ----")
print(f'Accuracy: {accuracy:.4f}')
print(report)
print(conf_matrix)

with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("Original model and vectorizer saved successfully.")

# =====================================================================
# IMPROVED MODEL (ADDED WITHOUT MODIFYING ORIGINAL MODEL)
# =====================================================================
# Enhancements included:
# - Added bigrams (ngram_range=(1,2))
# - Balanced class weights for fairness
# - Increased max_iter to prevent convergence issues
# - Trains on same dataset/splits for consistency
# - Saves separately as improved_model.pkl
# =====================================================================

print("\n\n================ IMPROVED MODEL ================")

improved_vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    min_df=5,
    ngram_range=(1, 2)  # includes bigrams
)

X_train_tfidf2 = improved_vectorizer.fit_transform(X_train)
X_test_tfidf2 = improved_vectorizer.transform(X_test)

improved_model = LogisticRegression(
    max_iter=500,
    class_weight='balanced'
)

improved_model.fit(X_train_tfidf2, y_train)

y_pred2 = improved_model.predict(X_test_tfidf2)

accuracy2 = accuracy_score(y_test, y_pred2)
report2 = classification_report(y_test, y_pred2)
conf_matrix2 = confusion_matrix(y_test, y_pred2)

print("Improved Model Accuracy:", accuracy2)
print(report2)
print(conf_matrix2)

# Save improved model (keeps original model untouched)
with open('improved_model.pkl', 'wb') as f:
    pickle.dump(improved_model, f)

with open('improved_vectorizer.pkl', 'wb') as f:
    pickle.dump(improved_vectorizer, f)

print("Improved model saved as improved_model.pkl")
print("Improved vectorizer saved as improved_vectorizer.pkl")
