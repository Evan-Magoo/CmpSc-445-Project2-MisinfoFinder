import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import learning_curve
import pickle
import matplotlib.pyplot as plt

count = 0 # For tracking article downloads

real = pd.read_csv('data/True.csv')
fake = pd.read_csv('data/Fake.csv')

real['label'] = 1
fake['label'] = 0

df = pd.concat([real, fake], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

df = df.fillna(' ')

df['content'] = df['text'] + ' ' + df['title']

X = df['content']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    min_df=0.01,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(
    max_iter=500,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("---- MODEL RESULTS ----")
print(f'Accuracy: {accuracy:.4f}')
print(report)
print(conf_matrix)

with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

with open('X_train_tfidf.pkl', 'wb') as f:
    pickle.dump(X_train_tfidf, f)

print("Original model and vectorizer saved successfully.")

train_sizes, train_scores, test_scores = learning_curve(
    model, X_train_tfidf, y_train,
    cv=5, scoring='accuracy', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5)
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes, train_mean, label='Train Accuracy')
plt.plot(train_sizes, test_mean, label='Validation Accuracy')
plt.xlabel('Training Size')
plt.ylim(0, 1.0)
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.show()


