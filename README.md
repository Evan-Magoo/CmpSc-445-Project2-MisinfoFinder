To regenerate both the original model and the improved model, run:
python3 model.py
This will automatically create the following files:

model.pkl                 # classification model
vectorizer.pkl            # TF-IDF vectorizer
X_train_tfidf.pkl         # TF-IDF sparse matrix for later analysis
fake_news_app.py          # Tkinter application providing misinfo tools
model.py                  # Contains preprocessing, classification model training, and model evaluations
Fake.csv                  # CSV of articles deemed fake or misinfo
True.csv                  # CSV of articles deemed real
favicon.png               # Program icon
CmpSc 445 - Report 2.pdf  # Project report

Install all files keeping their file structure.

Run fake_news_app.py for the main program.

Run model.py to generate another model. This isn't require as the models are already provided.


