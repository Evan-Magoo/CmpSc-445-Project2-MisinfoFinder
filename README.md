To regenerate both the original model and the improved model, run:
python3 model.py
This will automatically create the following files:

model.pkl                 # original model
vectorizer.pkl            # original TF-IDF vectorizer
improved_model.pkl        # improved model (bigrams + balanced weights)
improved_vectorizer.pkl   # improved TF-IDF vectorizer
