import tkinter as tk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from tkinter import scrolledtext
from tkinter import messagebox
from PIL import Image, ImageTk
from newspaper import Article
from wordcloud import WordCloud
import pickle

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# --- Extracted Data for Analysis ---

X_train_tfidf = pickle.load(open('X_train_tfidf.pkl', 'rb'))
tfidf_matrix = X_train_tfidf.toarray()
feature_names = vectorizer.get_feature_names_out()
weights = model.coef_[0]
avg_tfidf = np.mean(tfidf_matrix, axis=0)
max_tfidf = np.max(tfidf_matrix, axis=0)
idf_values = vectorizer.idf_
avg_tf = avg_tfidf / idf_values

binary_matrix = (tfidf_matrix > 0).astype(int)
doc_freq = np.sum(binary_matrix, axis=0)

# --- GUI Setup ---

window = tk.Tk()
window.title('Abington Evacuation Tool')
window.geometry('900x700')
window.configure(bg='#001E44')
logo = tk.PhotoImage(file="favicon.png")
window.iconphoto(False, logo)

# --- Navigation Bar ---

def navigation_bar():
    screen_selection = tk.Frame(window, bg="#001E44")
    screen_selection.pack(pady=10, side=tk.BOTTOM)

    fake_news_button = tk.Button(
        screen_selection,
        text="Fake News Detection",
        width=20,
        bg="#3b5998",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#96BEE6",
        activeforeground="white",
        highlightthickness=0,
        command=fake_news_screen
    )

    misinfo_button = tk.Button(
        screen_selection,
        text="Misinfo Analysis",
        width=20,
        bg="#3b5998",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#96BEE6",
        activeforeground="white",
        highlightthickness=0,
        command=misinfo_screen
    )

    article_word_cloud_button = tk.Button(
        screen_selection,
        text="Article Word Cloud",
        width=20,
        bg="#3b5998",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#96BEE6",
        activeforeground="white",
        highlightthickness=0,
        command=article_word_cloud_screen
    )

    article_summarizer_button = tk.Button(
        screen_selection,
        text="Article Summarizer",
        width=20,
        bg="#3b5998",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#96BEE6",
        activeforeground="white",
        highlightthickness=0,
        command=article_summarizer_screen
    )

    fake_news_button.grid(row=0, column=0, padx=10, pady=10)
    misinfo_button.grid(row=0, column=1, padx=10, pady=10)
    article_word_cloud_button.grid(row=0, column=2, padx=10, pady=10)
    article_summarizer_button.grid(row=0, column=3, padx=10, pady=10)

def predict_news(link, widget):
    widget.delete(1.0, tk.END)

    try:
        article = Article(link)
        article.download()
        article.parse()
        text = article.text

        vec = vectorizer.transform([text])
        prediction = model.predict(vec)[0]
        result = "REAL" if prediction == 1 else "Likely FAKE \n\nThis means the article contains several indicators commonly found in fake news."
        contributions = vec.toarray()[0] * model.coef_[0]

        feature_names = vectorizer.get_feature_names_out()

        word_contributions = list(zip(feature_names, contributions))
        word_contribs_sorted = sorted(word_contributions, key=lambda x: x[1], reverse=True)

        top_fake_words = [f"{word}: {contrib:.4f}" for word, contrib in word_contribs_sorted[:10]]
        top_real_words = [f"{word}: {contrib:.4f}" for word, contrib in word_contribs_sorted[-10:]]

        widget.insert(tk.END, f"Prediction: {result}\n")
        widget.insert(tk.END, "\n---- Top Fake-Influencing Words ----\n", "header")

        for word in top_fake_words:
            widget.insert(tk.END, word + "\n")
        widget.insert(tk.END, "\n---- Top Real-Influencing Words ----\n", "header")

        for word in top_real_words:
            widget.insert(tk.END, word + "\n")

        widget.insert(tk.END, "\n---- Article Text ----\n\n")
        widget.insert(tk.END, text)

    except Exception as e:
        widget.insert(tk.END, f"Error fetching article:\n{e}")



def summarize_article(link, widget):
    widget.delete(1.0, tk.END)




def clear_screen():
    for widget in window.winfo_children():
        widget.destroy()

def fake_news_screen():
    global output_box
    clear_screen()

    title_label = tk.Label(
        window,
        text="Fake News Detection",
        font=("Arial", 20, "bold"),
        fg="white",
        bg="#001E44"
    )
    title_label.pack(pady=10)

    link = tk.StringVar()

    controls = tk.Frame(window, bg="#001E44")
    controls.pack(pady=10)

    entry_label = tk.Label(
        controls, 
        text='Article Link:', 
        font=('Arial', 12, 'bold'),
        fg='white',
        background='#001E44'
    )

    entry = tk.Entry(
        controls,
        textvariable=link,
        width = 50,
        font=("Arial", 12)
    )

    entry_button = tk.Button(
        controls,
        text="Enter",
        width=12,
        bg="#3b5998",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#96BEE6",
        activeforeground="white",
        highlightthickness=0,
        command=lambda: predict_news(entry.get())
    )

    entry_label.grid(row=0, column=0, padx=5)
    entry.grid(row=0, column=1, padx=5)
    entry_button.grid(row=0, column=2, padx=5)

    output_box = scrolledtext.ScrolledText(
        window,
        wrap=tk.WORD,
        width=90,
        height=25,
        font=("Arial", 12)
    )
    output_box.pack(pady=20)

    navigation_bar()
 
def misinfo_screen():
    clear_screen()
    title_label = tk.Label(
        window,
        text="Misinformation Analysis",
        font=("Arial", 20, "bold"),
        fg="white",
        bg="#001E44"
    )
    title_label.pack(pady=5)

    misinfo_label = tk.Label(
        window,
        text="Top 50 Misinformation-Influencing Words",
        font=("Arial", 14, "bold"),
        fg="white",
        bg="#001E44"
    )
    misinfo_label.pack(pady=5)

    misinfo_box = scrolledtext.ScrolledText(
        window,
        wrap=tk.WORD,
        width=92,
        height=12,
        font=("Consolas", 12)
    )
    misinfo_box.pack(pady=5)

    real_label = tk.Label(
        window,
        text="Top 50 Real-Influencing Words ",
        font=("Arial", 14, "bold"),
        fg="white",
        bg="#001E44"
    )
    real_label.pack(pady=5)

    real_box = scrolledtext.ScrolledText(
        window,
        wrap=tk.WORD,
        width=92,
        height=12,
        font=("Consolas", 12)
    )
    real_box.pack(pady=5)

    # --- Misinfo Words ---
    indices = np.argsort(weights)[:50]
    misinfo_box.insert(tk.END, "  # | Word                 | Weight  | Avg TF-IDF | Max TF-IDF | Avg TF | IDF    | Doc Freq \n")
    misinfo_box.insert(tk.END, "----+----------------------+---------+------------+------------+--------+--------+----------\n")

    for index, i in enumerate(indices):
        misinfo_box.insert(
            tk.END, 
            f"{index+1:>3} | {feature_names[i]:<20} | {weights[i]:<7.4f} | {avg_tfidf[i]:<10.4f} | {max_tfidf[i]:<10.4f} | {avg_tf[i]:<5.4f} | {idf_values[i]:<5.4f} | {doc_freq[i]:<9} \n"
        )

    # --- Real Words ---
    indices = np.argsort(weights)[-50:][::-1]
    real_box.insert(tk.END, "  # | Word                 | Weight  | Avg TF-IDF | Max TF-IDF | Avg TF | IDF    | Doc Freq \n")
    real_box.insert(tk.END, "----+----------------------+---------+------------+------------+--------+--------+----------\n")
    for index, i in enumerate(indices):
        real_box.insert(
            tk.END,
            f"{index+1:>3} | {feature_names[i]:<20} | {weights[i]:<7.4f} | {avg_tfidf[i]:<10.4f} | {max_tfidf[i]:<10.4f} | {avg_tf[i]:<5.4f} | {idf_values[i]:<5.4f} | {doc_freq[i]:<9} \n"
        )

    navigation_bar()

def article_word_cloud_screen():
    clear_screen()
    title_label = tk.Label(
        window,
        text="Article Word Cloud",
        font=("Arial", 20, "bold"),
        fg="white",
        bg="#001E44"
    )
    title_label.pack(pady=10)

    link = tk.StringVar()

    controls = tk.Frame(window, bg="#001E44")
    controls.pack(pady=10)

    entry_label = tk.Label(
        controls, 
        text='Article Link:', 
        font=('Arial', 12, 'bold'),
        fg='white',
        background='#001E44'
    )

    entry = tk.Entry(
        controls,
        textvariable=link,
        width = 50,
        font=("Arial", 11)
    )

    entry_label.grid(row=0, column=0, padx=5)
    entry.grid(row=0, column=1, padx=5)

    def generate_word_cloud():
        try:
            article = Article(link.get())
            article.download()
            article.parse()
            text = article.text
            wc = WordCloud(width=827, height=450, background_color='white').generate(text)
            img = wc.to_image()
            img_tk = ImageTk.PhotoImage(img)

            if hasattr(article_word_cloud_screen, 'img_label') and article_word_cloud_screen.img_label.winfo_exists():
                article_word_cloud_screen.img_label.configure(image=img_tk)
                article_word_cloud_screen.img_label.image = img_tk
            else:
                article_word_cloud_screen.img_label = tk.Label(window, image=img_tk)
                article_word_cloud_screen.img_label.image = img_tk
                article_word_cloud_screen.img_label.pack(pady=20)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    generate_button = tk.Button(
        controls,  
        text="Generate Word Cloud",
        width=20,
        bg="#3b5998",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#96BEE6",
        activeforeground="white",
        highlightthickness=0,
        command=generate_word_cloud
    )
    generate_button.grid(row=0, column=2, padx=5)

    navigation_bar()

def article_summarizer_screen():
    global summary_box
    clear_screen()

    title_label = tk.Label(
        window,
        text="Article Summarizer",
        font=("Arial", 20, "bold"),
        fg="white",
        bg="#001E44"
    )
    title_label.pack(pady=10)

    link = tk.StringVar()

    controls = tk.Frame(window, bg="#001E44")
    controls.pack(pady=10)

    entry_label = tk.Label(
        controls, 
        text='Article Link:', 
        font=('Arial', 12, 'bold'),
        fg='white',
        background='#001E44'
    )

    entry = tk.Entry(
        controls,
        textvariable=link,
        width = 50,
        font=("Arial", 12)
    )

    entry_button = tk.Button(
        controls,
        text="Summarize",
        width=12,
        bg="#3b5998",
        fg="white",
        font=("Arial", 11, "bold"),
        activebackground="#96BEE6",
        activeforeground="white",
        highlightthickness=0,
        command=lambda: summarize_article(entry.get(), summary_box)
    )

    entry_label.grid(row=0, column=0, padx=5)
    entry.grid(row=0, column=1, padx=5)
    entry_button.grid(row=0, column=2, padx=5)

    summary_box = scrolledtext.ScrolledText(
        window,
        wrap=tk.WORD,
        width=90,
        height=25,
        font=("Arial", 12)
    )
    summary_box.pack(pady=20)
        
    navigation_bar()


if __name__ == "__main__":
    clear_screen()
    fake_news_screen()
    window.mainloop()